"""OpenAI-compatible FastAPI proxy server."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse

from tokenwise.config import get_settings
from tokenwise.executor import _TIER_STRENGTH
from tokenwise.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    EscalationPolicy,
    ModelTier,
    Usage,
)
from tokenwise.providers import ProviderResolver
from tokenwise.registry import ModelRegistry
from tokenwise.risk_gate import evaluate_risk
from tokenwise.router import Router

logger = logging.getLogger(__name__)

# HTTP status codes that indicate the model is unusable (should try a different model)
_MODEL_UNUSABLE_CODES = {402, 403, 404}

# HTTP status codes that indicate a transient upstream issue (retry same model once)
_TRANSIENT_CODES = {500, 502, 503, 504}

_MAX_RETRIES = 3

_FAILED_MODELS_TTL = 300.0  # seconds
_FAILED_MODELS_MAX_SIZE = 50


class _FailedModels:
    """Set-like container that expires entries after *ttl* seconds."""

    def __init__(
        self, ttl: float = _FAILED_MODELS_TTL, max_size: int = _FAILED_MODELS_MAX_SIZE
    ) -> None:
        self._ttl = ttl
        self._max_size = max_size
        self._entries: dict[str, float] = {}  # model_id → monotonic timestamp

    def _evict_stale(self) -> None:
        now = time.monotonic()
        self._entries = {k: v for k, v in self._entries.items() if now - v < self._ttl}

    def add(self, model_id: str) -> None:
        self._evict_stale()
        self._entries[model_id] = time.monotonic()
        # Enforce max size by removing oldest entries
        while len(self._entries) > self._max_size:
            oldest = min(self._entries, key=self._entries.get)  # type: ignore[arg-type]
            del self._entries[oldest]

    def __contains__(self, model_id: object) -> bool:
        self._evict_stale()
        return model_id in self._entries

    def to_set(self) -> set[str]:
        self._evict_stale()
        return set(self._entries)


class _State:
    """Mutable application state managed by the lifespan."""

    http_client: httpx.AsyncClient
    registry: ModelRegistry
    router: Router
    resolver: ProviderResolver
    failed_models: _FailedModels


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle — create/destroy shared HTTP client."""
    state.http_client = httpx.AsyncClient(timeout=120.0)
    state.registry = ModelRegistry()
    state.router = Router(state.registry)
    state.resolver = ProviderResolver(http_client=state.http_client)
    state.failed_models = _FailedModels()
    yield
    await state.http_client.aclose()


app = FastAPI(
    title="TokenWise Proxy",
    description="OpenAI-compatible proxy that routes requests to optimal LLM models",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """List available models (OpenAI-compatible)."""
    state.registry.ensure_loaded()
    models = state.registry.list_all()
    return {
        "object": "list",
        "data": [
            {
                "id": m.id,
                "object": "model",
                "created": 0,
                "owned_by": m.provider,
            }
            for m in models
        ],
    }


async def _forward_to_upstream(
    model_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Forward a request to the resolved LLM provider. Returns parsed JSON."""
    provider, provider_model = state.resolver.resolve(model_id)
    messages = payload.get("messages", [])
    return await provider.achat_completion(
        model=provider_model,
        messages=messages,
        temperature=payload.get("temperature"),
        max_tokens=payload.get("max_tokens"),
    )


def _get_fallback_models(exclude: set[str], failed_model_id: str | None = None) -> list[str]:
    """Get fallback model IDs, escalating from the failed model's tier upward.

    Tries stronger tiers first (FLAGSHIP, then MID), then same tier.
    Filters by the failed model's capabilities when possible.
    Falls back to BUDGET+MID if the failed model is unknown.
    """
    state.registry.ensure_loaded()
    failed_model = state.registry.get_model(failed_model_id) if failed_model_id else None
    failed_tier = failed_model.tier if failed_model else ModelTier.BUDGET

    # Determine required capability from the failed model
    required_cap: str | None = None
    if failed_model and failed_model.capabilities:
        for cap in failed_model.capabilities:
            if cap != "general":
                required_cap = cap
                break

    min_strength = _TIER_STRENGTH.get(failed_tier, 0)

    # Monotonic mode: only allow strictly stronger tiers
    try:
        policy = EscalationPolicy(get_settings().escalation_policy)
    except ValueError:
        policy = EscalationPolicy.FLEXIBLE
    strictly_greater = policy == EscalationPolicy.MONOTONIC

    # Collect tiers in descending strength order (stronger first)
    candidates: list[str] = []
    for tier in [ModelTier.FLAGSHIP, ModelTier.MID, ModelTier.BUDGET]:
        tier_strength = _TIER_STRENGTH[tier]
        if strictly_greater and tier_strength <= min_strength:
            continue
        if not strictly_greater and tier_strength < min_strength:
            continue
        for m in state.registry.find_models(tier=tier):
            if m.id in exclude or m.input_price <= 0:
                continue
            if required_cap and required_cap not in m.capabilities:
                continue
            candidates.append(m.id)

    # If capability filtering eliminated everything, relax and try without it
    if not candidates and required_cap:
        for tier in [ModelTier.FLAGSHIP, ModelTier.MID, ModelTier.BUDGET]:
            tier_strength = _TIER_STRENGTH[tier]
            if strictly_greater and tier_strength <= min_strength:
                continue
            if not strictly_greater and tier_strength < min_strength:
                continue
            for m in state.registry.find_models(tier=tier):
                if m.id not in exclude and m.input_price > 0:
                    candidates.append(m.id)

    return candidates


def _resolve_model_and_payload(
    request: ChatCompletionRequest,
) -> tuple[str, bool, dict[str, Any]]:
    """Resolve model ID and build payload."""
    settings = get_settings()

    is_auto = request.model == "auto" or request.model.startswith("tokenwise/")

    if is_auto:
        last_message = request.messages[-1].content or "" if request.messages else ""
        strategy = request.tokenwise_opts.get("strategy", settings.default_strategy)
        budget = request.tokenwise_opts.get("budget", settings.default_budget)

        try:
            model = state.router.route(
                query=last_message,
                strategy=strategy,
                budget=budget,
            )
            model_id = model.id
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Routing failed: {e}")
    else:
        model_id = request.model

    payload: dict[str, Any] = {
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens

    return model_id, is_auto, payload


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """Handle chat completion requests with intelligent routing."""
    # Risk gate check
    settings = get_settings()
    if settings.risk_gate.enabled:
        last_msg = request.messages[-1].content or "" if request.messages else ""
        risk_result = evaluate_risk(last_msg, settings.risk_gate)
        if risk_result.blocked:
            raise HTTPException(status_code=422, detail=f"Risk gate blocked: {risk_result.reason}")

    model_id, is_auto, payload = _resolve_model_and_payload(request)

    if request.stream:
        return await _handle_streaming(model_id, is_auto, payload)

    # --- non-streaming path ---
    tried: set[str] = state.failed_models.to_set()
    last_error: Exception | None = None

    models_to_try = [model_id]
    if is_auto:
        models_to_try.extend(
            mid
            for mid in _get_fallback_models(tried | {model_id}, failed_model_id=model_id)[
                :_MAX_RETRIES
            ]
        )

    for mid in models_to_try:
        if mid in tried:
            continue
        tried.add(mid)
        try:
            data = await _forward_to_upstream(mid, payload)
            model_id = mid
            break
        except httpx.HTTPStatusError as e:
            last_error = e
            code = e.response.status_code
            if code in _MODEL_UNUSABLE_CODES:
                state.failed_models.add(mid)
                logger.info("Model %s returned %d (unusable), trying next", mid, code)
                continue
            if code in _TRANSIENT_CODES:
                logger.info("Model %s returned %d (transient), trying next", mid, code)
                continue
            # 400, 422, or other client errors — hard failure, stop retrying
            raise HTTPException(status_code=code, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}")
    else:
        if isinstance(last_error, httpx.HTTPStatusError):
            raise HTTPException(status_code=last_error.response.status_code, detail=str(last_error))
        raise HTTPException(status_code=502, detail="All models failed")

    choices = []
    for i, choice in enumerate(data.get("choices", [])):
        msg = choice.get("message", {})
        choices.append(
            ChatCompletionChoice(
                index=i,
                message=ChatMessage(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content", ""),
                ),
                finish_reason=choice.get("finish_reason", "stop"),
            )
        )

    usage_data = data.get("usage", {})
    usage = Usage(
        prompt_tokens=usage_data.get("prompt_tokens", 0),
        completion_tokens=usage_data.get("completion_tokens", 0),
        total_tokens=usage_data.get("total_tokens", 0),
    )

    trace_dict = None
    if is_auto:
        models_tried_list = [mid for mid in models_to_try if mid in tried]
        trace_dict = {
            "request_id": uuid.uuid4().hex[:12],
            "initial_model": models_to_try[0] if models_to_try else model_id,
            "final_model": model_id,
            "termination_state": "completed",
            "models_tried": models_tried_list,
        }

    return ChatCompletionResponse(
        id=f"tw-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model_id,
        choices=choices,
        usage=usage,
        tokenwise_trace=trace_dict,
    )


async def _handle_streaming(
    model_id: str,
    is_auto: bool,
    payload: dict[str, Any],
) -> StreamingResponse:
    """Forward a streaming request via the provider abstraction."""
    tried: set[str] = state.failed_models.to_set()
    models_to_try = [model_id]
    if is_auto:
        models_to_try.extend(
            mid
            for mid in _get_fallback_models(tried | {model_id}, failed_model_id=model_id)[
                :_MAX_RETRIES
            ]
        )

    last_error: Exception | None = None
    messages = payload.get("messages", [])

    for mid in models_to_try:
        if mid in tried:
            continue
        tried.add(mid)

        try:
            provider, provider_model = state.resolver.resolve(mid)
            sse_gen = provider.astream_completion(
                model=provider_model,
                messages=messages,
                temperature=payload.get("temperature"),
                max_tokens=payload.get("max_tokens"),
            )

            # Wrap the provider's async generator to add SSE framing
            async def _generate(gen: AsyncIterator[str]) -> AsyncIterator[str]:
                async for line in gen:
                    if line:
                        yield f"{line}\n\n"
                        if line.strip() == "data: [DONE]":
                            break

            return StreamingResponse(
                _generate(sse_gen),
                media_type="text/event-stream",
            )
        except httpx.HTTPStatusError as exc:
            last_error = exc
            code = exc.response.status_code
            if code in _MODEL_UNUSABLE_CODES:
                state.failed_models.add(mid)
                logger.info("Model %s returned %d (unusable), trying next", mid, code)
                continue
            if code in _TRANSIENT_CODES:
                logger.info("Model %s returned %d (transient), trying next", mid, code)
                continue
            # 400, 422, or other client errors — hard failure
            raise HTTPException(status_code=code, detail=str(exc))
        except HTTPException:
            raise
        except Exception as e:
            last_error = e
            logger.info("Model %s failed: %s, trying next", mid, e)
            continue

    if isinstance(last_error, httpx.HTTPStatusError):
        raise HTTPException(
            status_code=last_error.response.status_code,
            detail=str(last_error),
        )
    raise HTTPException(status_code=502, detail="All models failed")


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    from tokenwise import __version__

    return {"status": "ok", "version": __version__}
