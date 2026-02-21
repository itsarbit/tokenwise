"""OpenAI-compatible FastAPI proxy server."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse

from tokenwise.config import get_settings
from tokenwise.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelTier,
    Usage,
)
from tokenwise.providers import ProviderResolver
from tokenwise.registry import ModelRegistry
from tokenwise.router import Router

logger = logging.getLogger(__name__)

# HTTP status codes that indicate the model is unusable (should try another)
_RETRYABLE_CODES = {400, 402, 403, 404, 422}

_MAX_RETRIES = 3


class _State:
    """Mutable application state managed by the lifespan."""

    http_client: httpx.AsyncClient
    registry: ModelRegistry
    router: Router
    resolver: ProviderResolver
    failed_models: set[str]


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle — create/destroy shared HTTP client."""
    state.http_client = httpx.AsyncClient(timeout=120.0)
    state.registry = ModelRegistry()
    state.router = Router(state.registry)
    state.resolver = ProviderResolver()
    state.failed_models = set()
    yield
    await state.http_client.aclose()


app = FastAPI(
    title="TokenWise Proxy",
    description="OpenAI-compatible proxy that routes requests to optimal LLM models",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/v1/models")
async def list_models() -> dict:
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
    payload: dict,
) -> dict:
    """Forward a request to the resolved LLM provider. Returns parsed JSON."""
    provider, provider_model = state.resolver.resolve(model_id)
    messages = payload.get("messages", [])
    return await provider.achat_completion(
        model=provider_model,
        messages=messages,
        temperature=payload.get("temperature"),
        max_tokens=payload.get("max_tokens"),
    )


def _get_fallback_models(exclude: set[str]) -> list[str]:
    """Get fallback model IDs, preferring budget tier for broad accessibility."""
    state.registry.ensure_loaded()
    candidates = []
    for tier in [ModelTier.BUDGET, ModelTier.MID]:
        for m in state.registry.find_models(tier=tier):
            if m.id not in exclude and m.input_price > 0:
                candidates.append(m.id)
    return candidates


def _resolve_model_and_payload(
    request: ChatCompletionRequest,
) -> tuple[str, bool, dict]:
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

    payload: dict = {
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
    model_id, is_auto, payload = _resolve_model_and_payload(request)

    if request.stream:
        return await _handle_streaming(model_id, is_auto, payload)

    # --- non-streaming path ---
    tried: set[str] = set(state.failed_models)
    last_error: Exception | None = None

    models_to_try = [model_id]
    if is_auto:
        models_to_try.extend(
            mid for mid in _get_fallback_models(tried | {model_id})[:_MAX_RETRIES]
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
            if e.response.status_code in _RETRYABLE_CODES:
                state.failed_models.add(mid)
                logger.info("Model %s returned %d, trying next", mid, e.response.status_code)
                continue
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
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

    return ChatCompletionResponse(
        id=f"tw-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model_id,
        choices=choices,
        usage=usage,
    )


async def _handle_streaming(
    model_id: str,
    is_auto: bool,
    payload: dict,
) -> StreamingResponse:
    """Forward a streaming request via the provider abstraction."""
    tried: set[str] = set(state.failed_models)
    models_to_try = [model_id]
    if is_auto:
        models_to_try.extend(
            mid for mid in _get_fallback_models(tried | {model_id})[:_MAX_RETRIES]
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
            async def _generate(gen):  # type: ignore[no-untyped-def]  # noqa: ANN202
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
            if exc.response.status_code in _RETRYABLE_CODES:
                state.failed_models.add(mid)
                logger.info(
                    "Model %s returned %d, trying next",
                    mid, exc.response.status_code,
                )
                continue
            raise HTTPException(
                status_code=exc.response.status_code, detail=str(exc),
            )
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
async def health() -> dict:
    """Health check endpoint."""
    from tokenwise import __version__

    return {"status": "ok", "version": __version__}
