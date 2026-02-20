"""OpenAI-compatible FastAPI proxy server."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException

from tokenwise.config import get_settings
from tokenwise.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelTier,
    Usage,
)
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
    failed_models: set[str]


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle — create/destroy shared HTTP client."""
    state.http_client = httpx.AsyncClient(timeout=120.0)
    state.registry = ModelRegistry()
    state.router = Router(state.registry)
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
    headers: dict[str, str],
) -> dict:
    """Forward a request to the upstream LLM provider. Returns parsed JSON."""
    settings = get_settings()
    payload["model"] = model_id

    resp = await state.http_client.post(
        f"{settings.openrouter_base_url}/chat/completions",
        headers=headers,
        json=payload,
    )
    resp.raise_for_status()
    return resp.json()


def _get_fallback_models(exclude: set[str]) -> list[str]:
    """Get fallback model IDs, preferring budget tier for broad accessibility."""
    state.registry.ensure_loaded()
    candidates = []
    for tier in [ModelTier.BUDGET, ModelTier.MID]:
        for m in state.registry.find_models(tier=tier):
            if m.id not in exclude and m.input_price > 0:
                candidates.append(m.id)
    return candidates


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Handle chat completion requests with intelligent routing."""
    settings = get_settings()

    # Streaming is not yet supported
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not supported yet. Set stream=false.",
        )

    # Determine which model to use
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

    # Build upstream request
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.openrouter_api_key:
        headers["Authorization"] = f"Bearer {settings.openrouter_api_key}"

    payload: dict = {
        "model": model_id,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens

    # Try the primary model, then fallback if auto-routed
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
            data = await _forward_to_upstream(mid, payload, headers)
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
        # All models failed
        if isinstance(last_error, httpx.HTTPStatusError):
            raise HTTPException(status_code=last_error.response.status_code, detail=str(last_error))
        raise HTTPException(status_code=502, detail="All models failed")

    # Parse response and return in OpenAI format
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


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    from tokenwise import __version__

    return {"status": "ok", "version": __version__}
