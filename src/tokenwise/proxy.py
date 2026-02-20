"""OpenAI-compatible FastAPI proxy server."""

from __future__ import annotations

import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException

from tokenwise.config import get_settings
from tokenwise.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Usage,
)
from tokenwise.registry import ModelRegistry
from tokenwise.router import Router

app = FastAPI(
    title="TokenWise Proxy",
    description="OpenAI-compatible proxy that routes requests to optimal LLM models",
    version="0.1.0",
)

_registry = ModelRegistry()
_router = Router(_registry)


@app.get("/v1/models")
async def list_models() -> dict:
    """List available models (OpenAI-compatible)."""
    _registry.ensure_loaded()
    models = _registry.list_all()
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


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Handle chat completion requests with intelligent routing."""
    settings = get_settings()

    # Determine which model to use
    if request.model == "auto" or request.model.startswith("tokenwise/"):
        # Route automatically
        last_message = request.messages[-1].content or "" if request.messages else ""
        strategy = request.extra.get("strategy", settings.default_strategy)
        budget = request.extra.get("budget", settings.default_budget)

        try:
            model = _router.route(
                query=last_message,
                strategy=strategy,
                budget=budget,
            )
            model_id = model.id
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Routing failed: {e}")
    else:
        model_id = request.model

    # Forward the request to OpenRouter
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

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    # Parse response and return in OpenAI format
    choices = []
    for i, choice in enumerate(data.get("choices", [])):
        msg = choice.get("message", {})
        choices.append(ChatCompletionChoice(
            index=i,
            message=ChatMessage(role=msg.get("role", "assistant"), content=msg.get("content", "")),
            finish_reason=choice.get("finish_reason", "stop"),
        ))

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
    return {"status": "ok", "version": "0.1.0"}
