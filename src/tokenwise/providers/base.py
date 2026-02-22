"""Base protocol for LLM provider adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

import httpx


@runtime_checkable
class LLMProvider(Protocol):
    """Interface for LLM provider adapters.

    All methods accept and return OpenAI-compatible formats.
    Each adapter handles translation to/from native API format internally.
    """

    @property
    def name(self) -> str:
        """Provider name, e.g. 'openrouter', 'openai', 'anthropic'."""
        ...

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Synchronous chat completion. Returns OpenAI-format response."""
        ...

    async def achat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Async chat completion. Returns OpenAI-format response."""
        ...

    def astream_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        """Async streaming. Yields SSE lines (e.g. 'data: {...}')."""
        ...


@asynccontextmanager
async def _shared_or_ephemeral(
    client: httpx.AsyncClient | None,
    timeout: float,
) -> AsyncIterator[httpx.AsyncClient]:
    """Yield a shared client if available, otherwise create a short-lived one."""
    if client is not None:
        yield client
    else:
        async with httpx.AsyncClient(timeout=timeout) as ephemeral:
            yield ephemeral
