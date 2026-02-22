"""Shared base for providers using the OpenAI chat completions format."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from tokenwise.providers.base import _shared_or_ephemeral


class OpenAICompatibleProvider:
    """Base for providers that use the OpenAI /chat/completions format.

    Subclass and set ``name`` and the default ``base_url`` to create a
    concrete provider (e.g. OpenRouter, OpenAI, or any compatible API).
    """

    name: str = ""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self._http_client = http_client

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_payload(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    # -- Synchronous ---------------------------------------------------------

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self._auth_headers(),
            json=self._build_payload(model, messages, temperature, max_tokens),
            timeout=timeout,
        )
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result

    # -- Async ---------------------------------------------------------------

    async def achat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        async with _shared_or_ephemeral(self._http_client, timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._auth_headers(),
                json=self._build_payload(
                    model,
                    messages,
                    temperature,
                    max_tokens,
                ),
                timeout=timeout,
            )
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result

    async def astream_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(model, messages, temperature, max_tokens)
        payload["stream"] = True
        async with _shared_or_ephemeral(self._http_client, timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._auth_headers(),
                json=payload,
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        yield line
                        if line.strip() == "data: [DONE]":
                            break
