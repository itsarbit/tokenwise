"""Anthropic Messages API provider adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
_API_VERSION = "2023-06-01"

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
}


class AnthropicProvider:
    """Direct Anthropic API adapter.

    Translates between OpenAI-compatible format and the Anthropic Messages API.
    """

    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.base_url = _DEFAULT_BASE_URL

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": _API_VERSION,
        }

    # -- Format translation ---------------------------------------------------

    @staticmethod
    def _to_anthropic_request(
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Convert OpenAI-format messages to Anthropic request payload."""
        system: str | None = None
        anthropic_msgs: list[dict[str, Any]] = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg.get("content", "")
            else:
                anthropic_msgs.append(
                    {
                        "role": msg["role"],
                        "content": msg.get("content", ""),
                    }
                )

        payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_msgs,
            "max_tokens": max_tokens or 4096,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

    @staticmethod
    def _to_openai_response(data: dict[str, Any]) -> dict[str, Any]:
        """Convert Anthropic response to OpenAI-compatible format."""
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        stop_reason = data.get("stop_reason", "end_turn")

        return {
            "id": data.get("id", ""),
            "object": "chat.completion",
            "model": data.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": _STOP_REASON_MAP.get(stop_reason, "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

    # -- Synchronous ----------------------------------------------------------

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        payload = self._to_anthropic_request(
            model,
            messages,
            temperature,
            max_tokens,
        )
        resp = httpx.post(
            f"{self.base_url}/messages",
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return self._to_openai_response(resp.json())

    # -- Async ----------------------------------------------------------------

    async def achat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        payload = self._to_anthropic_request(
            model,
            messages,
            temperature,
            max_tokens,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{self.base_url}/messages",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            return self._to_openai_response(resp.json())

    async def astream_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        """Stream from Anthropic, converting events to OpenAI SSE format."""
        payload = self._to_anthropic_request(
            model,
            messages,
            temperature,
            max_tokens,
        )
        payload["stream"] = True

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/messages",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line.removeprefix("data: ").strip()
                    if data_str == "[DONE]":
                        yield "data: [DONE]"
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    chunk_line = self._convert_stream_event(
                        event,
                        model,
                    )
                    if chunk_line:
                        yield chunk_line

        # Ensure we always send [DONE]
        yield "data: [DONE]"

    @staticmethod
    def _convert_stream_event(
        event: dict[str, Any],
        model: str,
    ) -> str | None:
        """Convert an Anthropic streaming event to an OpenAI SSE data line."""
        event_type = event.get("type", "")

        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            text = delta.get("text", "")
            if text:
                chunk = {
                    "id": "",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }
                    ],
                }
                return "data: " + json.dumps(chunk)

        elif event_type == "message_delta":
            stop_reason = event.get("delta", {}).get("stop_reason")
            finish = _STOP_REASON_MAP.get(stop_reason, "stop")
            chunk = {
                "id": "",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish,
                    }
                ],
            }
            return "data: " + json.dumps(chunk)

        return None
