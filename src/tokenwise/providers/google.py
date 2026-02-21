"""Google Gemini API provider adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
}


class GoogleProvider:
    """Direct Google Gemini API adapter.

    Translates between OpenAI-compatible format and the Gemini REST API.
    """

    name = "google"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.base_url = _DEFAULT_BASE_URL

    # -- Format translation ---------------------------------------------------

    @staticmethod
    def _to_gemini_request(
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Convert OpenAI-format messages to Gemini payload."""
        system_instruction: dict[str, Any] | None = None
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = {
                    "parts": [{"text": msg.get("content", "")}],
                }
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(
                    {
                        "role": role,
                        "parts": [{"text": msg.get("content", "")}],
                    }
                )

        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["system_instruction"] = system_instruction

        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if generation_config:
            payload["generationConfig"] = generation_config

        return payload

    @staticmethod
    def _to_openai_response(
        data: dict[str, Any],
        model: str,
    ) -> dict[str, Any]:
        """Convert Gemini response to OpenAI-compatible format."""
        candidates = data.get("candidates", [])
        content = ""
        finish_reason = "stop"

        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            content = "".join(p.get("text", "") for p in parts)
            raw_reason = candidates[0].get("finishReason", "STOP")
            finish_reason = _FINISH_REASON_MAP.get(raw_reason, "stop")

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)

        return {
            "id": "",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage.get(
                    "totalTokenCount",
                    prompt_tokens + completion_tokens,
                ),
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
        payload = self._to_gemini_request(messages, temperature, max_tokens)
        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        resp = httpx.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return self._to_openai_response(resp.json(), model)

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
        payload = self._to_gemini_request(messages, temperature, max_tokens)
        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            return self._to_openai_response(resp.json(), model)

    async def astream_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        """Stream from Gemini, converting to OpenAI SSE format."""
        payload = self._to_gemini_request(messages, temperature, max_tokens)
        url = f"{self.base_url}/models/{model}:streamGenerateContent?key={self.api_key}&alt=sse"
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line.removeprefix("data: ").strip()
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    chunk_line = self._convert_stream_event(event, model)
                    if chunk_line:
                        yield chunk_line

        yield "data: [DONE]"

    @staticmethod
    def _convert_stream_event(
        event: dict[str, Any],
        model: str,
    ) -> str | None:
        """Convert a Gemini streaming event to an OpenAI SSE data line."""
        candidates = event.get("candidates", [])
        if not candidates:
            return None

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        if not text:
            return None

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
