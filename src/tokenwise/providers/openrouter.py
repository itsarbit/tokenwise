"""OpenRouter provider — passes model IDs through as-is."""

from __future__ import annotations

import httpx

from tokenwise.providers._openai_compat import OpenAICompatibleProvider

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter API (OpenAI-compatible)."""

    name = "openrouter"

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(api_key, base_url, http_client=http_client)
