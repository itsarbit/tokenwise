"""Direct OpenAI provider."""

from __future__ import annotations

import httpx

from tokenwise.providers._openai_compat import OpenAICompatibleProvider

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider(OpenAICompatibleProvider):
    """Direct OpenAI API (same format as OpenRouter)."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(api_key, base_url, http_client=http_client)
