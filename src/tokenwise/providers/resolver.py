"""Provider resolver — maps model IDs to the right LLM provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tokenwise.config import get_settings

if TYPE_CHECKING:
    from tokenwise.providers.base import LLMProvider

# Maps model ID prefix → (settings key field, provider class import path)
_DIRECT_PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("openai_api_key", "tokenwise.providers.openai.OpenAIProvider"),
    "anthropic": (
        "anthropic_api_key",
        "tokenwise.providers.anthropic.AnthropicProvider",
    ),
    "google": ("google_api_key", "tokenwise.providers.google.GoogleProvider"),
}


def _import_class(dotted_path: str) -> type:
    """Import a class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class ProviderResolver:
    """Resolves a model ID to (provider_instance, model_name_for_provider).

    Resolution logic:
      1. Extract prefix from model ID (e.g. ``anthropic`` from
         ``anthropic/claude-sonnet-4``)
      2. If a direct API key exists for that provider, use it (strip prefix)
      3. Otherwise, fall back to OpenRouter (pass model ID as-is)
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def resolve(self, model_id: str) -> tuple[LLMProvider, str]:
        """Return ``(provider, model_name)`` for *model_id*."""
        settings = get_settings()
        prefix = model_id.split("/")[0] if "/" in model_id else ""
        bare_model = model_id.split("/", 1)[1] if "/" in model_id else model_id

        # Check for a direct provider key
        if prefix in _DIRECT_PROVIDERS:
            key_field, class_path = _DIRECT_PROVIDERS[prefix]
            api_key = getattr(settings, key_field, "")
            if api_key:
                provider = self._get_or_create(
                    prefix,
                    lambda: _import_class(class_path)(api_key),
                )
                return provider, bare_model

        # Fall back to OpenRouter
        from tokenwise.providers.openrouter import OpenRouterProvider

        api_key = settings.require_api_key()
        provider = self._get_or_create(
            "openrouter",
            lambda: OpenRouterProvider(api_key, settings.openrouter_base_url),
        )
        return provider, model_id  # OpenRouter wants the full prefixed ID

    def _get_or_create(
        self,
        key: str,
        factory: Any,
    ) -> LLMProvider:
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]
