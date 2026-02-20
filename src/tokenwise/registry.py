"""ModelRegistry — loads and caches model metadata and pricing from OpenRouter."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import yaml

from tokenwise.config import get_settings
from tokenwise.models import ModelInfo, ModelTier

# Keyword-based capability detection
_CAPABILITY_KEYWORDS: dict[str, list[str]] = {
    "code": ["code", "coder", "codestral", "deepseek-coder", "starcoder"],
    "reasoning": ["o1", "o3", "o4", "reasoning", "think", "r1"],
    "vision": ["vision", "4o", "gpt-4.1", "claude-3", "claude-4", "gemini"],
    "math": ["math", "o1", "o3", "o4"],
    "creative": ["claude", "gpt-4", "gemini"],
}


def _infer_capabilities(model_id: str) -> list[str]:
    """Infer capabilities from the model ID using keyword heuristics."""
    caps = []
    model_lower = model_id.lower()
    for cap, keywords in _CAPABILITY_KEYWORDS.items():
        if any(kw in model_lower for kw in keywords):
            caps.append(cap)
    return caps


def _infer_tier(input_price: float) -> ModelTier:
    """Infer tier from input price per million tokens."""
    if input_price >= 5.0:
        return ModelTier.FLAGSHIP
    if input_price >= 0.5:
        return ModelTier.MID
    return ModelTier.BUDGET


def _parse_openrouter_model(data: dict) -> ModelInfo:
    """Parse an OpenRouter API model entry into ModelInfo."""
    model_id = data.get("id", "")
    name = data.get("name", model_id)

    pricing = data.get("pricing", {})
    # OpenRouter returns prices as strings in $/token — convert to $/M tokens
    try:
        input_price = float(pricing.get("prompt", 0)) * 1_000_000
    except (ValueError, TypeError):
        input_price = 0.0
    try:
        output_price = float(pricing.get("completion", 0)) * 1_000_000
    except (ValueError, TypeError):
        output_price = 0.0

    context_window = data.get("context_length", 4096)

    # Infer provider from model ID prefix
    provider = model_id.split("/")[0] if "/" in model_id else ""

    return ModelInfo(
        id=model_id,
        name=name,
        provider=provider,
        input_price=input_price,
        output_price=output_price,
        context_window=context_window,
        capabilities=_infer_capabilities(model_id),
        tier=_infer_tier(input_price),
    )


class ModelRegistry:
    """Loads, caches, and queries model metadata."""

    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._last_fetched: float = 0.0

    @property
    def models(self) -> dict[str, ModelInfo]:
        return dict(self._models)

    def load_from_openrouter(self) -> int:
        """Fetch model list from OpenRouter API. Returns count of models loaded."""
        settings = get_settings()
        headers: dict[str, str] = {}
        if settings.openrouter_api_key:
            headers["Authorization"] = f"Bearer {settings.openrouter_api_key}"

        resp = httpx.get(
            f"{settings.openrouter_base_url}/models",
            headers=headers,
            timeout=30.0,
        )
        resp.raise_for_status()

        data = resp.json().get("data", [])
        self._models.clear()
        for entry in data:
            model = _parse_openrouter_model(entry)
            self._models[model.id] = model

        self._last_fetched = time.time()
        return len(self._models)

    def load_from_file(self, path: str | Path) -> int:
        """Load models from a local YAML file. Returns count of models loaded."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or "models" not in data:
            raise ValueError(f"Invalid models file: expected a 'models' key in {path}")

        self._models.clear()
        for entry in data["models"]:
            model = ModelInfo(**entry)
            self._models[model.id] = model

        self._last_fetched = time.time()
        return len(self._models)

    def ensure_loaded(self) -> None:
        """Load models if not already loaded or if cache is stale."""
        settings = get_settings()
        ttl = settings.cache_ttl

        if self._models and (time.time() - self._last_fetched) < ttl:
            return

        # Try local file first, then OpenRouter
        if settings.local_models_file:
            self.load_from_file(settings.local_models_file)
        else:
            self.load_from_openrouter()

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get a specific model by ID."""
        self.ensure_loaded()
        return self._models.get(model_id)

    def find_models(
        self,
        capability: str | None = None,
        max_input_price: float | None = None,
        tier: ModelTier | None = None,
        min_context: int | None = None,
    ) -> list[ModelInfo]:
        """Find models matching the given filters, sorted by input price ascending."""
        self.ensure_loaded()
        results = list(self._models.values())

        if capability:
            results = [m for m in results if capability in m.capabilities]
        if max_input_price is not None:
            results = [m for m in results if m.input_price <= max_input_price]
        if tier:
            results = [m for m in results if m.tier == tier]
        if min_context is not None:
            results = [m for m in results if m.context_window >= min_context]

        results.sort(key=lambda m: m.input_price)
        return results

    def cheapest(self, capability: str | None = None) -> ModelInfo | None:
        """Return the cheapest model, optionally filtered by capability."""
        models = self.find_models(capability=capability)
        # Filter out free models (price == 0) as they're often rate-limited
        paid = [m for m in models if m.input_price > 0]
        return paid[0] if paid else (models[0] if models else None)

    def list_all(self) -> list[ModelInfo]:
        """Return all loaded models sorted by input price."""
        self.ensure_loaded()
        models = list(self._models.values())
        models.sort(key=lambda m: m.input_price)
        return models
