"""Tests for ModelRegistry."""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from tokenwise.models import ModelTier
from tokenwise.registry import (
    ModelRegistry,
    _infer_capabilities,
    _infer_tier,
    _parse_openrouter_model,
)


class TestInferCapabilities:
    def test_code_model(self):
        caps = _infer_capabilities("deepseek/deepseek-coder-v2")
        assert "code" in caps

    def test_reasoning_model(self):
        caps = _infer_capabilities("openai/o3")
        assert "reasoning" in caps

    def test_vision_model(self):
        caps = _infer_capabilities("openai/gpt-4.1")
        assert "vision" in caps

    def test_no_capabilities(self):
        caps = _infer_capabilities("some/unknown-model")
        assert caps == []


class TestInferTier:
    def test_flagship(self):
        assert _infer_tier(15.0) == ModelTier.FLAGSHIP

    def test_mid(self):
        assert _infer_tier(2.0) == ModelTier.MID

    def test_budget(self):
        assert _infer_tier(0.1) == ModelTier.BUDGET


class TestParseOpenRouterModel:
    def test_basic_parsing(self):
        data = {
            "id": "openai/gpt-4.1-mini",
            "name": "GPT-4.1 Mini",
            "pricing": {"prompt": "0.0000004", "completion": "0.0000016"},
            "context_length": 1000000,
        }
        model = _parse_openrouter_model(data)
        assert model.id == "openai/gpt-4.1-mini"
        assert model.provider == "openai"
        assert model.input_price == pytest.approx(0.4)
        assert model.output_price == pytest.approx(1.6)
        assert model.context_window == 1000000

    def test_missing_pricing(self):
        data = {"id": "test/model", "name": "Test"}
        model = _parse_openrouter_model(data)
        assert model.input_price == 0.0
        assert model.output_price == 0.0


class TestModelRegistry:
    def test_get_model(self, sample_registry: ModelRegistry):
        model = sample_registry.get_model("openai/gpt-4.1-mini")
        assert model is not None
        assert model.provider == "openai"

    def test_get_model_missing(self, sample_registry: ModelRegistry):
        assert sample_registry.get_model("nonexistent/model") is None

    def test_find_models_by_capability(self, sample_registry: ModelRegistry):
        models = sample_registry.find_models(capability="code")
        assert len(models) >= 3
        assert all("code" in m.capabilities for m in models)

    def test_find_models_by_tier(self, sample_registry: ModelRegistry):
        models = sample_registry.find_models(tier=ModelTier.FLAGSHIP)
        assert len(models) >= 2
        assert all(m.tier == ModelTier.FLAGSHIP for m in models)

    def test_find_models_by_max_price(self, sample_registry: ModelRegistry):
        models = sample_registry.find_models(max_input_price=1.0)
        assert all(m.input_price <= 1.0 for m in models)

    def test_find_models_sorted_by_price(self, sample_registry: ModelRegistry):
        models = sample_registry.find_models()
        prices = [m.input_price for m in models]
        assert prices == sorted(prices)

    def test_cheapest(self, sample_registry: ModelRegistry):
        model = sample_registry.cheapest()
        assert model is not None
        assert model.input_price <= 0.20

    def test_cheapest_with_capability(self, sample_registry: ModelRegistry):
        model = sample_registry.cheapest(capability="reasoning")
        assert model is not None
        assert "reasoning" in model.capabilities

    def test_list_all(self, sample_registry: ModelRegistry):
        models = sample_registry.list_all()
        assert len(models) == 7

    @respx.mock
    def test_load_from_openrouter(self):
        respx.get("https://openrouter.ai/api/v1/models").mock(
            return_value=Response(
                200,
                json={
                    "data": [
                        {
                            "id": "openai/gpt-4.1-mini",
                            "name": "GPT-4.1 Mini",
                            "pricing": {"prompt": "0.0000004", "completion": "0.0000016"},
                            "context_length": 1000000,
                        },
                        {
                            "id": "anthropic/claude-sonnet-4",
                            "name": "Claude Sonnet 4",
                            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                            "context_length": 200000,
                        },
                    ]
                },
            )
        )

        registry = ModelRegistry()
        count = registry.load_from_openrouter()
        assert count == 2
        assert registry.get_model("openai/gpt-4.1-mini") is not None

    def test_load_from_file(self, tmp_path):
        models_file = tmp_path / "models.yaml"
        models_file.write_text(
            "models:\n"
            "  - id: test/model-a\n"
            "    name: Model A\n"
            "    provider: test\n"
            "    input_price: 1.0\n"
            "    output_price: 2.0\n"
            "    context_window: 8192\n"
            "    capabilities: [code]\n"
            "    tier: budget\n"
        )
        registry = ModelRegistry()
        count = registry.load_from_file(models_file)
        assert count == 1
        assert registry.get_model("test/model-a") is not None
