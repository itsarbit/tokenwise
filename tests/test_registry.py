"""Tests for ModelRegistry."""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from tokenwise.models import ModelTier
from tokenwise.registry import (
    ModelRegistry,
    _has_vision,
    _infer_capabilities_fallback,
    _infer_tier_from_price,
    _match_family,
    _parse_openrouter_model,
)


class TestCapabilityResolution:
    def test_curated_match_gpt4_mini(self):
        family = _match_family("openai/gpt-4.1-mini")
        assert family is not None
        assert "code" in family["capabilities"]
        assert "general" in family["capabilities"]

    def test_curated_match_o3(self):
        family = _match_family("openai/o3")
        assert family is not None
        assert "reasoning" in family["capabilities"]
        assert "math" in family["capabilities"]
        assert family["tier"] == "flagship"

    def test_curated_match_o3_mini_before_o3(self):
        """o3-mini should match its own entry, not the broader o3 entry."""
        family = _match_family("openai/o3-mini")
        assert family is not None
        assert family["tier"] == "mid"

    def test_curated_match_deepseek_r1(self):
        family = _match_family("deepseek/deepseek-r1")
        assert family is not None
        assert "reasoning" in family["capabilities"]
        assert family["tier"] == "mid"

    def test_no_curated_match_unknown(self):
        assert _match_family("some/unknown-model") is None

    def test_fallback_code_model(self):
        caps = _infer_capabilities_fallback("some/code-model")
        assert "code" in caps

    def test_fallback_unknown_gets_general(self):
        caps = _infer_capabilities_fallback("some/unknown-model")
        assert "general" in caps

    def test_no_false_positive_o1_substring(self):
        """'o1' in 'bolt-101' should not match reasoning in fallback."""
        assert _match_family("some/bolt-101") is None
        caps = _infer_capabilities_fallback("some/bolt-101")
        assert "reasoning" not in caps


class TestTierResolution:
    def test_curated_tier_overrides_price(self):
        """DeepSeek R1 is cheap but should be mid tier per curated mapping."""
        family = _match_family("deepseek/deepseek-r1")
        assert family is not None
        assert family["tier"] == "mid"

    def test_price_fallback_flagship(self):
        assert _infer_tier_from_price(15.0) == ModelTier.FLAGSHIP

    def test_price_fallback_mid(self):
        assert _infer_tier_from_price(2.0) == ModelTier.MID

    def test_price_fallback_budget(self):
        assert _infer_tier_from_price(0.1) == ModelTier.BUDGET


class TestVisionDetection:
    def test_vision_from_modality(self):
        data = {"architecture": {"modality": "text+image->text"}}
        assert _has_vision(data) is True

    def test_no_vision_text_only(self):
        data = {"architecture": {"modality": "text->text"}}
        assert _has_vision(data) is False

    def test_no_vision_missing_architecture(self):
        assert _has_vision({}) is False


class TestParseOpenRouterModel:
    def test_basic_parsing(self):
        data = {
            "id": "openai/gpt-4.1-mini",
            "name": "GPT-4.1 Mini",
            "pricing": {"prompt": "0.0000004", "completion": "0.0000016"},
            "context_length": 1000000,
            "architecture": {"modality": "text+image->text"},
        }
        model = _parse_openrouter_model(data)
        assert model.id == "openai/gpt-4.1-mini"
        assert model.provider == "openai"
        assert model.input_price == pytest.approx(0.4)
        assert model.output_price == pytest.approx(1.6)
        assert model.context_window == 1000000
        assert "code" in model.capabilities
        assert "vision" in model.capabilities
        assert model.tier == ModelTier.BUDGET

    def test_missing_pricing(self):
        data = {"id": "test/model", "name": "Test"}
        model = _parse_openrouter_model(data)
        assert model.input_price == 0.0
        assert model.output_price == 0.0

    def test_unknown_model_uses_fallback(self):
        data = {
            "id": "unknown/some-model",
            "name": "Unknown",
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            "context_length": 8192,
        }
        model = _parse_openrouter_model(data)
        assert "general" in model.capabilities
        assert model.tier == ModelTier.MID  # price = 1.0, so mid tier


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

    def test_find_models_by_multiple_capabilities(self, sample_registry: ModelRegistry):
        """find_models with capabilities=[...] returns only models having all."""
        models = sample_registry.find_models(capabilities=["code", "reasoning"])
        assert len(models) >= 1
        for m in models:
            assert "code" in m.capabilities
            assert "reasoning" in m.capabilities

    def test_cheapest_with_multiple_capabilities(self, sample_registry: ModelRegistry):
        """cheapest(capabilities=[...]) returns cheapest model having all capabilities."""
        model = sample_registry.cheapest(capabilities=["code", "reasoning"])
        assert model is not None
        assert "code" in model.capabilities
        assert "reasoning" in model.capabilities
        # Should be cheapest among models with both code + reasoning
        all_matching = sample_registry.find_models(capabilities=["code", "reasoning"])
        paid = [m for m in all_matching if m.input_price > 0]
        assert model.input_price <= paid[0].input_price

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
