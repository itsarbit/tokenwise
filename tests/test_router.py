"""Tests for Router."""

from __future__ import annotations

import pytest

from tokenwise.models import ModelTier, RoutingStrategy
from tokenwise.registry import ModelRegistry
from tokenwise.router import Router, _detect_capabilities, _estimate_complexity


class TestDetectCapabilities:
    def test_code_query(self):
        caps = _detect_capabilities("Write a Python function to sort a list")
        assert "code" in caps

    def test_reasoning_query(self):
        caps = _detect_capabilities("Analyze and compare these two approaches step by step")
        assert "reasoning" in caps

    def test_math_query(self):
        caps = _detect_capabilities("Calculate the integral of x^2")
        assert "math" in caps

    def test_simple_query(self):
        caps = _detect_capabilities("Write a haiku")
        assert caps == []

    def test_no_false_positive_class(self):
        """'class' in 'classical' should not trigger code detection."""
        caps = _detect_capabilities("Write about classical music")
        assert "code" not in caps

    def test_no_false_positive_fix(self):
        """'fix' in 'fixings' should not trigger code detection."""
        caps = _detect_capabilities("What are the fixings for dinner")
        assert "code" not in caps

    def test_no_false_positive_reason(self):
        """'reason' as substring in 'treason' should not trigger."""
        caps = _detect_capabilities("Tell me about treason in history")
        assert "reasoning" not in caps


class TestEstimateComplexity:
    def test_simple(self):
        assert _estimate_complexity("Write a haiku") == "simple"

    def test_moderate(self):
        query = " ".join(["word"] * 30)
        assert _estimate_complexity(query) == "moderate"

    def test_complex(self):
        query = " ".join(["word"] * 80)
        assert _estimate_complexity(query) == "complex"


class TestRouter:
    def test_route_cheapest(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        model = router.route("Hello world", strategy="cheapest")
        # Should be one of the cheapest models
        assert model.input_price <= 0.20

    def test_route_best_quality(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        model = router.route("Complex task", strategy="best_quality")
        assert model.tier == ModelTier.FLAGSHIP

    def test_route_balanced_simple(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        model = router.route("Write a haiku", strategy="balanced")
        # Simple query → budget tier
        assert model.tier == ModelTier.BUDGET

    def test_route_balanced_complex(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        long_query = " ".join(["analyze"] * 80)
        model = router.route(long_query, strategy="balanced")
        # Complex query → flagship tier
        assert model.tier == ModelTier.FLAGSHIP

    def test_route_budget_constrained(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        model = router.route(
            "Write some code",
            strategy="budget_constrained",
            budget=0.01,
        )
        # Should pick a cheap model
        assert model.input_price < 5.0

    def test_route_budget_constrained_requires_budget(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        with pytest.raises(ValueError, match="budget is required"):
            router.route("Hello", strategy="budget_constrained")

    def test_route_with_required_capability(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        model = router.route("Do something", strategy="cheapest", required_capability="reasoning")
        assert "reasoning" in model.capabilities

    def test_route_string_strategy(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        # Should accept string strategies
        model = router.route("Hello", strategy="cheapest")
        assert model is not None

    def test_route_enum_strategy(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        model = router.route("Hello", strategy=RoutingStrategy.CHEAPEST)
        assert model is not None

    def test_route_no_models_raises(self):
        empty_registry = ModelRegistry()
        # Pre-load with no models but mark as loaded
        empty_registry._models = {}
        empty_registry._last_fetched = 9999999999.0
        # Prevent ensure_loaded from fetching by overriding it
        empty_registry.ensure_loaded = lambda: None
        router = Router(empty_registry)
        with pytest.raises(ValueError, match="No models found"):
            router.route("Hello", strategy="cheapest")
