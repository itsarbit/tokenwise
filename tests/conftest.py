"""Shared test fixtures and mock data."""

from __future__ import annotations

import pytest

from tokenwise.config import reset_settings
from tokenwise.models import ModelInfo, ModelTier
from tokenwise.registry import ModelRegistry


# Sample models for testing
SAMPLE_MODELS = [
    ModelInfo(
        id="openai/gpt-4.1-mini",
        name="GPT-4.1 Mini",
        provider="openai",
        input_price=0.40,
        output_price=1.60,
        context_window=1_000_000,
        capabilities=["code", "reasoning", "vision"],
        tier=ModelTier.BUDGET,
    ),
    ModelInfo(
        id="openai/gpt-4.1",
        name="GPT-4.1",
        provider="openai",
        input_price=2.00,
        output_price=8.00,
        context_window=1_000_000,
        capabilities=["code", "reasoning", "vision"],
        tier=ModelTier.MID,
    ),
    ModelInfo(
        id="anthropic/claude-sonnet-4",
        name="Claude Sonnet 4",
        provider="anthropic",
        input_price=3.00,
        output_price=15.00,
        context_window=200_000,
        capabilities=["code", "reasoning", "creative", "vision"],
        tier=ModelTier.MID,
    ),
    ModelInfo(
        id="anthropic/claude-opus-4",
        name="Claude Opus 4",
        provider="anthropic",
        input_price=15.00,
        output_price=75.00,
        context_window=200_000,
        capabilities=["code", "reasoning", "creative", "vision"],
        tier=ModelTier.FLAGSHIP,
    ),
    ModelInfo(
        id="openai/o3",
        name="o3",
        provider="openai",
        input_price=10.00,
        output_price=40.00,
        context_window=200_000,
        capabilities=["code", "reasoning", "math"],
        tier=ModelTier.FLAGSHIP,
    ),
    ModelInfo(
        id="google/gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        provider="google",
        input_price=0.15,
        output_price=0.60,
        context_window=1_000_000,
        capabilities=["code", "vision"],
        tier=ModelTier.BUDGET,
    ),
    ModelInfo(
        id="deepseek/deepseek-chat",
        name="DeepSeek V3",
        provider="deepseek",
        input_price=0.14,
        output_price=0.28,
        context_window=128_000,
        capabilities=["code"],
        tier=ModelTier.BUDGET,
    ),
]


@pytest.fixture
def sample_registry() -> ModelRegistry:
    """Create a ModelRegistry pre-loaded with sample models."""
    registry = ModelRegistry()
    for model in SAMPLE_MODELS:
        registry._models[model.id] = model
    registry._last_fetched = 9999999999.0  # far future so cache never expires
    return registry


@pytest.fixture(autouse=True)
def _reset_settings():
    """Reset global settings between tests."""
    reset_settings()
    yield
    reset_settings()
