"""Settings loading from environment variables and config files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class MissingAPIKeyError(Exception):
    """Raised when no API key is configured for any provider."""

    def __init__(self) -> None:
        super().__init__(
            "No API key configured. Set one of:\n"
            "  export OPENROUTER_API_KEY='sk-or-...'  "
            "(routes all providers via OpenRouter)\n"
            "  export OPENAI_API_KEY='sk-...'          "
            "(direct OpenAI access)\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'   "
            "(direct Anthropic access)\n"
            "  export GOOGLE_API_KEY='...'             "
            "(direct Google AI access)"
        )


_DEFAULT_CONFIG_PATH = Path.home() / ".config" / "tokenwise" / "config.yaml"


class RiskGateConfig(BaseModel):
    """Configuration for the risk gate."""

    enabled: bool = False
    irreversible_block: bool = True
    ambiguity_threshold: float = 0.9


class Settings(BaseModel):
    """Application settings."""

    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )
    default_strategy: str = Field(default="balanced")
    default_budget: float = Field(default=1.0, description="Default budget in USD")
    planner_model: str = Field(
        default="openai/gpt-4.1-mini",
        description="Model used for task decomposition",
    )
    proxy_host: str = Field(default="127.0.0.1")
    proxy_port: int = Field(default=8000)
    cache_ttl: int = Field(default=3600, description="Model registry cache TTL in seconds")
    local_models_file: str | None = Field(
        default=None, description="Path to a local models YAML file for offline use"
    )
    openai_api_key: str = Field(default="", description="OpenAI API key (direct)")
    anthropic_api_key: str = Field(default="", description="Anthropic API key (direct)")
    google_api_key: str = Field(default="", description="Google AI API key (direct)")
    ledger_path: str = Field(
        default="",
        description="Path to ledger JSONL file (default: ~/.config/tokenwise/ledger.jsonl)",
    )
    min_output_tokens: int = Field(
        default=100,
        description="Minimum output tokens below which a step is skipped rather than producing "
        "truncated output. Set lower for workflows that need tiny outputs under small budgets.",
    )
    model_overrides: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description="Per-model capability/tier overrides",
    )
    escalation_policy: str = Field(default="flexible", description="flexible or monotonic")
    risk_gate: RiskGateConfig = Field(default_factory=RiskGateConfig)
    trace_level: str = Field(default="basic", description="basic or verbose")

    def require_api_key(self) -> str:
        """Return the OpenRouter API key or raise MissingAPIKeyError."""
        if not self.openrouter_api_key:
            raise MissingAPIKeyError
        return self.openrouter_api_key


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from environment variables, then overlay config file values."""
    env_values: dict[str, Any] = {}

    env_map = {
        "OPENROUTER_API_KEY": "openrouter_api_key",
        "OPENROUTER_BASE_URL": "openrouter_base_url",
        "TOKENWISE_DEFAULT_STRATEGY": "default_strategy",
        "TOKENWISE_DEFAULT_BUDGET": "default_budget",
        "TOKENWISE_PLANNER_MODEL": "planner_model",
        "TOKENWISE_PROXY_HOST": "proxy_host",
        "TOKENWISE_PROXY_PORT": "proxy_port",
        "TOKENWISE_CACHE_TTL": "cache_ttl",
        "TOKENWISE_LOCAL_MODELS": "local_models_file",
        "TOKENWISE_LEDGER_PATH": "ledger_path",
        "OPENAI_API_KEY": "openai_api_key",
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "GOOGLE_API_KEY": "google_api_key",
        "TOKENWISE_MIN_OUTPUT_TOKENS": "min_output_tokens",
        "TOKENWISE_ESCALATION_POLICY": "escalation_policy",
        "TOKENWISE_TRACE_LEVEL": "trace_level",
    }

    for env_var, field_name in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            env_values[field_name] = val

    # Load config file (lower priority than env vars)
    path = config_path or _DEFAULT_CONFIG_PATH
    file_values: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                file_values = data

    # Env vars override config file
    merged = {**file_values, **env_values}

    # Merge TOKENWISE_RISK_GATE_* env vars into risk_gate dict
    risk_gate_env: dict[str, Any] = {}
    rg_enabled = os.environ.get("TOKENWISE_RISK_GATE_ENABLED")
    if rg_enabled is not None:
        risk_gate_env["enabled"] = rg_enabled.lower() in ("1", "true", "yes")
    rg_block = os.environ.get("TOKENWISE_RISK_GATE_IRREVERSIBLE_BLOCK")
    if rg_block is not None:
        risk_gate_env["irreversible_block"] = rg_block.lower() in ("1", "true", "yes")
    rg_threshold = os.environ.get("TOKENWISE_RISK_GATE_AMBIGUITY_THRESHOLD")
    if rg_threshold is not None:
        risk_gate_env["ambiguity_threshold"] = float(rg_threshold)
    if risk_gate_env:
        existing = merged.get("risk_gate", {})
        if isinstance(existing, dict):
            merged["risk_gate"] = {**existing, **risk_gate_env}
        else:
            merged["risk_gate"] = risk_gate_env

    return Settings(**merged)


# Singleton for convenience
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reset_settings() -> None:
    """Reset cached settings (useful for testing)."""
    global _settings
    _settings = None
