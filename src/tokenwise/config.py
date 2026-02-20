"""Settings loading from environment variables and config files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class MissingAPIKeyError(Exception):
    """Raised when an OpenRouter API key is required but not configured."""

    def __init__(self) -> None:
        super().__init__(
            "OPENROUTER_API_KEY is not set. "
            "Get one at https://openrouter.ai/keys and set it:\n"
            "  export OPENROUTER_API_KEY='sk-or-...'"
        )

_DEFAULT_CONFIG_PATH = Path.home() / ".config" / "tokenwise" / "config.yaml"


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

    def require_api_key(self) -> str:
        """Return the API key or raise MissingAPIKeyError."""
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
