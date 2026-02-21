"""Tests for LLM provider adapters and resolver."""

from __future__ import annotations

import respx
from httpx import Response

from tokenwise.config import reset_settings
from tokenwise.providers.anthropic import AnthropicProvider
from tokenwise.providers.google import GoogleProvider
from tokenwise.providers.openai import OpenAIProvider
from tokenwise.providers.openrouter import OpenRouterProvider
from tokenwise.providers.resolver import ProviderResolver

# -- OpenAI-compatible providers (OpenRouter + OpenAI) -----------------------

_OPENAI_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "model": "gpt-4.1-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
    },
}


class TestOpenAIProvider:
    @respx.mock
    def test_chat_completion(self):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(200, json=_OPENAI_RESPONSE),
        )
        provider = OpenAIProvider(api_key="sk-test")
        result = provider.chat_completion(
            "gpt-4.1-mini",
            [{"role": "user", "content": "Hi"}],
        )
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["usage"]["total_tokens"] == 8

    def test_name(self):
        assert OpenAIProvider(api_key="x").name == "openai"


class TestOpenRouterProvider:
    @respx.mock
    def test_chat_completion(self):
        respx.post(
            "https://openrouter.ai/api/v1/chat/completions",
        ).mock(return_value=Response(200, json=_OPENAI_RESPONSE))
        provider = OpenRouterProvider(api_key="sk-or-test")
        result = provider.chat_completion(
            "openai/gpt-4.1-mini",
            [{"role": "user", "content": "Hi"}],
        )
        assert result["choices"][0]["message"]["content"] == "Hello!"

    def test_name(self):
        assert OpenRouterProvider(api_key="x").name == "openrouter"


# -- Anthropic provider ------------------------------------------------------

_ANTHROPIC_RESPONSE = {
    "id": "msg_123",
    "model": "claude-sonnet-4",
    "content": [{"type": "text", "text": "Hello!"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


class TestAnthropicProvider:
    def test_to_anthropic_request_extracts_system(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        payload = AnthropicProvider._to_anthropic_request(
            "claude-sonnet-4",
            messages,
            None,
            1024,
        )
        assert payload["system"] == "You are helpful"
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["max_tokens"] == 1024

    def test_to_anthropic_request_default_max_tokens(self):
        payload = AnthropicProvider._to_anthropic_request(
            "claude-sonnet-4",
            [{"role": "user", "content": "Hi"}],
            None,
            None,
        )
        assert payload["max_tokens"] == 4096

    def test_to_openai_response(self):
        result = AnthropicProvider._to_openai_response(_ANTHROPIC_RESPONSE)
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    @respx.mock
    def test_chat_completion(self):
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_ANTHROPIC_RESPONSE),
        )
        provider = AnthropicProvider(api_key="sk-ant-test")
        result = provider.chat_completion(
            "claude-sonnet-4",
            [{"role": "user", "content": "Hi"}],
        )
        assert result["choices"][0]["message"]["content"] == "Hello!"

    def test_name(self):
        assert AnthropicProvider(api_key="x").name == "anthropic"


# -- Google provider ---------------------------------------------------------

_GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {"parts": [{"text": "Hello!"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 5,
        "candidatesTokenCount": 2,
        "totalTokenCount": 7,
    },
}


class TestGoogleProvider:
    def test_to_gemini_request_maps_roles(self):
        messages = [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        ]
        payload = GoogleProvider._to_gemini_request(messages, 0.5, 100)
        assert payload["system_instruction"]["parts"][0]["text"] == "Be concise"
        assert len(payload["contents"]) == 3
        assert payload["contents"][1]["role"] == "model"
        assert payload["generationConfig"]["temperature"] == 0.5
        assert payload["generationConfig"]["maxOutputTokens"] == 100

    def test_to_openai_response(self):
        result = GoogleProvider._to_openai_response(
            _GEMINI_RESPONSE,
            "gemini-2.5-flash",
        )
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 5
        assert result["usage"]["completion_tokens"] == 2

    @respx.mock
    def test_chat_completion(self):
        respx.post(
            url__regex=r"generativelanguage.*generateContent",
        ).mock(return_value=Response(200, json=_GEMINI_RESPONSE))
        provider = GoogleProvider(api_key="test-key")
        result = provider.chat_completion(
            "gemini-2.5-flash",
            [{"role": "user", "content": "Hi"}],
        )
        assert result["choices"][0]["message"]["content"] == "Hello!"

    def test_name(self):
        assert GoogleProvider(api_key="x").name == "google"


# -- ProviderResolver --------------------------------------------------------


class TestProviderResolver:
    def test_routes_to_openrouter_by_default(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        reset_settings()
        resolver = ProviderResolver()
        provider, model = resolver.resolve("anthropic/claude-sonnet-4")
        assert provider.name == "openrouter"
        assert model == "anthropic/claude-sonnet-4"

    def test_routes_to_direct_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        reset_settings()
        resolver = ProviderResolver()
        provider, model = resolver.resolve("anthropic/claude-sonnet-4")
        assert provider.name == "anthropic"
        assert model == "claude-sonnet-4"

    def test_routes_to_direct_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        reset_settings()
        resolver = ProviderResolver()
        provider, model = resolver.resolve("openai/gpt-4.1-mini")
        assert provider.name == "openai"
        assert model == "gpt-4.1-mini"

    def test_routes_to_direct_google(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        reset_settings()
        resolver = ProviderResolver()
        provider, model = resolver.resolve("google/gemini-2.5-flash")
        assert provider.name == "google"
        assert model == "gemini-2.5-flash"

    def test_falls_back_without_direct_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        reset_settings()
        resolver = ProviderResolver()
        provider, model = resolver.resolve("anthropic/claude-sonnet-4")
        assert provider.name == "openrouter"

    def test_caches_provider_instances(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        reset_settings()
        resolver = ProviderResolver()
        p1, _ = resolver.resolve("openai/gpt-4.1-mini")
        p2, _ = resolver.resolve("openai/gpt-4.1")
        assert p1 is p2
