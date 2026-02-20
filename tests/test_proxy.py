"""Tests for the proxy server."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tokenwise.models import ModelInfo, ModelTier
from tokenwise.proxy import app, state
from tokenwise.registry import ModelRegistry
from tokenwise.router import Router

SAMPLE_MODELS = [
    ModelInfo(
        id="openai/gpt-4.1-mini",
        name="GPT-4.1 Mini",
        provider="openai",
        input_price=0.40,
        output_price=1.60,
        context_window=1_000_000,
        capabilities=["code", "vision"],
        tier=ModelTier.BUDGET,
    ),
]


@pytest.fixture
def client():
    """Create a test client with mocked state."""
    registry = ModelRegistry()
    for m in SAMPLE_MODELS:
        registry._models[m.id] = m
    registry._last_fetched = 9999999999.0

    state.registry = registry
    state.router = Router(registry)

    # Mock the async HTTP client
    import httpx

    state.http_client = httpx.AsyncClient()

    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestListModels:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        model_ids = [m["id"] for m in data["data"]]
        assert "openai/gpt-4.1-mini" in model_ids


class TestChatCompletions:
    def test_rejects_streaming(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 400
        assert "stream" in resp.json()["detail"].lower()

    def test_ignores_extra_fields(self, client):
        """Extra fields in the request should not cause validation errors."""
        with patch.object(state, "http_client") as mock_client:
            mock_response = MockResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Hello!",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "total_tokens": 8,
                    },
                },
            )
            mock_client.post = AsyncMock(return_value=mock_response)

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-4.1-mini",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "some_unknown_field": True,
                    "top_p": 0.9,
                },
            )
            assert resp.status_code == 200

    def test_auto_routing(self, client):
        """model='auto' should trigger routing."""
        with patch.object(state, "http_client") as mock_client:
            mock_response = MockResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Hi!",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                    },
                },
            )
            mock_client.post = AsyncMock(return_value=mock_response)

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "chat.completion"
            assert data["choices"][0]["message"]["content"] == "Hi!"
            assert data["model"]  # should be filled in

    def test_passthrough_model(self, client):
        """Explicit model ID should be passed through."""
        with patch.object(state, "http_client") as mock_client:
            mock_response = MockResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Done",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "total_tokens": 6,
                    },
                },
            )
            mock_client.post = AsyncMock(return_value=mock_response)

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-4.1-mini",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert resp.status_code == 200
            assert resp.json()["model"] == "openai/gpt-4.1-mini"


# -- Helpers --


class MockResponse:
    """Mock httpx.Response for async client."""

    def __init__(self, status_code: int, json_data: dict):
        self.status_code = status_code
        self._json = json_data

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class AsyncMock:
    """Simple async mock for httpx.AsyncClient.post."""

    def __init__(self, return_value):
        self._return_value = return_value
        self.call_args = None

    async def __call__(self, *args, **kwargs):
        self.call_args = (args, kwargs)
        return self._return_value
