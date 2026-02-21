"""Tests for the proxy server."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from tokenwise.models import ModelInfo, ModelTier
from tokenwise.providers import ProviderResolver
from tokenwise.proxy import _FailedModels, app, state
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
        capabilities=["general", "code", "vision"],
        tier=ModelTier.BUDGET,
    ),
]

_UPSTREAM_RESPONSE = {
    "choices": [
        {
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


@pytest.fixture
def client():
    """Create a test client with mocked state."""
    registry = ModelRegistry()
    for m in SAMPLE_MODELS:
        registry._models[m.id] = m
    registry._last_fetched = 9999999999.0

    state.registry = registry
    state.router = Router(registry)
    state.http_client = httpx.AsyncClient()
    state.resolver = ProviderResolver(http_client=state.http_client)
    state.failed_models = _FailedModels()

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
    def test_streaming_response(self, client):
        """stream=true should return SSE chunks."""
        chunk_base = {
            "id": "ch-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "openai/gpt-4.1-mini",
        }

        def _chunk(delta, finish_reason=None):
            return "data: " + json.dumps(
                {
                    **chunk_base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            )

        sse_lines = [
            _chunk({"role": "assistant", "content": "Hi"}),
            _chunk({"content": "!"}),
            _chunk({}, finish_reason="stop"),
            "data: [DONE]",
        ]

        mock_provider = MockProvider(stream_lines=sse_lines)
        with patch.object(
            state.resolver,
            "resolve",
            return_value=(mock_provider, "gpt-4.1-mini"),
        ):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-4.1-mini",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            body = resp.text
            events = [evt for evt in body.strip().split("\n\n") if evt.startswith("data:")]
            assert len(events) >= 2

            first_data = json.loads(events[0].removeprefix("data: "))
            assert first_data["object"] == "chat.completion.chunk"
            assert first_data["choices"][0]["delta"]["content"] == "Hi"

            assert events[-1].strip() == "data: [DONE]"

    def test_ignores_extra_fields(self, client):
        """Extra fields in the request should not cause validation errors."""
        mock_provider = MockProvider(response=_UPSTREAM_RESPONSE)
        with patch.object(
            state.resolver,
            "resolve",
            return_value=(mock_provider, "gpt-4.1-mini"),
        ):
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
        mock_provider = MockProvider(
            response={
                **_UPSTREAM_RESPONSE,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hi!"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        with patch.object(
            state.resolver,
            "resolve",
            return_value=(mock_provider, "gpt-4.1-mini"),
        ):
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
            assert data["model"]

    def test_passthrough_model(self, client):
        """Explicit model ID should be passed through."""
        mock_provider = MockProvider(
            response={
                **_UPSTREAM_RESPONSE,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Done"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        with patch.object(
            state.resolver,
            "resolve",
            return_value=(mock_provider, "gpt-4.1-mini"),
        ):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-4.1-mini",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert resp.status_code == 200
            assert resp.json()["model"] == "openai/gpt-4.1-mini"


class TestFailedModels:
    def test_add_and_contains(self):
        fm = _FailedModels()
        fm.add("model-a")
        assert "model-a" in fm
        assert "model-b" not in fm

    def test_to_set(self):
        fm = _FailedModels()
        fm.add("model-a")
        fm.add("model-b")
        assert fm.to_set() == {"model-a", "model-b"}

    def test_ttl_expiry(self):
        """Entries should expire after TTL."""
        import time

        fm = _FailedModels(ttl=0.1)  # 100ms TTL
        fm.add("model-a")
        assert "model-a" in fm
        time.sleep(0.15)
        assert "model-a" not in fm
        assert fm.to_set() == set()

    def test_max_size_eviction(self):
        fm = _FailedModels(max_size=3)
        fm.add("m1")
        fm.add("m2")
        fm.add("m3")
        fm.add("m4")  # Should evict oldest (m1)
        assert "m1" not in fm
        assert "m4" in fm
        assert len(fm.to_set()) == 3


# -- Helpers --


class MockProvider:
    """Mock LLM provider for proxy tests."""

    name = "mock"

    def __init__(
        self,
        response: dict | None = None,
        stream_lines: list[str] | None = None,
    ):
        self._response = response or {}
        self._stream_lines = stream_lines or []

    async def achat_completion(self, model, messages, **kwargs):
        return self._response

    async def astream_completion(self, model, messages, **kwargs):
        for line in self._stream_lines:
            yield line
