"""Tests for routing trace functionality."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from tokenwise.config import reset_settings
from tokenwise.models import (
    EscalationPolicy,
    ModelInfo,
    RiskGateBlockedError,
    RoutingTrace,
    TerminationState,
    TraceLevel,
)
from tokenwise.registry import ModelRegistry
from tokenwise.router import Router


class TestRouteWithTrace:
    def test_returns_trace(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        model, trace = router.route_with_trace("Write a Python sort function")
        assert isinstance(model, ModelInfo)
        assert isinstance(trace, RoutingTrace)

    def test_request_id_present(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        _, trace = router.route_with_trace("Write a Python sort function")
        assert trace.request_id
        assert len(trace.request_id) == 12

    def test_request_ids_unique(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        _, trace1 = router.route_with_trace("Write a Python sort function")
        _, trace2 = router.route_with_trace("Write a Python sort function")
        assert trace1.request_id != trace2.request_id

    def test_initial_equals_final_without_escalation(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        model, trace = router.route_with_trace("Write a Python sort function")
        assert trace.initial_model == model.id
        assert trace.final_model == model.id
        assert trace.initial_tier == model.tier
        assert trace.final_tier == model.tier

    def test_termination_state_completed(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        _, trace = router.route_with_trace("Write a Python sort function")
        assert trace.termination_state == TerminationState.COMPLETED

    def test_no_escalations(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        _, trace = router.route_with_trace("Write a Python sort function")
        assert trace.escalations == []

    def test_policy_from_config(self, sample_registry: ModelRegistry):
        with patch.dict(os.environ, {"TOKENWISE_ESCALATION_POLICY": "monotonic"}):
            reset_settings()
            router = Router(sample_registry)
            _, trace = router.route_with_trace("Write a Python sort function")
            assert trace.escalation_policy == EscalationPolicy.MONOTONIC

    def test_level_from_config(self, sample_registry: ModelRegistry):
        with patch.dict(os.environ, {"TOKENWISE_TRACE_LEVEL": "verbose"}):
            reset_settings()
            router = Router(sample_registry)
            _, trace = router.route_with_trace("Write a Python sort function")
            assert trace.trace_level == TraceLevel.VERBOSE

    def test_budget_fields_with_budget(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        _, trace = router.route_with_trace("Write a Python sort function", budget=1.0)
        assert trace.budget_used >= 0
        assert trace.budget_remaining >= 0
        assert trace.budget_used + trace.budget_remaining == pytest.approx(1.0)

    def test_budget_fields_without_budget(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        _, trace = router.route_with_trace("Write a Python sort function")
        # Without explicit budget, these stay at defaults
        assert trace.budget_used == 0.0
        assert trace.budget_remaining == 0.0


class TestRouteWithTraceRiskGate:
    def test_risk_gate_raises_with_trace(self, sample_registry: ModelRegistry):
        with patch.dict(os.environ, {"TOKENWISE_RISK_GATE_ENABLED": "true"}):
            reset_settings()
            router = Router(sample_registry)
            with pytest.raises(RiskGateBlockedError) as exc_info:
                router.route_with_trace("delete the production database")
            assert exc_info.value.trace is not None
            assert exc_info.value.trace.termination_state == TerminationState.NO_GO

    def test_risk_gate_disabled_passes(self, sample_registry: ModelRegistry):
        router = Router(sample_registry)
        model, trace = router.route_with_trace("delete everything from the list in Python")
        assert isinstance(model, ModelInfo)
        assert trace.termination_state == TerminationState.COMPLETED
