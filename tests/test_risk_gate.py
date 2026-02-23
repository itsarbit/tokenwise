"""Tests for the risk gate module."""

from __future__ import annotations

from tokenwise.config import RiskGateConfig
from tokenwise.risk_gate import _compute_ambiguity_score, evaluate_risk


class TestRiskGateDisabled:
    def test_disabled_passes_all(self):
        config = RiskGateConfig(enabled=False)
        result = evaluate_risk("delete the production database", config)
        assert not result.blocked

    def test_disabled_passes_ambiguous(self):
        config = RiskGateConfig(enabled=False)
        result = evaluate_risk("do it", config)
        assert not result.blocked


class TestIrreversiblePatterns:
    def test_delete_database_blocked(self):
        config = RiskGateConfig(enabled=True, irreversible_block=True)
        result = evaluate_risk("delete the production database", config)
        assert result.blocked
        assert "irreversible" in result.reason.lower()

    def test_rm_rf_blocked(self):
        config = RiskGateConfig(enabled=True, irreversible_block=True)
        result = evaluate_risk("run rm -rf /", config)
        assert result.blocked

    def test_drop_table_blocked(self):
        config = RiskGateConfig(enabled=True, irreversible_block=True)
        result = evaluate_risk("please drop table users", config)
        assert result.blocked

    def test_truncate_table_blocked(self):
        config = RiskGateConfig(enabled=True, irreversible_block=True)
        result = evaluate_risk("truncate table logs", config)
        assert result.blocked

    def test_destroy_infrastructure_blocked(self):
        config = RiskGateConfig(enabled=True, irreversible_block=True)
        result = evaluate_risk("destroy the production cluster", config)
        assert result.blocked

    def test_safe_query_passes(self):
        config = RiskGateConfig(enabled=True, irreversible_block=True)
        result = evaluate_risk(
            "Write a Python function that sorts a list of integers using merge sort", config
        )
        assert not result.blocked

    def test_irreversible_block_disabled(self):
        config = RiskGateConfig(enabled=True, irreversible_block=False)
        result = evaluate_risk("delete the production database", config)
        # With irreversible_block=False, should not block on patterns
        # May still block on ambiguity if query is short enough
        assert "irreversible" not in result.reason.lower()


class TestAmbiguityThreshold:
    def test_very_short_query_blocked(self):
        config = RiskGateConfig(enabled=True, ambiguity_threshold=0.7)
        result = evaluate_risk("do it", config)
        assert result.blocked
        assert "ambiguous" in result.reason.lower()

    def test_clear_query_passes(self):
        config = RiskGateConfig(enabled=True, ambiguity_threshold=0.9)
        result = evaluate_risk(
            "Write a Python function that takes a list of integers and returns "
            "the sum of all even numbers",
            config,
        )
        assert not result.blocked

    def test_custom_threshold(self):
        """Lower threshold blocks more queries."""
        config_strict = RiskGateConfig(enabled=True, ambiguity_threshold=0.3)
        result_strict = evaluate_risk("sort my list", config_strict)
        assert result_strict.blocked

        config_lax = RiskGateConfig(enabled=True, ambiguity_threshold=0.99)
        result_lax = evaluate_risk("sort my list", config_lax)
        assert not result_lax.blocked

    def test_ambiguity_pattern_match(self):
        config = RiskGateConfig(enabled=True, ambiguity_threshold=0.8)
        result = evaluate_risk("go ahead", config)
        assert result.blocked
        assert result.ambiguity_score > 0.8


class TestAmbiguityScoring:
    def test_very_short_query_high_score(self):
        score = _compute_ambiguity_score("ok")
        assert score >= 0.8

    def test_medium_query_moderate_score(self):
        score = _compute_ambiguity_score("sort this list please")
        assert 0.3 <= score <= 0.7

    def test_long_query_low_score(self):
        score = _compute_ambiguity_score(
            "Write a Python function that implements the quicksort algorithm "
            "with a partition scheme that handles duplicate elements efficiently"
        )
        assert score <= 0.3

    def test_score_bounded(self):
        score = _compute_ambiguity_score("do it")
        assert 0.0 <= score <= 1.0
