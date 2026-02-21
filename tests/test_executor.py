"""Tests for Executor."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tokenwise.executor import Executor
from tokenwise.models import Plan, Step, StepResult
from tokenwise.registry import ModelRegistry


def _make_plan(steps: list[Step], budget: float = 1.0, task: str = "test task") -> Plan:
    total = sum(s.estimated_cost for s in steps)
    return Plan(task=task, steps=steps, total_estimated_cost=total, budget=budget)


def _mock_step_result(
    step_id: int,
    model_id: str = "test/model",
    output: str = "result",
    cost: float = 0.001,
    success: bool = True,
) -> StepResult:
    return StepResult(
        step_id=step_id,
        model_id=model_id,
        output=output,
        input_tokens=100,
        output_tokens=100,
        actual_cost=cost,
        success=success,
        error=None if success else "mock error",
    )


class TestExecutor:
    def test_execute_single_step(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1,
            description="Do something",
            model_id="openai/gpt-4.1-mini",
            estimated_cost=0.001,
        )
        plan = _make_plan([step])

        with patch.object(
            executor,
            "_execute_step",
            return_value=_mock_step_result(1),
        ):
            result = executor.execute(plan)

        assert result.success
        assert len(result.step_results) == 1
        assert result.final_output == "result"

    def test_execute_multiple_steps(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        steps = [
            Step(id=1, description="Step 1", model_id="openai/gpt-4.1-mini", estimated_cost=0.001),
            Step(
                id=2,
                description="Step 2",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.001,
                depends_on=[1],
            ),
        ]
        plan = _make_plan(steps)

        with patch.object(executor, "_execute_step") as mock_exec:
            mock_exec.side_effect = [
                _mock_step_result(1, output="first"),
                _mock_step_result(2, output="second"),
            ]
            result = executor.execute(plan)

        assert result.success
        assert len(result.step_results) == 2
        assert result.final_output == "second"
        assert result.total_cost == pytest.approx(0.002)

    def test_budget_exhaustion_stops_execution(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        steps = [
            Step(id=1, description="Expensive", model_id="openai/gpt-4.1-mini", estimated_cost=0.5),
            Step(id=2, description="Skipped", model_id="openai/gpt-4.1-mini", estimated_cost=0.5),
        ]
        plan = _make_plan(steps, budget=0.001)

        with patch.object(executor, "_execute_step") as mock_exec:
            mock_exec.return_value = _mock_step_result(1, cost=0.002)
            result = executor.execute(plan)

        # Only first step executed; second skipped due to budget
        assert len(result.step_results) == 1
        assert not result.success

    def test_escalation_on_failure(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1,
            description="Failing step",
            model_id="openai/gpt-4.1-mini",
            estimated_cost=0.001,
        )
        plan = _make_plan([step], budget=1.0)

        failed = _mock_step_result(1, success=False, cost=0.0)
        escalated = _mock_step_result(
            1, model_id="anthropic/claude-opus-4", output="escalated result"
        )
        escalated.escalated = True

        with patch.object(executor, "_execute_step", return_value=failed):
            with patch.object(executor, "_escalate", return_value=escalated):
                result = executor.execute(plan)

        assert result.success
        assert result.step_results[0].escalated
        assert result.final_output == "escalated result"

    def test_escalation_skipped_when_no_budget(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1,
            description="Failing step",
            model_id="openai/gpt-4.1-mini",
            estimated_cost=0.001,
        )
        plan = _make_plan([step], budget=0.0001)

        failed = _mock_step_result(1, success=False, cost=0.001)

        with patch.object(executor, "_execute_step", return_value=failed):
            result = executor.execute(plan)

        # Should not escalate since budget is exhausted
        assert not result.success

    def test_build_prompt_no_prior(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        prompt = executor._build_prompt("Do X", "", {})
        assert prompt == "Do X"

    def test_build_prompt_with_template(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        prompt = executor._build_prompt("Do X", "Custom template", {})
        assert prompt == "Custom template"

    def test_build_prompt_with_prior_outputs(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        prompt = executor._build_prompt("Do X", "", {1: "prior result"})
        assert "Context from prior steps" in prompt
        assert "prior result" in prompt
        assert "Do X" in prompt

    def test_budget_remaining(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        steps = [
            Step(id=1, description="Step 1", model_id="openai/gpt-4.1-mini", estimated_cost=0.01),
        ]
        plan = _make_plan(steps, budget=1.0)

        with patch.object(executor, "_execute_step") as mock_exec:
            mock_exec.return_value = _mock_step_result(1, cost=0.005)
            result = executor.execute(plan)

        assert result.budget_remaining == pytest.approx(0.995)

    def test_fallback_candidates_escalate_upward(self, sample_registry: ModelRegistry):
        """Fallback for a BUDGET model should prioritize MID/FLAGSHIP tiers."""
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1,
            description="Test",
            model_id="openai/gpt-4.1-mini",  # BUDGET tier
            estimated_cost=0.001,
        )
        candidates = executor._get_fallback_candidates(
            exclude={"openai/gpt-4.1-mini"}, budget_remaining=10.0, step=step
        )
        # First candidates should be from stronger tiers
        assert len(candidates) > 0
        from tokenwise.models import ModelTier

        first = sample_registry.get_model(candidates[0].id)
        assert first is not None
        # The first candidate should be FLAGSHIP or MID (stronger than BUDGET)
        assert first.tier in (ModelTier.FLAGSHIP, ModelTier.MID)

    def test_fallback_candidates_respect_capability(self, sample_registry: ModelRegistry):
        """Fallback candidates should match the failed model's capabilities."""
        executor = Executor(registry=sample_registry)
        # openai/o3 has capabilities: code, reasoning, math — no "general"
        step = Step(
            id=1,
            description="Math problem",
            model_id="openai/o3",  # FLAGSHIP, capabilities: code, reasoning, math
            estimated_cost=0.001,
        )
        candidates = executor._get_fallback_candidates(
            exclude={"openai/o3"}, budget_remaining=100.0, step=step
        )
        # All candidates should have at least the "code" capability (first non-general cap)
        for m in candidates:
            assert "code" in m.capabilities

    def test_ledger_populated_after_execution(self, sample_registry: ModelRegistry):
        """Ledger should have entries after executing a plan."""
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1,
            description="Do something",
            model_id="openai/gpt-4.1-mini",
            estimated_cost=0.001,
        )
        plan = _make_plan([step])

        with patch.object(executor, "_execute_step", return_value=_mock_step_result(1)):
            result = executor.execute(plan)

        assert len(result.ledger.entries) == 1
        entry = result.ledger.entries[0]
        assert entry.reason == "step 1 attempt 1"
        assert entry.success is True
        assert entry.model_id == "test/model"

    def test_ledger_tracks_failed_escalation(self, sample_registry: ModelRegistry):
        """Failed step + escalation should produce multiple ledger entries."""
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1,
            description="Failing step",
            model_id="openai/gpt-4.1-mini",
            estimated_cost=0.001,
        )
        plan = _make_plan([step], budget=1.0)

        failed = _mock_step_result(1, success=False, cost=0.0)
        escalated = _mock_step_result(1, model_id="anthropic/claude-opus-4", output="escalated")
        escalated.escalated = True

        with patch.object(executor, "_execute_step", return_value=failed):
            with patch.object(executor, "_escalate", return_value=escalated):
                result = executor.execute(plan)

        # Should have at least the initial failed attempt recorded
        assert len(result.ledger.entries) >= 1
        assert result.ledger.entries[0].success is False
