"""Tests for Executor."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tokenwise.executor import Executor
from tokenwise.models import Plan, Step, StepResult
from tokenwise.registry import ModelRegistry


def _make_plan(
    steps: list[Step], budget: float = 1.0, task: str = "test task"
) -> Plan:
    total = sum(s.estimated_cost for s in steps)
    return Plan(
        task=task, steps=steps, total_estimated_cost=total, budget=budget
    )


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
            Step(id=1, description="Step 1", model_id="openai/gpt-4.1-mini",
                 estimated_cost=0.001),
            Step(id=2, description="Step 2", model_id="openai/gpt-4.1-mini",
                 estimated_cost=0.001, depends_on=[1]),
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

    def test_budget_exhaustion_stops_execution(
        self, sample_registry: ModelRegistry
    ):
        executor = Executor(registry=sample_registry)
        steps = [
            Step(id=1, description="Expensive", model_id="openai/gpt-4.1-mini",
                 estimated_cost=0.5),
            Step(id=2, description="Skipped", model_id="openai/gpt-4.1-mini",
                 estimated_cost=0.5),
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

    def test_escalation_skipped_when_no_budget(
        self, sample_registry: ModelRegistry
    ):
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

    def test_build_prompt_with_prior_outputs(
        self, sample_registry: ModelRegistry
    ):
        executor = Executor(registry=sample_registry)
        prompt = executor._build_prompt(
            "Do X", "", {1: "prior result"}
        )
        assert "Context from prior steps" in prompt
        assert "prior result" in prompt
        assert "Do X" in prompt

    def test_budget_remaining(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        steps = [
            Step(id=1, description="Step 1", model_id="openai/gpt-4.1-mini",
                 estimated_cost=0.01),
        ]
        plan = _make_plan(steps, budget=1.0)

        with patch.object(executor, "_execute_step") as mock_exec:
            mock_exec.return_value = _mock_step_result(1, cost=0.005)
            result = executor.execute(plan)

        assert result.budget_remaining == pytest.approx(0.995)
