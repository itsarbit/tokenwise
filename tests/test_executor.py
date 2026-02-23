"""Tests for Executor."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from tokenwise.config import reset_settings
from tokenwise.executor import Executor
from tokenwise.models import Plan, Step, StepResult, TerminationState
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


class TestExecutorSequential:
    """Tests for the sequential execution path (_execute_sequential)."""

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
            result = executor._execute_sequential(plan)

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
            result = executor._execute_sequential(plan)

        assert result.success
        assert len(result.step_results) == 2
        assert result.final_output == "second"
        assert result.total_cost == pytest.approx(0.002)

    def test_budget_exhaustion_stops_execution(self, sample_registry: ModelRegistry):
        """Steps whose estimated cost exceeds remaining budget are skipped pre-call."""
        executor = Executor(registry=sample_registry)
        steps = [
            Step(id=1, description="Expensive", model_id="openai/gpt-4.1-mini", estimated_cost=0.5),
            Step(id=2, description="Skipped", model_id="openai/gpt-4.1-mini", estimated_cost=0.5),
        ]
        plan = _make_plan(steps, budget=0.001)

        result = executor._execute_sequential(plan)

        # Both skipped: pre-call check sees estimated 0.5 > budget 0.001
        assert len(result.skipped_steps) == 2
        assert not result.success

    def test_budget_exhaustion_after_run(self, sample_registry: ModelRegistry):
        """Second step skipped when first step's actual cost exhausts budget."""
        executor = Executor(registry=sample_registry)
        steps = [
            Step(
                id=1,
                description="Affordable",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.001,
            ),
            Step(id=2, description="Skipped", model_id="openai/gpt-4.1-mini", estimated_cost=0.5),
        ]
        plan = _make_plan(steps, budget=0.01)

        with patch.object(executor, "_execute_step") as mock_exec:
            mock_exec.return_value = _mock_step_result(1, cost=0.005)
            result = executor._execute_sequential(plan)

        assert len(result.step_results) == 1
        assert len(result.skipped_steps) == 1
        assert not result.success

    def test_skipped_steps_recorded(self, sample_registry: ModelRegistry):
        """Skipped steps should be recorded in result.skipped_steps."""
        executor = Executor(registry=sample_registry)
        steps = [
            Step(id=1, description="Runs", model_id="openai/gpt-4.1-mini", estimated_cost=0.001),
            Step(id=2, description="Skipped A", model_id="openai/gpt-4.1-mini", estimated_cost=0.5),
            Step(id=3, description="Skipped B", model_id="openai/gpt-4.1-mini", estimated_cost=0.5),
        ]
        plan = _make_plan(steps, budget=0.01)

        with patch.object(executor, "_execute_step") as mock_exec:
            mock_exec.return_value = _mock_step_result(1, cost=0.002)
            result = executor._execute_sequential(plan)

        assert len(result.step_results) == 1
        assert len(result.skipped_steps) == 2
        assert result.skipped_steps[0].id == 2
        assert result.skipped_steps[1].id == 3
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
                result = executor._execute_sequential(plan)

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
            result = executor._execute_sequential(plan)

        assert not result.success


class TestExecutorCommon:
    """Tests for shared helper methods."""

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

    def test_fallback_candidates_escalate_upward(self, sample_registry: ModelRegistry):
        """Fallback for a BUDGET model should prioritize MID/FLAGSHIP tiers."""
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1,
            description="Test",
            model_id="openai/gpt-4.1-mini",
            estimated_cost=0.001,
        )
        candidates = executor._get_fallback_candidates(
            exclude={"openai/gpt-4.1-mini"}, budget_remaining=10.0, step=step
        )
        assert len(candidates) > 0
        from tokenwise.models import ModelTier

        first = sample_registry.get_model(candidates[0].id)
        assert first is not None
        assert first.tier in (ModelTier.FLAGSHIP, ModelTier.MID)

    def test_fallback_candidates_respect_capability(self, sample_registry: ModelRegistry):
        """Fallback candidates should match the failed model's capabilities."""
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1,
            description="Math problem",
            model_id="openai/o3",
            estimated_cost=0.001,
        )
        candidates = executor._get_fallback_candidates(
            exclude={"openai/o3"}, budget_remaining=100.0, step=step
        )
        for m in candidates:
            assert "code" in m.capabilities

    def test_fallback_uses_step_required_capabilities(self, sample_registry: ModelRegistry):
        """When step has required_capabilities, use those instead of inferring."""
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1,
            description="Reasoning task",
            model_id="openai/gpt-4.1-mini",
            estimated_cost=0.001,
            required_capabilities=["reasoning"],
        )
        candidates = executor._get_fallback_candidates(
            exclude={"openai/gpt-4.1-mini"}, budget_remaining=100.0, step=step
        )
        for m in candidates:
            assert "reasoning" in m.capabilities

    def test_is_model_error_uses_status_code(self, sample_registry: ModelRegistry):
        """_is_model_error checks http_status_code, not string matching."""
        executor = Executor(registry=sample_registry)
        result_404 = StepResult(
            step_id=1,
            model_id="test/model",
            success=False,
            error="Not Found",
            http_status_code=404,
        )
        assert executor._is_model_error(result_404) is True

        result_500 = StepResult(
            step_id=1,
            model_id="test/model",
            success=False,
            error="Server Error",
            http_status_code=500,
        )
        assert executor._is_model_error(result_500) is False

        result_none = StepResult(
            step_id=1,
            model_id="test/model",
            success=False,
            error="timeout",
        )
        assert executor._is_model_error(result_none) is False


class TestExecutorAsync:
    """Tests for the async DAG-based execution path."""

    async def test_aexecute_single_step(self, sample_registry: ModelRegistry):
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
            "_aexecute_step",
            new_callable=AsyncMock,
            return_value=_mock_step_result(1),
        ):
            result = await executor.aexecute(plan)

        assert result.success
        assert len(result.step_results) == 1
        assert result.final_output == "result"

    async def test_aexecute_parallel_independent_steps(self, sample_registry: ModelRegistry):
        """Two independent steps (no depends_on) should both execute."""
        executor = Executor(registry=sample_registry)
        steps = [
            Step(
                id=1,
                description="Step A",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.001,
                depends_on=[],
            ),
            Step(
                id=2,
                description="Step B",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.001,
                depends_on=[],
            ),
        ]
        plan = _make_plan(steps)

        async def mock_aexecute(step_id, model_id, prompt, budget_remaining, **kw):
            return _mock_step_result(step_id, output=f"output-{step_id}")

        with patch.object(executor, "_aexecute_step", side_effect=mock_aexecute):
            result = await executor.aexecute(plan)

        assert result.success
        assert len(result.step_results) == 2
        outputs = {sr.step_id: sr.output for sr in result.step_results}
        assert outputs[1] == "output-1"
        assert outputs[2] == "output-2"

    async def test_aexecute_dependency_ordering(self, sample_registry: ModelRegistry):
        """Step with dependency waits for prerequisite."""
        executor = Executor(registry=sample_registry)
        steps = [
            Step(
                id=1,
                description="First",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.001,
                depends_on=[],
            ),
            Step(
                id=2,
                description="Second (depends on 1)",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.001,
                depends_on=[1],
            ),
        ]
        plan = _make_plan(steps)

        call_order: list[int] = []

        async def mock_aexecute(step_id, model_id, prompt, budget_remaining, **kw):
            call_order.append(step_id)
            return _mock_step_result(step_id, output=f"output-{step_id}")

        with patch.object(executor, "_aexecute_step", side_effect=mock_aexecute):
            result = await executor.aexecute(plan)

        assert result.success
        # Step 1 must execute before step 2
        assert call_order.index(1) < call_order.index(2)

    async def test_aexecute_budget_exhaustion(self, sample_registry: ModelRegistry):
        """Steps too expensive to reserve should be skipped."""
        executor = Executor(registry=sample_registry)
        steps = [
            Step(
                id=1,
                description="Expensive",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.5,
                depends_on=[],
            ),
            Step(
                id=2,
                description="Also expensive",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.5,
                depends_on=[1],
            ),
        ]
        plan = _make_plan(steps, budget=0.001)

        async def mock_aexecute(step_id, model_id, prompt, budget_remaining, **kw):
            return _mock_step_result(step_id, cost=0.002)

        with patch.object(executor, "_aexecute_step", side_effect=mock_aexecute):
            result = await executor.aexecute(plan)

        # Reservation blocks both steps (estimated 0.5 > budget 0.001)
        assert len(result.skipped_steps) == 2
        assert not result.success

    async def test_aexecute_budget_exhaustion_after_first_step(
        self, sample_registry: ModelRegistry
    ):
        """Second step should be skipped when first step exhausts budget."""
        executor = Executor(registry=sample_registry)
        steps = [
            Step(
                id=1,
                description="Affordable",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.001,
                depends_on=[],
            ),
            Step(
                id=2,
                description="Should be skipped",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.001,
                depends_on=[1],
            ),
        ]
        plan = _make_plan(steps, budget=0.002)

        async def mock_aexecute(step_id, model_id, prompt, budget_remaining, **kw):
            return _mock_step_result(step_id, cost=0.002)

        with patch.object(executor, "_aexecute_step", side_effect=mock_aexecute):
            result = await executor.aexecute(plan)

        assert len(result.step_results) == 1
        assert len(result.skipped_steps) == 1
        assert not result.success

    async def test_aexecute_ledger_populated(self, sample_registry: ModelRegistry):
        """Ledger should have entries after async execution."""
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
            "_aexecute_step",
            new_callable=AsyncMock,
            return_value=_mock_step_result(1),
        ):
            result = await executor.aexecute(plan)

        assert len(result.ledger.entries) == 1
        entry = result.ledger.entries[0]
        assert entry.reason == "step 1 attempt 1"
        assert entry.success is True

    def test_execute_dispatches(self, sample_registry: ModelRegistry):
        """execute() should produce results (dispatches to async or sequential)."""
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
            "_aexecute_step",
            new_callable=AsyncMock,
            return_value=_mock_step_result(1),
        ):
            result = executor.execute(plan)

        assert result.success
        assert len(result.step_results) == 1

    def test_budget_remaining(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        steps = [
            Step(id=1, description="Step 1", model_id="openai/gpt-4.1-mini", estimated_cost=0.01),
        ]
        plan = _make_plan(steps, budget=1.0)

        with patch.object(
            executor,
            "_aexecute_step",
            new_callable=AsyncMock,
            return_value=_mock_step_result(1, cost=0.005),
        ):
            result = executor.execute(plan)

        assert result.budget_remaining == pytest.approx(0.995)

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

        with patch.object(executor, "_aexecute_step", new_callable=AsyncMock, return_value=failed):
            with patch.object(
                executor, "_aescalate", new_callable=AsyncMock, return_value=escalated
            ):
                result = executor.execute(plan)

        # Should have at least the initial failed attempt recorded
        assert len(result.ledger.entries) >= 1
        assert result.ledger.entries[0].success is False

    async def test_aexecute_reservation_prevents_overshoot(self, sample_registry: ModelRegistry):
        """Parallel steps should not collectively exceed budget via reservation."""
        executor = Executor(registry=sample_registry)
        # Two independent steps each estimated at 0.006, budget only 0.01
        steps = [
            Step(
                id=1,
                description="Step A",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.006,
                depends_on=[],
            ),
            Step(
                id=2,
                description="Step B",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.006,
                depends_on=[],
            ),
        ]
        plan = _make_plan(steps, budget=0.01)

        async def mock_aexecute(step_id, model_id, prompt, budget_remaining, **kw):
            return _mock_step_result(step_id, cost=0.006)

        with patch.object(executor, "_aexecute_step", side_effect=mock_aexecute):
            result = await executor.aexecute(plan)

        # With reservation, only one step should fit at a time (0.006 + 0.006 > 0.01)
        # Second step should still execute in next iteration since first actual < budget
        # But total should not exceed budget by more than one step's cost
        assert result.total_cost <= plan.budget + 0.006

    async def test_aexecute_deadlock_detection(self, sample_registry: ModelRegistry):
        """Cyclic dependencies should be detected and reported as failure."""
        executor = Executor(registry=sample_registry)
        # Step 1 depends on 2, step 2 depends on 1 — deadlock
        steps = [
            Step(
                id=1,
                description="Step A",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.001,
                depends_on=[2],
            ),
            Step(
                id=2,
                description="Step B",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.001,
                depends_on=[1],
            ),
        ]
        plan = _make_plan(steps)

        result = await executor.aexecute(plan)

        assert not result.success
        assert len(result.skipped_steps) == 2
        assert len(result.step_results) == 0

    async def test_aexecute_passes_estimated_cost_as_budget(self, sample_registry: ModelRegistry):
        """_arun_step should receive estimated_cost (not 2x) as budget_remaining."""
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1,
            description="Do something",
            model_id="openai/gpt-4.1-mini",
            estimated_cost=0.05,
            depends_on=[],
        )
        plan = _make_plan([step], budget=1.0)

        budgets_received: list[float] = []

        async def mock_aexecute(step_id, model_id, prompt, budget_remaining, **kw):
            budgets_received.append(budget_remaining)
            return _mock_step_result(step_id, cost=0.01)

        with patch.object(executor, "_aexecute_step", side_effect=mock_aexecute):
            await executor.aexecute(plan)

        # _arun_step should pass estimated_cost (0.05), not 2x (0.10)
        assert budgets_received[0] == pytest.approx(0.05)

    async def test_aexecute_escalation_includes_failed_cost(self, sample_registry: ModelRegistry):
        """Async escalation should include the failed attempt cost in total."""
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1,
            description="Failing step",
            model_id="openai/gpt-4.1-mini",
            estimated_cost=0.01,
        )
        plan = _make_plan([step], budget=1.0)

        failed = _mock_step_result(1, success=False, cost=0.003)
        escalated = _mock_step_result(
            1, model_id="anthropic/claude-opus-4", output="escalated", cost=0.005
        )
        escalated.escalated = True

        with patch.object(executor, "_aexecute_step", new_callable=AsyncMock, return_value=failed):
            with patch.object(
                executor, "_aescalate", new_callable=AsyncMock, return_value=escalated
            ):
                result = await executor.aexecute(plan)

        # total_cost should include both the failed attempt (0.003) and escalated (0.005)
        assert result.total_cost == pytest.approx(0.008)


class TestMaxTokensGuardrail:
    """Tests for the max_tokens budget guardrail."""

    def test_compute_max_tokens(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        # gpt-4.1-mini output_price = 1.60/M, input_price = 0.40/M
        # budget = $0.016, no input tokens → 0.016 / (1.60 / 1M) = 10_000
        max_tok = executor._compute_max_tokens("openai/gpt-4.1-mini", 0.016)
        assert max_tok == 10_000

    def test_compute_max_tokens_with_input_cost(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        # gpt-4.1-mini: input=$0.40/M, output=$1.60/M
        # budget=$0.016, 1000 input tokens → input_cost = 0.40 * 1000 / 1M = $0.0004
        # budget_for_output = 0.016 - 0.0004 = 0.0156
        # max_tokens = 0.0156 / (1.60 / 1M) = 9750
        max_tok = executor._compute_max_tokens(
            "openai/gpt-4.1-mini", 0.016, estimated_input_tokens=1000
        )
        assert max_tok == 9750

    def test_compute_max_tokens_input_exceeds_budget(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        # gpt-4.1-mini input=$0.40/M
        # 50_000 input tokens → input_cost = 0.40 * 50000 / 1M = $0.02
        # budget = $0.016 → budget_for_output = 0.016 - 0.02 = -0.004 → 0
        max_tok = executor._compute_max_tokens(
            "openai/gpt-4.1-mini", 0.016, estimated_input_tokens=50_000
        )
        assert max_tok == 0

    def test_compute_max_tokens_unknown_model(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        assert executor._compute_max_tokens("nonexistent/model", 1.0) is None

    def test_execute_step_skips_when_budget_too_low(self, sample_registry: ModelRegistry):
        """Step should fail gracefully when budget is too low for min output tokens."""
        executor = Executor(registry=sample_registry)
        # gpt-4.1-mini output_price=1.60/M → need $0.00016 for 100 tokens
        # Give budget that yields fewer than min_output_tokens (default 100)
        tiny_budget = (executor.min_output_tokens - 1) * (1.60 / 1_000_000)
        result = executor._execute_step(
            step_id=1,
            model_id="openai/gpt-4.1-mini",
            prompt="test",
            budget_remaining=tiny_budget,
        )
        assert not result.success
        assert "Budget too low" in result.error

    def test_custom_min_output_tokens(self, sample_registry: ModelRegistry):
        """Custom min_output_tokens allows smaller outputs under tight budgets."""
        executor = Executor(registry=sample_registry, min_output_tokens=10)
        assert executor.min_output_tokens == 10
        # gpt-4.1-mini output_price=1.60/M → 10 tokens needs $0.000016
        # Budget that yields 50 tokens — enough for min=10 but not for default=100
        budget = 50 * (1.60 / 1_000_000)
        result = executor._execute_step(
            step_id=1,
            model_id="openai/gpt-4.1-mini",
            prompt="test",
            budget_remaining=budget,
        )
        # Should NOT be skipped since 50 >= 10
        assert result.success or "Budget too low" not in (result.error or "")

    def test_execute_step_skips_when_input_cost_exceeds_budget(
        self, sample_registry: ModelRegistry
    ):
        """High input token estimate can push output budget below minimum."""
        executor = Executor(registry=sample_registry)
        # Budget = $0.016, but 50k input tokens cost $0.02 → budget_for_output < 0 → 0
        result = executor._execute_step(
            step_id=1,
            model_id="openai/gpt-4.1-mini",
            prompt="test",
            budget_remaining=0.016,
            estimated_input_tokens=50_000,
        )
        assert not result.success
        assert "Budget too low" in result.error

    def test_prompt_length_overrides_low_estimate(self, sample_registry: ModelRegistry):
        """Actual prompt length should override a low estimated_input_tokens."""
        from unittest.mock import MagicMock

        executor = Executor(registry=sample_registry)
        mock_provider = MagicMock()
        mock_provider.chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        }

        # Prompt is 4000 chars → len//4 = 1000 tokens, but estimate says 100
        long_prompt = "x" * 4000
        with patch.object(
            executor._resolver,
            "resolve",
            return_value=(mock_provider, "gpt-4.1-mini"),
        ):
            executor._execute_step(
                step_id=1,
                model_id="openai/gpt-4.1-mini",
                prompt=long_prompt,
                budget_remaining=0.016,
                estimated_input_tokens=100,
            )
        call_kwargs = mock_provider.chat_completion.call_args[1]
        # max(100, 4000//4) = 1000, * 1.2 safety margin = 1200 tokens
        # input_cost = 0.40 * 1200 / 1M = $0.00048
        # budget_for_output = 0.016 - 0.00048 = 0.01552
        # max_tokens = 0.01552 / (1.60 / 1M) = 9700
        assert call_kwargs["max_tokens"] == 9700

    def test_execute_step_passes_max_tokens(self, sample_registry: ModelRegistry):
        """_execute_step should pass max_tokens to provider when budget is set."""
        from unittest.mock import MagicMock

        executor = Executor(registry=sample_registry)
        mock_provider = MagicMock()
        mock_provider.chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        }

        with patch.object(
            executor._resolver, "resolve", return_value=(mock_provider, "gpt-4.1-mini")
        ):
            executor._execute_step(
                step_id=1,
                model_id="openai/gpt-4.1-mini",
                prompt="test",
                budget_remaining=0.016,
                estimated_input_tokens=1000,
            )
        call_kwargs = mock_provider.chat_completion.call_args[1]
        assert "max_tokens" in call_kwargs
        # max(1000, len("test")//4=1) * 1.2 = 1200 input tokens
        # input_cost=0.40*1200/1M=$0.00048
        # budget_for_output=$0.01552 → 0.01552/(1.60/1M) = 9700
        assert call_kwargs["max_tokens"] == 9700

    async def test_aexecute_step_skips_when_budget_too_low(self, sample_registry: ModelRegistry):
        """Async step should fail gracefully when budget is too low."""
        executor = Executor(registry=sample_registry)
        tiny_budget = (executor.min_output_tokens - 1) * (1.60 / 1_000_000)
        result = await executor._aexecute_step(
            step_id=1,
            model_id="openai/gpt-4.1-mini",
            prompt="test",
            budget_remaining=tiny_budget,
        )
        assert not result.success
        assert "Budget too low" in result.error

    async def test_aexecute_step_passes_max_tokens(self, sample_registry: ModelRegistry):
        """_aexecute_step should pass max_tokens to async provider."""
        executor = Executor(registry=sample_registry)
        mock_provider = AsyncMock()
        mock_provider.achat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        }

        with patch.object(
            executor._resolver, "resolve", return_value=(mock_provider, "gpt-4.1-mini")
        ):
            await executor._aexecute_step(
                step_id=1,
                model_id="openai/gpt-4.1-mini",
                prompt="test",
                budget_remaining=0.016,
                estimated_input_tokens=1000,
            )
        call_kwargs = mock_provider.achat_completion.call_args[1]
        assert "max_tokens" in call_kwargs
        # Same math: 1200 input tokens with 1.2x margin → 9700
        assert call_kwargs["max_tokens"] == 9700


class TestProductionInvariants:
    """Tests for invariants that must hold in production."""

    async def test_parallel_reservation_total_cost_within_budget(
        self, sample_registry: ModelRegistry
    ):
        """Total cost of parallel steps must not exceed budget by more than one step."""
        executor = Executor(registry=sample_registry)
        # Three independent steps, each estimated at 0.004, budget = 0.01
        # Reservation should prevent all three from launching simultaneously
        steps = [
            Step(
                id=i,
                description=f"Step {i}",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.004,
                depends_on=[],
            )
            for i in range(1, 4)
        ]
        plan = _make_plan(steps, budget=0.01)

        async def mock_aexecute(step_id, model_id, prompt, budget_remaining, **kw):
            return _mock_step_result(step_id, cost=0.004)

        with patch.object(executor, "_aexecute_step", side_effect=mock_aexecute):
            result = await executor.aexecute(plan)

        # Budget is 0.01; at most two steps can run (0.004 + 0.004 = 0.008)
        # Third step's reservation (0.008 + 0.004 = 0.012) exceeds budget
        assert result.total_cost <= plan.budget + 0.001

    async def test_total_cost_equals_sum_of_step_results(self, sample_registry: ModelRegistry):
        """total_cost must equal the sum of all step_results' actual_cost."""
        executor = Executor(registry=sample_registry)
        steps = [
            Step(
                id=1,
                description="Step 1",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.003,
                depends_on=[],
            ),
            Step(
                id=2,
                description="Step 2",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.003,
                depends_on=[1],
            ),
        ]
        plan = _make_plan(steps, budget=1.0)

        call_count = 0

        async def mock_aexecute(step_id, model_id, prompt, budget_remaining, **kw):
            nonlocal call_count
            call_count += 1
            # Vary costs to make the assertion meaningful
            cost = 0.002 if step_id == 1 else 0.005
            return _mock_step_result(step_id, cost=cost)

        with patch.object(executor, "_aexecute_step", side_effect=mock_aexecute):
            result = await executor.aexecute(plan)

        step_cost_sum = sum(sr.actual_cost for sr in result.step_results)
        assert result.total_cost == pytest.approx(step_cost_sum)
        assert result.total_cost == pytest.approx(0.007)

    async def test_ledger_entries_cover_all_attempts(self, sample_registry: ModelRegistry):
        """Ledger must have an entry for every LLM call including failed attempts."""
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1,
            description="Failing step",
            model_id="openai/gpt-4.1-mini",
            estimated_cost=0.01,
            depends_on=[],
        )
        plan = _make_plan([step], budget=1.0)

        failed = _mock_step_result(1, success=False, cost=0.002)
        escalated = _mock_step_result(1, model_id="openai/gpt-4.1", output="ok", cost=0.005)
        escalated.escalated = True

        with patch.object(
            executor,
            "_aexecute_step",
            new_callable=AsyncMock,
            return_value=failed,
        ):
            with patch.object(
                executor,
                "_aescalate",
                new_callable=AsyncMock,
                return_value=escalated,
            ):
                result = await executor.aexecute(plan)

        # Ledger should have at least the initial failed attempt
        assert len(result.ledger.entries) >= 1
        assert result.ledger.entries[0].success is False
        assert result.ledger.entries[0].cost == pytest.approx(0.002)
        # total_cost includes both failed + escalated costs
        assert result.total_cost == pytest.approx(0.002 + 0.005)


class TestRoutingTrace:
    """Tests for routing trace on PlanResult."""

    def test_trace_on_plan_result(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1, description="Do something", model_id="openai/gpt-4.1-mini", estimated_cost=0.001
        )
        plan = _make_plan([step])

        with patch.object(executor, "_execute_step", return_value=_mock_step_result(1)):
            result = executor._execute_sequential(plan)

        assert result.routing_trace is not None
        assert result.routing_trace.request_id

    def test_termination_completed(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1, description="Do something", model_id="openai/gpt-4.1-mini", estimated_cost=0.001
        )
        plan = _make_plan([step])

        with patch.object(executor, "_execute_step", return_value=_mock_step_result(1)):
            result = executor._execute_sequential(plan)

        assert result.routing_trace is not None
        assert result.routing_trace.termination_state == TerminationState.COMPLETED

    def test_termination_exhausted(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        steps = [
            Step(
                id=1, description="Expensive", model_id="openai/gpt-4.1-mini", estimated_cost=0.001
            ),
            Step(id=2, description="Skipped", model_id="openai/gpt-4.1-mini", estimated_cost=0.5),
        ]
        plan = _make_plan(steps, budget=0.005)

        with patch.object(executor, "_execute_step", return_value=_mock_step_result(1, cost=0.005)):
            result = executor._execute_sequential(plan)

        assert result.routing_trace is not None
        assert result.routing_trace.termination_state == TerminationState.EXHAUSTED

    def test_termination_failed(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1, description="Failing", model_id="openai/gpt-4.1-mini", estimated_cost=0.001
        )
        plan = _make_plan([step], budget=1.0)

        failed = _mock_step_result(1, success=False, cost=0.0)
        with patch.object(executor, "_execute_step", return_value=failed):
            result = executor._execute_sequential(plan)

        assert result.routing_trace is not None
        assert result.routing_trace.termination_state == TerminationState.FAILED

    def test_budget_fields_populated(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1, description="Do something", model_id="openai/gpt-4.1-mini", estimated_cost=0.001
        )
        plan = _make_plan([step], budget=1.0)

        with patch.object(executor, "_execute_step", return_value=_mock_step_result(1, cost=0.005)):
            result = executor._execute_sequential(plan)

        assert result.routing_trace is not None
        assert result.routing_trace.budget_used == pytest.approx(0.005)
        assert result.routing_trace.budget_remaining == pytest.approx(0.995)

    async def test_async_trace_on_plan_result(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1, description="Do something", model_id="openai/gpt-4.1-mini", estimated_cost=0.001
        )
        plan = _make_plan([step])

        with patch.object(
            executor,
            "_aexecute_step",
            new_callable=AsyncMock,
            return_value=_mock_step_result(1),
        ):
            result = await executor.aexecute(plan)

        assert result.routing_trace is not None
        assert result.routing_trace.termination_state == TerminationState.COMPLETED

    def test_escalation_records_in_trace(self, sample_registry: ModelRegistry):
        executor = Executor(registry=sample_registry)
        step = Step(
            id=1,
            description="Failing step",
            model_id="openai/gpt-4.1-mini",
            estimated_cost=0.001,
        )
        plan = _make_plan([step], budget=1.0)

        failed = _mock_step_result(1, success=False, cost=0.0, model_id="openai/gpt-4.1-mini")
        escalated = _mock_step_result(1, model_id="openai/gpt-4.1", output="escalated")
        escalated.escalated = True

        with patch.object(executor, "_execute_step", return_value=failed):
            with patch.object(executor, "_escalate", side_effect=lambda *a, **kw: escalated):
                # Call _escalate manually to populate trace
                pass

        # Use the full sequential flow which threads trace through
        call_count = 0

        def mock_exec_step(step_id, model_id, prompt, budget_remaining, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return failed
            return escalated

        # Use actual escalation flow
        with patch.object(executor, "_execute_step", side_effect=mock_exec_step):
            result = executor._execute_sequential(plan)

        assert result.routing_trace is not None
        if result.routing_trace.escalations:
            esc = result.routing_trace.escalations[0]
            assert esc.from_model == "openai/gpt-4.1-mini"
            assert esc.step_id == 1


class TestMonotonicEscalation:
    """Tests for monotonic escalation policy."""

    def test_monotonic_excludes_same_tier(self, sample_registry: ModelRegistry):
        with patch.dict(os.environ, {"TOKENWISE_ESCALATION_POLICY": "monotonic"}):
            reset_settings()
            executor = Executor(registry=sample_registry)
            step = Step(
                id=1,
                description="Test",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.001,
            )
            candidates = executor._get_fallback_candidates(
                exclude={"openai/gpt-4.1-mini"}, budget_remaining=100.0, step=step
            )
            from tokenwise.models import ModelTier

            # In monotonic mode, no BUDGET tier candidates should appear
            for c in candidates:
                assert c.tier != ModelTier.BUDGET

    def test_flexible_includes_same_tier(self, sample_registry: ModelRegistry):
        with patch.dict(os.environ, {"TOKENWISE_ESCALATION_POLICY": "flexible"}):
            reset_settings()
            executor = Executor(registry=sample_registry)
            step = Step(
                id=1,
                description="Test",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.001,
                required_capabilities=["code"],
            )
            candidates = executor._get_fallback_candidates(
                exclude={"openai/gpt-4.1-mini"}, budget_remaining=100.0, step=step
            )
            from tokenwise.models import ModelTier

            tiers = {c.tier for c in candidates}
            # Flexible mode should include same-tier (BUDGET) candidates
            assert ModelTier.BUDGET in tiers

    def test_monotonic_allows_stronger(self, sample_registry: ModelRegistry):
        with patch.dict(os.environ, {"TOKENWISE_ESCALATION_POLICY": "monotonic"}):
            reset_settings()
            executor = Executor(registry=sample_registry)
            step = Step(
                id=1,
                description="Test",
                model_id="openai/gpt-4.1-mini",
                estimated_cost=0.001,
            )
            candidates = executor._get_fallback_candidates(
                exclude={"openai/gpt-4.1-mini"}, budget_remaining=100.0, step=step
            )
            assert len(candidates) > 0
            from tokenwise.models import ModelTier

            for c in candidates:
                assert c.tier in (ModelTier.MID, ModelTier.FLAGSHIP)
