"""Tests for Planner."""

from __future__ import annotations

import json
from unittest.mock import patch

from tokenwise.planner import Planner, _DecomposeResult
from tokenwise.registry import ModelRegistry


class TestPlanner:
    def test_plan_with_fallback(self, sample_registry: ModelRegistry):
        """When LLM decomposition fails, fallback should produce a single-step plan."""
        planner = Planner(registry=sample_registry)

        fallback_steps = planner._fallback_decomposition("Test task")
        decompose_result = _DecomposeResult(
            steps=fallback_steps, source="fallback", error="LLM unavailable", planner_cost=0.0
        )
        with patch.object(planner, "_decompose_task", return_value=decompose_result):
            plan = planner.plan("Test task", budget=1.0)

        assert len(plan.steps) == 1
        assert plan.steps[0].description == "Test task"
        assert plan.budget == 1.0
        assert plan.is_within_budget()
        assert plan.decomposition_source == "fallback"
        assert plan.decomposition_error == "LLM unavailable"
        assert plan.planner_cost == 0.0

    def test_plan_assigns_models(self, sample_registry: ModelRegistry):
        """Each step should get a valid model from the registry."""
        planner = Planner(registry=sample_registry)

        raw_steps = [
            {
                "description": "Write code",
                "capability": "code",
                "estimated_input_tokens": 500,
                "estimated_output_tokens": 500,
            },
            {
                "description": "Review logic",
                "capability": "reasoning",
                "estimated_input_tokens": 800,
                "estimated_output_tokens": 400,
            },
        ]
        decompose_result = _DecomposeResult(steps=raw_steps, source="llm", planner_cost=0.0)
        with patch.object(planner, "_decompose_task", return_value=decompose_result):
            plan = planner.plan("Build something", budget=5.0)

        assert len(plan.steps) == 2
        assert plan.decomposition_source == "llm"
        assert plan.decomposition_error is None
        for step in plan.steps:
            model = sample_registry.get_model(step.model_id)
            assert model is not None

    def test_plan_respects_budget(self, sample_registry: ModelRegistry):
        """Plan should try to fit within budget."""
        planner = Planner(registry=sample_registry)

        raw_steps = [
            {
                "description": f"Step {i}",
                "capability": "general",
                "estimated_input_tokens": 500,
                "estimated_output_tokens": 500,
            }
            for i in range(5)
        ]
        decompose_result = _DecomposeResult(steps=raw_steps, source="llm", planner_cost=0.0)
        with patch.object(planner, "_decompose_task", return_value=decompose_result):
            plan = planner.plan("Do many things", budget=0.01)

        for step in plan.steps:
            model = sample_registry.get_model(step.model_id)
            assert model is not None
            assert model.input_price < 5.0

    def test_plan_step_dependencies_sequential_fallback(self, sample_registry: ModelRegistry):
        """Steps without depends_on should have sequential dependencies."""
        planner = Planner(registry=sample_registry)

        raw_steps = [
            {
                "description": "First",
                "capability": "general",
                "estimated_input_tokens": 500,
                "estimated_output_tokens": 500,
            },
            {
                "description": "Second",
                "capability": "general",
                "estimated_input_tokens": 500,
                "estimated_output_tokens": 500,
            },
            {
                "description": "Third",
                "capability": "general",
                "estimated_input_tokens": 500,
                "estimated_output_tokens": 500,
            },
        ]
        decompose_result = _DecomposeResult(steps=raw_steps, source="llm", planner_cost=0.0)
        with patch.object(planner, "_decompose_task", return_value=decompose_result):
            plan = planner.plan("Multi-step task", budget=5.0)

        assert plan.steps[0].depends_on == []
        assert plan.steps[1].depends_on == [1]
        assert plan.steps[2].depends_on == [2]

    def test_plan_step_dependencies_from_llm(self, sample_registry: ModelRegistry):
        """Steps with explicit depends_on from LLM should be parsed correctly."""
        planner = Planner(registry=sample_registry)

        raw_steps = [
            {
                "description": "Design API",
                "capability": "code",
                "estimated_input_tokens": 500,
                "estimated_output_tokens": 500,
                "depends_on": [],
            },
            {
                "description": "Design DB schema",
                "capability": "code",
                "estimated_input_tokens": 500,
                "estimated_output_tokens": 500,
                "depends_on": [],
            },
            {
                "description": "Implement endpoints",
                "capability": "code",
                "estimated_input_tokens": 500,
                "estimated_output_tokens": 500,
                "depends_on": [0, 1],
            },
        ]
        decompose_result = _DecomposeResult(steps=raw_steps, source="llm", planner_cost=0.0)
        with patch.object(planner, "_decompose_task", return_value=decompose_result):
            plan = planner.plan("Build API", budget=5.0)

        # Step 1 and 2 are independent
        assert plan.steps[0].depends_on == []
        assert plan.steps[1].depends_on == []
        # Step 3 depends on steps 1 and 2 (0-indexed [0,1] → 1-indexed [1,2])
        assert plan.steps[2].depends_on == [1, 2]

    def test_planner_cost_deducted_from_budget(self, sample_registry: ModelRegistry):
        """Planner cost should be tracked and reduce effective budget."""
        planner = Planner(registry=sample_registry)

        raw_steps = [
            {
                "description": "Do work",
                "capability": "general",
                "estimated_input_tokens": 500,
                "estimated_output_tokens": 500,
            },
        ]
        decompose_result = _DecomposeResult(
            steps=raw_steps,
            source="llm",
            planner_cost=0.005,
            planner_input_tokens=200,
            planner_output_tokens=100,
        )
        with patch.object(planner, "_decompose_task", return_value=decompose_result):
            plan = planner.plan("Do work", budget=1.0)

        assert plan.planner_cost == 0.005

    def test_optimize_for_budget(self, sample_registry: ModelRegistry):
        """_optimize_for_budget should downgrade expensive models."""
        planner = Planner(registry=sample_registry)

        # Create an over-budget plan manually
        from tokenwise.models import Plan, Step

        steps = [
            Step(
                id=1,
                description="Expensive step",
                model_id="anthropic/claude-opus-4",
                estimated_input_tokens=10000,
                estimated_output_tokens=10000,
                estimated_cost=0.90,
            ),
        ]
        plan = Plan(task="Test", steps=steps, total_estimated_cost=0.90, budget=0.01)

        optimized = planner._optimize_for_budget(plan)
        assert optimized.total_estimated_cost < 0.90

    def test_parse_steps_json(self, sample_registry: ModelRegistry):
        planner = Planner(registry=sample_registry)

        json_str = json.dumps(
            [
                {
                    "description": "Step 1",
                    "capability": "code",
                    "estimated_input_tokens": 500,
                    "estimated_output_tokens": 500,
                }
            ]
        )
        result = planner._parse_steps_json(json_str)
        assert len(result) == 1
        assert result[0]["description"] == "Step 1"

    def test_parse_steps_json_with_markdown_fences(self, sample_registry: ModelRegistry):
        planner = Planner(registry=sample_registry)

        step_json = json.dumps(
            [
                {
                    "description": "Step 1",
                    "capability": "code",
                    "estimated_input_tokens": 500,
                    "estimated_output_tokens": 500,
                }
            ]
        )
        content = f"```json\n{step_json}\n```"
        result = planner._parse_steps_json(content)
        assert len(result) == 1

    def test_fallback_decomposition(self, sample_registry: ModelRegistry):
        planner = Planner(registry=sample_registry)
        result = planner._fallback_decomposition("Build a website")
        assert len(result) == 1
        assert result[0]["description"] == "Build a website"

    def test_parse_steps_json_no_newline_after_fence(self, sample_registry: ModelRegistry):
        """Fence without newline after opening should still parse."""
        planner = Planner(registry=sample_registry)
        step_json = json.dumps(
            [
                {
                    "description": "Step 1",
                    "capability": "code",
                    "estimated_input_tokens": 500,
                    "estimated_output_tokens": 500,
                }
            ]
        )
        content = f"```json{step_json}```"
        result = planner._parse_steps_json(content)
        assert len(result) == 1
        assert result[0]["description"] == "Step 1"

    def test_parse_steps_json_bracket_extraction(self, sample_registry: ModelRegistry):
        """When fences don't match, extract the [...] block."""
        planner = Planner(registry=sample_registry)
        step_json = json.dumps(
            [
                {
                    "description": "Step 1",
                    "capability": "code",
                    "estimated_input_tokens": 500,
                    "estimated_output_tokens": 500,
                }
            ]
        )
        content = f"Here are the steps: {step_json} Hope that helps!"
        result = planner._parse_steps_json(content)
        assert len(result) == 1
        assert result[0]["description"] == "Step 1"

    def test_optimize_for_budget_uses_step_capabilities(self, sample_registry: ModelRegistry):
        """_optimize_for_budget should use step.required_capabilities, not model's."""
        planner = Planner(registry=sample_registry)
        from tokenwise.models import Plan, Step

        steps = [
            Step(
                id=1,
                description="Code step",
                model_id="anthropic/claude-opus-4",
                estimated_input_tokens=10000,
                estimated_output_tokens=10000,
                estimated_cost=0.90,
                required_capabilities=["code"],
            ),
        ]
        plan = Plan(task="Test", steps=steps, total_estimated_cost=0.90, budget=0.01)

        optimized = planner._optimize_for_budget(plan)
        # The downgraded model should still have the code capability
        downgraded = sample_registry.get_model(optimized.steps[0].model_id)
        assert downgraded is not None
        assert "code" in downgraded.capabilities
