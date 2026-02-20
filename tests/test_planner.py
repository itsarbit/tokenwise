"""Tests for Planner."""

from __future__ import annotations

import json
from unittest.mock import patch

from tokenwise.planner import Planner
from tokenwise.registry import ModelRegistry


class TestPlanner:
    def test_plan_with_fallback(self, sample_registry: ModelRegistry):
        """When LLM decomposition fails, fallback should produce a single-step plan."""
        planner = Planner(registry=sample_registry)

        # Patch _decompose_task to simulate LLM failure (returns fallback)
        fallback = planner._fallback_decomposition("Test task")
        with patch.object(planner, "_decompose_task", return_value=fallback):
            plan = planner.plan("Test task", budget=1.0)

        assert len(plan.steps) == 1
        assert plan.steps[0].description == "Test task"
        assert plan.budget == 1.0
        assert plan.is_within_budget()

    def test_plan_assigns_models(self, sample_registry: ModelRegistry):
        """Each step should get a valid model from the registry."""
        planner = Planner(registry=sample_registry)

        raw_steps = [
            {"description": "Write code", "capability": "code",
             "estimated_input_tokens": 500, "estimated_output_tokens": 500},
            {"description": "Review logic", "capability": "reasoning",
             "estimated_input_tokens": 800, "estimated_output_tokens": 400},
        ]
        with patch.object(planner, "_decompose_task", return_value=raw_steps):
            plan = planner.plan("Build something", budget=5.0)

        assert len(plan.steps) == 2
        for step in plan.steps:
            model = sample_registry.get_model(step.model_id)
            assert model is not None

    def test_plan_respects_budget(self, sample_registry: ModelRegistry):
        """Plan should try to fit within budget."""
        planner = Planner(registry=sample_registry)

        raw_steps = [
            {"description": f"Step {i}", "capability": "general",
             "estimated_input_tokens": 500, "estimated_output_tokens": 500}
            for i in range(5)
        ]
        with patch.object(planner, "_decompose_task", return_value=raw_steps):
            plan = planner.plan("Do many things", budget=0.01)

        # With a tiny budget, the plan should use cheap models
        for step in plan.steps:
            model = sample_registry.get_model(step.model_id)
            assert model is not None
            assert model.input_price < 5.0

    def test_plan_step_dependencies(self, sample_registry: ModelRegistry):
        """Steps should have sequential dependencies."""
        planner = Planner(registry=sample_registry)

        raw_steps = [
            {"description": "First", "capability": "general",
             "estimated_input_tokens": 500, "estimated_output_tokens": 500},
            {"description": "Second", "capability": "general",
             "estimated_input_tokens": 500, "estimated_output_tokens": 500},
            {"description": "Third", "capability": "general",
             "estimated_input_tokens": 500, "estimated_output_tokens": 500},
        ]
        with patch.object(planner, "_decompose_task", return_value=raw_steps):
            plan = planner.plan("Multi-step task", budget=5.0)

        assert plan.steps[0].depends_on == []
        assert plan.steps[1].depends_on == [1]
        assert plan.steps[2].depends_on == [2]

    def test_optimize_for_budget(self, sample_registry: ModelRegistry):
        """_optimize_for_budget should downgrade expensive models."""
        planner = Planner(registry=sample_registry)

        # Create an over-budget plan manually
        from tokenwise.models import Plan, Step
        steps = [
            Step(id=1, description="Expensive step", model_id="anthropic/claude-opus-4",
                 estimated_input_tokens=10000, estimated_output_tokens=10000,
                 estimated_cost=0.90),
        ]
        plan = Plan(task="Test", steps=steps, total_estimated_cost=0.90, budget=0.01)

        optimized = planner._optimize_for_budget(plan)
        assert optimized.total_estimated_cost < 0.90

    def test_parse_steps_json(self, sample_registry: ModelRegistry):
        planner = Planner(registry=sample_registry)

        json_str = json.dumps([
            {"description": "Step 1", "capability": "code",
             "estimated_input_tokens": 500, "estimated_output_tokens": 500}
        ])
        result = planner._parse_steps_json(json_str)
        assert len(result) == 1
        assert result[0]["description"] == "Step 1"

    def test_parse_steps_json_with_markdown_fences(self, sample_registry: ModelRegistry):
        planner = Planner(registry=sample_registry)

        step_json = json.dumps([{
            "description": "Step 1", "capability": "code",
            "estimated_input_tokens": 500, "estimated_output_tokens": 500,
        }])
        content = f"```json\n{step_json}\n```"
        result = planner._parse_steps_json(content)
        assert len(result) == 1

    def test_fallback_decomposition(self, sample_registry: ModelRegistry):
        planner = Planner(registry=sample_registry)
        result = planner._fallback_decomposition("Build a website")
        assert len(result) == 1
        assert result[0]["description"] == "Build a website"
