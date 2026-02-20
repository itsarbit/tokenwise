"""Tests for CLI commands."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from tokenwise.cli import app
from tokenwise.models import ModelInfo, ModelTier

runner = CliRunner()


class TestCLI:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "TokenWise" in result.output

    def test_models_command_help(self):
        result = runner.invoke(app, ["models", "--help"])
        assert result.exit_code == 0
        assert "models" in result.output.lower()

    def test_route_command_help(self):
        result = runner.invoke(app, ["route", "--help"])
        assert result.exit_code == 0

    def test_plan_command_help(self):
        result = runner.invoke(app, ["plan", "--help"])
        assert result.exit_code == 0

    def test_serve_command_help(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0

    def test_route_with_mock_registry(self):
        mock_model = ModelInfo(
            id="test/cheap-model",
            name="Cheap Model",
            provider="test",
            input_price=0.10,
            output_price=0.20,
            context_window=8192,
            capabilities=["code"],
            tier=ModelTier.BUDGET,
        )

        with patch("tokenwise.cli.Router") as mock_router_cls:
            instance = mock_router_cls.return_value
            instance.route.return_value = mock_model

            result = runner.invoke(app, ["route", "Write a haiku"])
            assert result.exit_code == 0
            assert "test/cheap-model" in result.output

    def test_plan_with_mock_planner(self):
        from tokenwise.models import Plan, Step

        mock_plan = Plan(
            task="Test task",
            steps=[
                Step(id=1, description="Do the thing", model_id="test/model",
                     estimated_cost=0.001),
            ],
            total_estimated_cost=0.001,
            budget=1.0,
        )

        with patch("tokenwise.cli.Planner") as mock_planner_cls:
            instance = mock_planner_cls.return_value
            instance.plan.return_value = mock_plan

            result = runner.invoke(app, ["plan", "Test task"])
            assert result.exit_code == 0
            assert "Test task" in result.output
            assert "Do the thing" in result.output
