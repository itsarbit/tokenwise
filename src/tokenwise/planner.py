"""Planner — decomposes tasks into steps and assigns models."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from tokenwise.config import MissingAPIKeyError, get_settings
from tokenwise.models import ModelInfo, Plan, Step
from tokenwise.providers import ProviderResolver
from tokenwise.registry import ModelRegistry
from tokenwise.router import Router

logger = logging.getLogger(__name__)

_DECOMPOSITION_PROMPT = """\
You are a task planner. Given a task, decompose it into a list of concrete subtasks.
Each subtask should be a single, focused step that can be handled by one LLM call.

Return ONLY a JSON array of objects with these fields:
- "description": what this step does
- "capability": the type of skill needed ("code", "reasoning", "creative", "math", or "general")
- "estimated_input_tokens": approximate input tokens (integer)
- "estimated_output_tokens": approximate output tokens (integer)

Example:
[
  {{"description": "Design the REST API endpoints",
   "capability": "code",
   "estimated_input_tokens": 500,
   "estimated_output_tokens": 800}},
  {{"description": "Write the database schema",
   "capability": "code",
   "estimated_input_tokens": 600,
   "estimated_output_tokens": 600}}
]

Task: {task}
"""


class Planner:
    """Decomposes complex tasks into steps with model assignments."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        router: Router | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.router = router or Router(self.registry)
        self._resolver = ProviderResolver()

    def plan(
        self,
        task: str,
        budget: float = 1.0,
        preferences: dict[str, Any] | None = None,
    ) -> Plan:
        """Create an execution plan for a task.

        Uses a cheap planner LLM to decompose the task, then assigns
        the best model to each step within the budget.

        Args:
            task: Natural language task description.
            budget: Maximum total budget in USD.
            preferences: Optional dict with keys like 'strategy', 'max_steps'.

        Returns:
            A Plan object with steps and model assignments.
        """
        preferences = preferences or {}

        # Step 1: Decompose the task using the planner model
        raw_steps = self._decompose_task(task)

        # Step 2: Assign models to each step within budget
        steps = self._assign_models(raw_steps, budget)

        # Step 3: Build the plan
        total_cost = sum(s.estimated_cost for s in steps)
        plan = Plan(
            task=task,
            steps=steps,
            total_estimated_cost=total_cost,
            budget=budget,
        )

        # If over budget, try to trim by using cheaper models
        if not plan.is_within_budget():
            plan = self._optimize_for_budget(plan)

        return plan

    def _decompose_task(self, task: str) -> list[dict[str, Any]]:
        """Call the planner LLM to decompose a task into subtasks."""
        settings = get_settings()
        prompt = _DECOMPOSITION_PROMPT.format(task=task)

        try:
            provider, provider_model = self._resolver.resolve(
                settings.planner_model,
            )
            data = provider.chat_completion(
                model=provider_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
                timeout=60.0,
            )
            content = data["choices"][0]["message"]["content"]
            return self._parse_steps_json(content)
        except MissingAPIKeyError:
            raise
        except Exception as e:
            logger.warning("LLM decomposition failed (%s), using fallback", e)
            return self._fallback_decomposition(task)

    def _parse_steps_json(self, content: str) -> list[dict[str, Any]]:
        """Parse the LLM's JSON response into step dicts."""
        # Strip markdown code fences if present
        content = content.strip()
        fence_match = re.search(r"```(?:\w+)?\n(.*?)```", content, re.DOTALL)
        if fence_match:
            content = fence_match.group(1).strip()

        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        logger.warning("Could not parse LLM response as JSON, using fallback")
        return []

    def _fallback_decomposition(self, task: str) -> list[dict[str, Any]]:
        """Simple rule-based fallback when LLM decomposition fails."""
        return [
            {
                "description": task,
                "capability": "general",
                "estimated_input_tokens": 1000,
                "estimated_output_tokens": 1000,
            }
        ]

    def _assign_models(self, raw_steps: list[dict[str, Any]], budget: float) -> list[Step]:
        """Assign a model to each step based on capability and budget."""
        if not raw_steps:
            raw_steps = [
                {
                    "description": "Complete the task",
                    "capability": "general",
                    "estimated_input_tokens": 1000,
                    "estimated_output_tokens": 1000,
                }
            ]

        steps: list[Step] = []
        remaining_budget = budget
        num_steps = len(raw_steps)

        for i, raw in enumerate(raw_steps):
            capability = raw.get("capability", "general")
            est_in = raw.get("estimated_input_tokens", 500)
            est_out = raw.get("estimated_output_tokens", 500)

            # Budget per step: proportional share of remaining budget
            step_budget = remaining_budget / (num_steps - i) if (num_steps - i) > 0 else 0

            try:
                model = self.router.route(
                    query=raw.get("description", ""),
                    strategy="cheapest",
                    budget=step_budget,
                    required_capability=capability if capability != "general" else None,
                )
            except ValueError:
                # Use whatever is cheapest
                model = self._get_any_model()

            cost = model.estimate_cost(est_in, est_out)
            remaining_budget -= cost

            steps.append(
                Step(
                    id=i + 1,
                    description=raw.get("description", f"Step {i + 1}"),
                    model_id=model.id,
                    estimated_input_tokens=est_in,
                    estimated_output_tokens=est_out,
                    estimated_cost=cost,
                    depends_on=[i] if i > 0 else [],
                )
            )

        return steps

    def _get_any_model(self) -> ModelInfo:
        """Get any available model as a last resort."""
        models = self.registry.list_all()
        paid = [m for m in models if m.input_price > 0]
        if paid:
            return paid[0]
        if models:
            return models[0]
        raise ValueError("No models available in registry")

    def _optimize_for_budget(self, plan: Plan) -> Plan:
        """Try to bring an over-budget plan within budget by downgrading models."""
        overage = plan.total_estimated_cost - plan.budget

        # Sort steps by cost descending — downgrade the most expensive first
        sorted_steps = sorted(
            enumerate(plan.steps),
            key=lambda x: x[1].estimated_cost,
            reverse=True,
        )

        for idx, step in sorted_steps:
            if overage <= 0:
                break

            # Detect capabilities the current model has
            current = self.registry.get_model(step.model_id)
            cap = None
            if current and current.capabilities:
                cap = current.capabilities[0]

            # Find cheapest model that still has the required capability
            cheapest = self.registry.cheapest(capability=cap)
            if not cheapest:
                cheapest = self.registry.cheapest()
            if cheapest and cheapest.id != step.model_id:
                old_cost = step.estimated_cost
                new_cost = cheapest.estimate_cost(
                    step.estimated_input_tokens,
                    step.estimated_output_tokens,
                )
                if new_cost < old_cost:
                    plan.steps[idx].model_id = cheapest.id
                    plan.steps[idx].estimated_cost = new_cost
                    overage -= old_cost - new_cost

        plan.total_estimated_cost = sum(s.estimated_cost for s in plan.steps)
        return plan
