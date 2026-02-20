"""Executor — runs a Plan by calling LLMs and tracking spend."""

from __future__ import annotations

import logging

import httpx

from tokenwise.config import get_settings
from tokenwise.models import Plan, PlanResult, Step, StepResult
from tokenwise.registry import ModelRegistry

logger = logging.getLogger(__name__)


class BudgetExhaustedError(Exception):
    """Raised when the budget is exhausted during execution."""


class Executor:
    """Executes a Plan step by step, tracking token usage and cost."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def execute(self, plan: Plan) -> PlanResult:
        """Execute all steps in a plan sequentially.

        Tracks actual cost and stops if budget is exceeded.
        On step failure, attempts escalation to a stronger model.

        Returns:
            PlanResult with all step outputs and cost tracking.
        """
        result = PlanResult(task=plan.task, budget=plan.budget)
        prior_outputs: dict[int, str] = {}

        for step in plan.steps:
            # Check budget before executing
            if result.total_cost >= plan.budget:
                logger.warning(
                    "Budget exhausted after $%.4f, skipping remaining steps",
                    result.total_cost,
                )
                result.success = False
                break

            remaining = plan.budget - result.total_cost

            # Build prompt with context from prior steps
            prompt = self._build_prompt(step.description, step.prompt_template, prior_outputs)

            step_result = self._execute_step(
                step_id=step.id,
                model_id=step.model_id,
                prompt=prompt,
                budget_remaining=remaining,
            )

            # If step failed and we have budget, try escalation
            has_budget = result.total_cost + step_result.actual_cost < plan.budget
            if not step_result.success and has_budget:
                escalated = self._escalate(step, prompt, remaining - step_result.actual_cost)
                if escalated is not None:
                    result.total_cost += step_result.actual_cost  # still pay for the failed attempt
                    step_result = escalated

            result.step_results.append(step_result)
            result.total_cost += step_result.actual_cost
            prior_outputs[step.id] = step_result.output

        # Final output is the last successful step's output
        for sr in reversed(result.step_results):
            if sr.success and sr.output:
                result.final_output = sr.output
                break

        result.success = all(sr.success for sr in result.step_results)
        return result

    def _build_prompt(
        self, description: str, template: str, prior_outputs: dict[int, str]
    ) -> str:
        """Build the prompt for a step, including prior context."""
        if template:
            prompt = template
        else:
            prompt = description

        if prior_outputs:
            context_parts = [
                f"[Step {sid} output]: {out[:500]}"
                for sid, out in prior_outputs.items()
            ]
            context = "\n".join(context_parts)
            prompt = f"Context from prior steps:\n{context}\n\nCurrent task: {prompt}"

        return prompt

    def _execute_step(
        self,
        step_id: int,
        model_id: str,
        prompt: str,
        budget_remaining: float,
    ) -> StepResult:
        """Execute a single step by calling the LLM."""
        settings = get_settings()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if settings.openrouter_api_key:
            headers["Authorization"] = f"Bearer {settings.openrouter_api_key}"

        try:
            resp = httpx.post(
                f"{settings.openrouter_base_url}/chat/completions",
                headers=headers,
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            # Calculate actual cost
            model = self.registry.get_model(model_id)
            if model:
                actual_cost = model.estimate_cost(input_tokens, output_tokens)
            else:
                actual_cost = 0.0

            return StepResult(
                step_id=step_id,
                model_id=model_id,
                output=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                actual_cost=actual_cost,
                success=True,
            )

        except Exception as e:
            logger.error("Step %d failed with model %s: %s", step_id, model_id, e)
            return StepResult(
                step_id=step_id,
                model_id=model_id,
                success=False,
                error=str(e),
            )

    def _escalate(self, step: Step, prompt: str, budget_remaining: float) -> StepResult | None:
        """Try re-running a failed step with a stronger model."""
        from tokenwise.models import ModelTier

        logger.info("Escalating step %d to a stronger model", step.id)

        # Find a flagship model that fits the remaining budget
        candidates = self.registry.find_models(tier=ModelTier.FLAGSHIP)
        if not candidates:
            candidates = self.registry.find_models(tier=ModelTier.MID)
        if not candidates:
            return None

        # Pick cheapest flagship
        model = min(candidates, key=lambda m: m.input_price)
        est_cost = model.estimate_cost(step.estimated_input_tokens, step.estimated_output_tokens)
        if est_cost > budget_remaining:
            return None

        result = self._execute_step(step.id, model.id, prompt, budget_remaining)
        result.escalated = True
        return result
