"""Executor — runs a Plan by calling LLMs and tracking spend."""

from __future__ import annotations

import logging

from tokenwise.models import ModelInfo, Plan, PlanResult, Step, StepResult
from tokenwise.providers import ProviderResolver
from tokenwise.registry import ModelRegistry

logger = logging.getLogger(__name__)

# HTTP status codes that indicate the model itself is unusable (not a transient error)
_MODEL_UNUSABLE_CODES = {400, 402, 403, 404, 422}


class BudgetExhaustedError(Exception):
    """Raised when the budget is exhausted during execution."""


class Executor:
    """Executes a Plan step by step, tracking token usage and cost."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()
        self._resolver = ProviderResolver()
        self._failed_models: set[str] = set()

    def execute(self, plan: Plan) -> PlanResult:
        """Execute all steps in a plan sequentially.

        Tracks actual cost and stops if budget is exceeded.
        On step failure, attempts escalation to a stronger model.

        Returns:
            PlanResult with all step outputs and cost tracking.
        """
        result = PlanResult(task=plan.task, budget=plan.budget)
        prior_outputs: dict[int, str] = {}
        self._failed_models.clear()

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

            # If the planned model already failed in a prior step, skip straight to fallback
            if step.model_id in self._failed_models:
                step_result = StepResult(
                    step_id=step.id, model_id=step.model_id, success=False,
                    error="Skipped (model failed earlier)",
                )
            else:
                step_result = self._execute_step(
                    step_id=step.id,
                    model_id=step.model_id,
                    prompt=prompt,
                    budget_remaining=remaining,
                )
                if not step_result.success and self._is_model_error(step_result.error):
                    self._failed_models.add(step.model_id)

            # If step failed and we have budget, try fallback models
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

        # Fail if any step failed or if steps were skipped due to budget
        all_steps_ran = len(result.step_results) == len(plan.steps)
        all_succeeded = all(sr.success for sr in result.step_results)
        result.success = all_steps_ran and all_succeeded
        return result

    def _build_prompt(self, description: str, template: str, prior_outputs: dict[int, str]) -> str:
        """Build the prompt for a step, including prior context."""
        if template:
            prompt = template
        else:
            prompt = description

        if prior_outputs:
            context_parts = [
                f"[Step {sid} output]: {out[:500]}" for sid, out in prior_outputs.items()
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
        try:
            provider, provider_model = self._resolver.resolve(model_id)
            data = provider.chat_completion(
                model=provider_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            model = self.registry.get_model(model_id)
            actual_cost = (
                model.estimate_cost(input_tokens, output_tokens) if model else 0.0
            )

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

    def _is_model_error(self, error: str | None) -> bool:
        """Check if an error indicates the model is unusable (not a transient issue)."""
        if not error:
            return False
        return any(f"'{code}" in error or f"{code}" in error for code in _MODEL_UNUSABLE_CODES)

    def _get_fallback_candidates(
        self, exclude: set[str], budget_remaining: float, step: Step
    ) -> list[ModelInfo]:
        """Get fallback model candidates, preferring budget tier first."""
        from tokenwise.models import ModelTier

        all_candidates: list[ModelInfo] = []
        # Budget tier first — cheapest and most widely accessible
        for tier in [ModelTier.BUDGET, ModelTier.MID, ModelTier.FLAGSHIP]:
            all_candidates.extend(self.registry.find_models(tier=tier))

        # Filter: not excluded, not known-failed, affordable, has a nonzero price
        results = []
        for m in all_candidates:
            if m.id in exclude:
                continue
            if m.input_price <= 0:
                continue
            est = m.estimate_cost(step.estimated_input_tokens, step.estimated_output_tokens)
            if est <= budget_remaining:
                results.append(m)

        return results

    def _escalate(self, step: Step, prompt: str, budget_remaining: float) -> StepResult | None:
        """Try re-running a failed step with alternative models.

        Tries multiple candidates across tiers (budget first, then mid, then
        flagship) until one succeeds or all affordable options are exhausted.
        Models that failed in prior steps are automatically skipped.
        """
        logger.info("Escalating step %d — trying alternative models", step.id)

        tried = {step.model_id} | self._failed_models
        candidates = self._get_fallback_candidates(tried, budget_remaining, step)

        # Try up to 5 alternative models
        for model in candidates[:5]:
            tried.add(model.id)
            result = self._execute_step(step.id, model.id, prompt, budget_remaining)
            result.escalated = True
            if result.success:
                return result
            # If this model is also unusable, remember it and continue
            if self._is_model_error(result.error):
                self._failed_models.add(model.id)
                logger.info("Model %s unavailable, trying next", model.id)
                continue
            # Non-model error (timeout, etc.) — stop trying
            break

        return None
