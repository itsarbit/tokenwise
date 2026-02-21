"""Executor — runs a Plan by calling LLMs and tracking spend."""

from __future__ import annotations

import logging

from tokenwise.models import CostLedger, ModelInfo, ModelTier, Plan, PlanResult, Step, StepResult
from tokenwise.providers import ProviderResolver
from tokenwise.registry import ModelRegistry

logger = logging.getLogger(__name__)

# HTTP status codes that indicate the model itself is unusable (not a transient error)
_MODEL_UNUSABLE_CODES = {402, 403, 404}

_TIER_STRENGTH: dict[ModelTier, int] = {
    ModelTier.BUDGET: 0,
    ModelTier.MID: 1,
    ModelTier.FLAGSHIP: 2,
}


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

        budget_exhausted = False
        for step in plan.steps:
            # Check budget before executing
            if result.total_cost >= plan.budget:
                if not budget_exhausted:
                    logger.warning(
                        "Budget exhausted after $%.4f, skipping remaining steps",
                        result.total_cost,
                    )
                    budget_exhausted = True
                result.skipped_steps.append(step)
                result.success = False
                continue

            remaining = plan.budget - result.total_cost

            # Build prompt with context from prior steps
            prompt = self._build_prompt(step.description, step.prompt_template, prior_outputs)

            # If the planned model already failed in a prior step, skip straight to fallback
            if step.model_id in self._failed_models:
                step_result = StepResult(
                    step_id=step.id,
                    model_id=step.model_id,
                    success=False,
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

            # Record the initial attempt in the ledger
            result.ledger.record(
                reason=f"step {step.id} attempt 1",
                model_id=step_result.model_id,
                input_tokens=step_result.input_tokens,
                output_tokens=step_result.output_tokens,
                cost=step_result.actual_cost,
                success=step_result.success,
            )

            # If step failed and we have budget, try fallback models
            has_budget = result.total_cost + step_result.actual_cost < plan.budget
            if not step_result.success and has_budget:
                escalated = self._escalate(
                    step, prompt, remaining - step_result.actual_cost, result.ledger
                )
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
            actual_cost = model.estimate_cost(input_tokens, output_tokens) if model else 0.0

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
        """Get fallback model candidates, escalating to stronger tiers.

        Looks up the failed model's tier and only considers equal or stronger
        tiers. Within tiers, stronger tiers come first; within the same tier,
        models are sorted by price descending (more expensive = likely higher
        quality). Also filters by the failed model's capabilities.
        """
        # Determine the failed model's tier and capabilities for filtering
        failed_model = self.registry.get_model(step.model_id)
        failed_tier = failed_model.tier if failed_model else ModelTier.BUDGET
        failed_strength = _TIER_STRENGTH.get(failed_tier, 0)
        # Use first capability of failed model for filtering (if available)
        required_cap = None
        if failed_model and failed_model.capabilities:
            # Pick the first non-"general" capability, or "general" if that's all there is
            for cap in failed_model.capabilities:
                if cap != "general":
                    required_cap = cap
                    break

        # Collect candidates from tiers >= failed tier, stronger tiers first
        stronger: list[ModelInfo] = []
        same_tier: list[ModelInfo] = []

        for tier in [ModelTier.FLAGSHIP, ModelTier.MID, ModelTier.BUDGET]:
            tier_strength = _TIER_STRENGTH[tier]
            if tier_strength < failed_strength:
                continue
            models = self.registry.find_models(tier=tier)
            for m in models:
                if m.id in exclude or m.input_price <= 0:
                    continue
                # Capability filtering
                if required_cap and required_cap not in m.capabilities:
                    continue
                est = m.estimate_cost(step.estimated_input_tokens, step.estimated_output_tokens)
                if est > budget_remaining:
                    continue
                if tier_strength > failed_strength:
                    stronger.append(m)
                else:
                    same_tier.append(m)

        # Sort each group by price descending (more expensive = likely better)
        stronger.sort(key=lambda m: m.input_price, reverse=True)
        same_tier.sort(key=lambda m: m.input_price, reverse=True)

        # Stronger tiers first, then same tier
        seen: set[str] = set()
        results: list[ModelInfo] = []
        for m in stronger + same_tier:
            if m.id not in seen:
                seen.add(m.id)
                results.append(m)

        return results

    def _escalate(
        self,
        step: Step,
        prompt: str,
        budget_remaining: float,
        ledger: CostLedger | None = None,
    ) -> StepResult | None:
        """Try re-running a failed step with alternative models.

        Escalates to stronger tiers first, then same tier.
        Models that failed in prior steps are automatically skipped.
        """
        if ledger is None:
            ledger = CostLedger()

        logger.info("Escalating step %d — trying alternative models", step.id)

        tried = {step.model_id} | self._failed_models
        candidates = self._get_fallback_candidates(tried, budget_remaining, step)

        # Try up to 5 alternative models
        for attempt_num, model in enumerate(candidates[:5], start=1):
            tried.add(model.id)
            result = self._execute_step(step.id, model.id, prompt, budget_remaining)
            result.escalated = True

            ledger.record(
                reason=f"step {step.id} escalation attempt {attempt_num}",
                model_id=model.id,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost=result.actual_cost,
                success=result.success,
            )

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
