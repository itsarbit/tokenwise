"""Executor — runs a Plan by calling LLMs and tracking spend."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from tokenwise.config import get_settings
from tokenwise.models import (
    CostLedger,
    EscalationPolicy,
    EscalationReasonCode,
    EscalationRecord,
    ModelInfo,
    ModelTier,
    Plan,
    PlanResult,
    RoutingTrace,
    Step,
    StepResult,
    TerminationState,
    TraceLevel,
)
from tokenwise.providers import ProviderResolver
from tokenwise.registry import ModelRegistry

logger = logging.getLogger(__name__)

# HTTP status codes that indicate the model itself is unusable (not a transient error)
_MODEL_UNUSABLE_CODES = {402, 403, 404}

# Safety margin multiplier on input token estimates to account for tokenizer variance
_INPUT_TOKEN_SAFETY_MARGIN = 1.2

_TIER_STRENGTH: dict[ModelTier, int] = {
    ModelTier.BUDGET: 0,
    ModelTier.MID: 1,
    ModelTier.FLAGSHIP: 2,
}


class BudgetExhaustedError(Exception):
    """Raised when the budget is exhausted during execution."""


class Executor:
    """Executes a Plan step by step, tracking token usage and cost."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        min_output_tokens: int | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self._resolver = ProviderResolver()
        self._failed_models: set[str] = set()
        self.min_output_tokens = (
            min_output_tokens if min_output_tokens is not None else get_settings().min_output_tokens
        )

    def execute(self, plan: Plan) -> PlanResult:
        """Execute all steps in a plan, using async DAG scheduling when possible.

        Steps with satisfied dependencies run concurrently via ``aexecute()``.

        **Note:** If called inside an existing async event loop (e.g. from a
        Jupyter notebook, FastAPI endpoint, or nested ``asyncio.run``), this
        method automatically falls back to sequential execution to avoid nested
        loop errors. In that case, use ``await aexecute(plan)`` directly for
        concurrent step execution.

        Tracks actual cost and stops if budget is exceeded.
        On step failure, attempts escalation to a stronger model.

        Returns:
            PlanResult with all step outputs and cost tracking.
        """
        try:
            asyncio.get_running_loop()
            logger.info(
                "Existing event loop detected — falling back to sequential execution. "
                "Use await aexecute(plan) for concurrent step execution.",
            )
            return self._execute_sequential(plan)
        except RuntimeError:
            pass
        return asyncio.run(self.aexecute(plan))

    def _execute_sequential(self, plan: Plan) -> PlanResult:
        """Execute all steps sequentially (used when already in async context)."""
        result = PlanResult(task=plan.task, budget=plan.budget)
        trace = self._make_trace()
        prior_outputs: dict[int, str] = {}
        self._failed_models.clear()

        budget_exhausted = False
        for step in plan.steps:
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

            # Pre-call estimate check: skip if estimated cost exceeds remaining budget
            if step.estimated_cost > remaining:
                logger.warning(
                    "Step %d estimated cost $%.4f exceeds remaining $%.4f, skipping",
                    step.id,
                    step.estimated_cost,
                    remaining,
                )
                result.skipped_steps.append(step)
                result.success = False
                continue

            prompt = self._build_prompt(step.description, step.prompt_template, prior_outputs)

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
                    estimated_input_tokens=step.estimated_input_tokens,
                )
                if not step_result.success and self._is_model_error(step_result):
                    self._failed_models.add(step.model_id)

            # Record every attempt — success or failure — before considering escalation
            result.ledger.record(
                reason=f"step {step.id} attempt 1",
                model_id=step_result.model_id,
                input_tokens=step_result.input_tokens,
                output_tokens=step_result.output_tokens,
                cost=step_result.actual_cost,
                success=step_result.success,
            )

            has_budget = result.total_cost + step_result.actual_cost < plan.budget
            if not step_result.success and has_budget:
                escalated = self._escalate(
                    step, prompt, remaining - step_result.actual_cost, result.ledger, trace,
                    self._classify_escalation_reason(step_result),
                )
                if escalated is not None:
                    result.total_cost += step_result.actual_cost
                    step_result = escalated

            result.step_results.append(step_result)
            result.total_cost += step_result.actual_cost
            prior_outputs[step.id] = step_result.output

        for sr in reversed(result.step_results):
            if sr.success and sr.output:
                result.final_output = sr.output
                break

        all_steps_ran = len(result.step_results) == len(plan.steps)
        all_succeeded = all(sr.success for sr in result.step_results)
        result.success = all_steps_ran and all_succeeded

        # Populate trace
        if result.step_results:
            trace.initial_model = plan.steps[0].model_id
            first_model = self.registry.get_model(plan.steps[0].model_id)
            trace.initial_tier = first_model.tier if first_model else ModelTier.BUDGET
            last_sr = result.step_results[-1]
            trace.final_model = last_sr.model_id
            final_model = self.registry.get_model(last_sr.model_id)
            trace.final_tier = final_model.tier if final_model else ModelTier.BUDGET
        trace.budget_used = result.total_cost
        trace.budget_remaining = result.budget_remaining
        if result.success:
            trace.termination_state = TerminationState.COMPLETED
        elif result.total_cost >= plan.budget:
            trace.termination_state = TerminationState.EXHAUSTED
        else:
            trace.termination_state = TerminationState.FAILED
        result.routing_trace = trace

        return result

    async def aexecute(self, plan: Plan) -> PlanResult:
        """Execute plan steps concurrently according to their dependency graph.

        Steps whose dependencies are all satisfied are launched in parallel.
        Uses reservation-based budgeting: each step reserves its estimated_cost
        before launch, preventing parallel steps from collectively overshooting.
        """
        result = PlanResult(task=plan.task, budget=plan.budget)
        trace = self._make_trace()
        self._failed_models.clear()

        # Build lookup structures
        step_map: dict[int, Step] = {s.id: s for s in plan.steps}
        completed: dict[int, StepResult] = {}
        outputs: dict[int, str] = {}
        all_ids = {s.id for s in plan.steps}
        # Track total spend including reserved amounts for in-flight steps
        reserved_budget: float = 0.0

        while True:
            if result.total_cost >= plan.budget:
                # Mark all remaining steps as skipped
                for sid in sorted(all_ids - set(completed)):
                    result.skipped_steps.append(step_map[sid])
                result.success = False
                break

            # Find ready steps: all deps satisfied, not yet completed
            ready: list[Step] = []
            for step in plan.steps:
                if step.id in completed:
                    continue
                deps_met = all(d in completed for d in step.depends_on)
                if deps_met:
                    ready.append(step)

            if not ready:
                # Deadlock detection: if steps remain but none are ready
                remaining_ids = all_ids - set(completed)
                if remaining_ids:
                    stuck = sorted(remaining_ids)
                    logger.error("Deadlock: steps %s have unsatisfied dependencies", stuck)
                    for sid in stuck:
                        result.skipped_steps.append(step_map[sid])
                    result.success = False
                break

            # Reservation-based budget: only launch steps that fit
            available_budget = plan.budget - result.total_cost - reserved_budget
            launchable: list[Step] = []
            for step in ready:
                reservation = step.estimated_cost
                if reservation <= available_budget:
                    launchable.append(step)
                    reserved_budget += reservation
                    available_budget -= reservation
                else:
                    # Can't afford this step right now; it stays pending
                    # and will be retried next iteration (or skipped at budget check)
                    pass

            if not launchable:
                # All ready steps are too expensive — skip remaining
                for sid in sorted(all_ids - set(completed)):
                    result.skipped_steps.append(step_map[sid])
                result.success = False
                break

            # Execute launchable steps concurrently, each with its reserved budget
            tasks = [
                self._arun_step(step, outputs, step.estimated_cost, result.ledger, trace)
                for step in launchable
            ]
            step_results = await asyncio.gather(*tasks)

            for step, step_result in zip(launchable, step_results):
                # Release reservation and account actual cost
                reserved_budget -= step.estimated_cost
                completed[step.id] = step_result
                result.step_results.append(step_result)
                result.total_cost += step_result.actual_cost
                outputs[step.id] = step_result.output

        # Final output is the last successful step's output (by step id order)
        for sr in sorted(result.step_results, key=lambda s: s.step_id, reverse=True):
            if sr.success and sr.output:
                result.final_output = sr.output
                break

        all_steps_ran = len(result.step_results) == len(plan.steps)
        all_succeeded = all(sr.success for sr in result.step_results)
        result.success = all_steps_ran and all_succeeded

        # Populate trace
        if result.step_results:
            trace.initial_model = plan.steps[0].model_id
            first_model = self.registry.get_model(plan.steps[0].model_id)
            trace.initial_tier = first_model.tier if first_model else ModelTier.BUDGET
            last_sr = sorted(result.step_results, key=lambda s: s.step_id)[-1]
            trace.final_model = last_sr.model_id
            final_model = self.registry.get_model(last_sr.model_id)
            trace.final_tier = final_model.tier if final_model else ModelTier.BUDGET
        trace.budget_used = result.total_cost
        trace.budget_remaining = result.budget_remaining
        if result.success:
            trace.termination_state = TerminationState.COMPLETED
        elif result.total_cost >= plan.budget:
            trace.termination_state = TerminationState.EXHAUSTED
        else:
            trace.termination_state = TerminationState.FAILED
        result.routing_trace = trace

        return result

    async def _arun_step(
        self,
        step: Step,
        prior_outputs: dict[int, str],
        budget_remaining: float,
        ledger: CostLedger,
        trace: RoutingTrace | None = None,
    ) -> StepResult:
        """Execute a single step asynchronously, with escalation on failure.

        Returns a StepResult whose actual_cost includes any wasted spend
        from failed attempts (so the caller can just sum actual_cost).
        """
        prompt = self._build_prompt(step.description, step.prompt_template, prior_outputs)

        if step.model_id in self._failed_models:
            step_result = StepResult(
                step_id=step.id,
                model_id=step.model_id,
                success=False,
                error="Skipped (model failed earlier)",
            )
        else:
            step_result = await self._aexecute_step(
                step_id=step.id,
                model_id=step.model_id,
                prompt=prompt,
                budget_remaining=budget_remaining,
                estimated_input_tokens=step.estimated_input_tokens,
            )
            if not step_result.success and self._is_model_error(step_result):
                self._failed_models.add(step.model_id)

        # Record every attempt — success or failure — before considering escalation
        ledger.record(
            reason=f"step {step.id} attempt 1",
            model_id=step_result.model_id,
            input_tokens=step_result.input_tokens,
            output_tokens=step_result.output_tokens,
            cost=step_result.actual_cost,
            success=step_result.success,
        )

        has_budget = step_result.actual_cost < budget_remaining
        if not step_result.success and has_budget:
            failed_cost = step_result.actual_cost
            escalated = await self._aescalate(
                step, prompt, budget_remaining - failed_cost, ledger, trace,
                self._classify_escalation_reason(step_result),
            )
            if escalated is not None:
                # Include the wasted cost from the failed attempt
                escalated.actual_cost += failed_cost
                step_result = escalated

        return step_result

    async def _aexecute_step(
        self,
        step_id: int,
        model_id: str,
        prompt: str,
        budget_remaining: float,
        estimated_input_tokens: int = 0,
    ) -> StepResult:
        """Execute a single step via async provider call."""
        prompt_based_estimate = len(prompt) // 4
        raw_estimate = max(estimated_input_tokens, prompt_based_estimate)
        input_tokens = int(raw_estimate * _INPUT_TOKEN_SAFETY_MARGIN)
        max_tokens = self._compute_max_tokens(model_id, budget_remaining, input_tokens)
        if max_tokens is not None and max_tokens < self.min_output_tokens:
            return StepResult(
                step_id=step_id,
                model_id=model_id,
                success=False,
                error=f"Budget too low: {max_tokens} output tokens "
                f"< minimum {self.min_output_tokens}",
            )

        try:
            provider, provider_model = self._resolver.resolve(model_id)
            kwargs: dict[str, Any] = {
                "model": provider_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            data = await provider.achat_completion(**kwargs)

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

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.error("Step %d failed with model %s: HTTP %d", step_id, model_id, status)
            return StepResult(
                step_id=step_id,
                model_id=model_id,
                success=False,
                error=str(e),
                http_status_code=status,
            )
        except Exception as e:
            logger.error("Step %d failed with model %s: %s", step_id, model_id, e)
            return StepResult(
                step_id=step_id,
                model_id=model_id,
                success=False,
                error=str(e),
            )

    async def _aescalate(
        self,
        step: Step,
        prompt: str,
        budget_remaining: float,
        ledger: CostLedger | None = None,
        trace: RoutingTrace | None = None,
        reason_code: EscalationReasonCode = EscalationReasonCode.MODEL_ERROR,
    ) -> StepResult | None:
        """Async version of _escalate: try alternative models on failure."""
        if ledger is None:
            ledger = CostLedger()

        logger.info("Escalating step %d — trying alternative models", step.id)

        failed_model = self.registry.get_model(step.model_id)
        failed_tier = failed_model.tier if failed_model else ModelTier.BUDGET

        tried = {step.model_id} | self._failed_models
        candidates = self._get_fallback_candidates(tried, budget_remaining, step)

        for attempt_num, model in enumerate(candidates[:5], start=1):
            tried.add(model.id)

            if trace is not None:
                trace.escalations.append(EscalationRecord(
                    from_model=step.model_id,
                    from_tier=failed_tier,
                    to_model=model.id,
                    to_tier=model.tier,
                    reason_code=reason_code,
                    step_id=step.id,
                ))

            result = await self._aexecute_step(
                step.id,
                model.id,
                prompt,
                budget_remaining,
                estimated_input_tokens=step.estimated_input_tokens,
            )
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
            if self._is_model_error(result):
                self._failed_models.add(model.id)
                logger.info("Model %s unavailable, trying next", model.id)
                continue
            break

        return None

    def _compute_max_tokens(
        self,
        model_id: str,
        budget_remaining: float,
        estimated_input_tokens: int = 0,
    ) -> int | None:
        """Compute max output tokens that fit within the remaining budget.

        Reserves budget for estimated input cost before computing the output cap,
        ensuring total cost (input + output) stays within budget.

        Returns None if the model is unknown or has no output pricing.
        """
        model = self.registry.get_model(model_id)
        if not model or model.output_price <= 0:
            return None
        input_cost = model.input_price * estimated_input_tokens / 1_000_000
        budget_for_output = budget_remaining - input_cost
        if budget_for_output <= 0:
            return 0
        return int(budget_for_output / (model.output_price / 1_000_000))

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
        estimated_input_tokens: int = 0,
    ) -> StepResult:
        """Execute a single step by calling the LLM."""
        prompt_based_estimate = len(prompt) // 4
        raw_estimate = max(estimated_input_tokens, prompt_based_estimate)
        input_tokens = int(raw_estimate * _INPUT_TOKEN_SAFETY_MARGIN)
        max_tokens = self._compute_max_tokens(model_id, budget_remaining, input_tokens)
        if max_tokens is not None and max_tokens < self.min_output_tokens:
            return StepResult(
                step_id=step_id,
                model_id=model_id,
                success=False,
                error=f"Budget too low: {max_tokens} output tokens "
                f"< minimum {self.min_output_tokens}",
            )

        try:
            provider, provider_model = self._resolver.resolve(model_id)
            kwargs: dict[str, Any] = {
                "model": provider_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            data = provider.chat_completion(**kwargs)

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

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.error("Step %d failed with model %s: HTTP %d", step_id, model_id, status)
            return StepResult(
                step_id=step_id,
                model_id=model_id,
                success=False,
                error=str(e),
                http_status_code=status,
            )
        except Exception as e:
            logger.error("Step %d failed with model %s: %s", step_id, model_id, e)
            return StepResult(
                step_id=step_id,
                model_id=model_id,
                success=False,
                error=str(e),
            )

    def _is_model_error(self, step_result: StepResult) -> bool:
        """Check if a step failure indicates the model is unusable (not transient)."""
        if step_result.http_status_code is not None:
            return step_result.http_status_code in _MODEL_UNUSABLE_CODES
        return False

    @staticmethod
    def _classify_escalation_reason(step_result: StepResult) -> EscalationReasonCode:
        """Map a failed step result to an escalation reason code."""
        if step_result.http_status_code in (402, 403, 404):
            return EscalationReasonCode.MODEL_ERROR
        error = step_result.error or ""
        if "budget" in error.lower():
            return EscalationReasonCode.BUDGET_EXHAUSTED
        if "capability" in error.lower():
            return EscalationReasonCode.CAPABILITY_MISMATCH
        return EscalationReasonCode.MODEL_ERROR

    @staticmethod
    def _make_trace() -> RoutingTrace:
        """Create a RoutingTrace populated from current settings."""
        settings = get_settings()
        try:
            policy = EscalationPolicy(settings.escalation_policy)
        except ValueError:
            policy = EscalationPolicy.FLEXIBLE
        try:
            level = TraceLevel(settings.trace_level)
        except ValueError:
            level = TraceLevel.BASIC
        return RoutingTrace(escalation_policy=policy, trace_level=level)

    def _get_fallback_candidates(
        self, exclude: set[str], budget_remaining: float, step: Step
    ) -> list[ModelInfo]:
        """Get fallback model candidates, escalating to stronger tiers.

        Looks up the failed model's tier and only considers equal or stronger
        tiers. Within tiers, stronger tiers come first; within the same tier,
        models are sorted by price descending (more expensive = likely higher
        quality). Filters by the step's required_capabilities when available,
        falling back to inferring from the failed model's capabilities.
        """
        # Determine the failed model's tier
        failed_model = self.registry.get_model(step.model_id)
        failed_tier = failed_model.tier if failed_model else ModelTier.BUDGET
        failed_strength = _TIER_STRENGTH.get(failed_tier, 0)

        # Determine required capabilities: prefer step's explicit list,
        # fall back to inferring from the failed model
        required_caps: set[str] = set()
        if step.required_capabilities:
            required_caps = {c for c in step.required_capabilities if c != "general"}
        elif failed_model and failed_model.capabilities:
            required_caps = {c for c in failed_model.capabilities if c != "general"}

        # Collect candidates from tiers >= failed tier, stronger tiers first
        stronger, same_tier = self._collect_candidates(
            exclude, failed_strength, budget_remaining, step, required_caps
        )

        # If strict capability filtering eliminated everything, relax and retry
        if not (stronger or same_tier) and required_caps:
            stronger, same_tier = self._collect_candidates(
                exclude, failed_strength, budget_remaining, step, required_caps=set()
            )

        # Monotonic mode: drop same-tier candidates (only escalate upward)
        try:
            policy = EscalationPolicy(get_settings().escalation_policy)
        except ValueError:
            policy = EscalationPolicy.FLEXIBLE
        if policy == EscalationPolicy.MONOTONIC:
            same_tier = []

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

    def _collect_candidates(
        self,
        exclude: set[str],
        failed_strength: int,
        budget_remaining: float,
        step: Step,
        required_caps: set[str],
    ) -> tuple[list[ModelInfo], list[ModelInfo]]:
        """Collect fallback models split into (stronger, same_tier) lists."""
        stronger: list[ModelInfo] = []
        same_tier: list[ModelInfo] = []
        for tier in [ModelTier.FLAGSHIP, ModelTier.MID, ModelTier.BUDGET]:
            tier_strength = _TIER_STRENGTH[tier]
            if tier_strength < failed_strength:
                continue
            for m in self.registry.find_models(tier=tier):
                if m.id in exclude or m.input_price <= 0:
                    continue
                if required_caps and not required_caps.issubset(set(m.capabilities)):
                    continue
                est = m.estimate_cost(step.estimated_input_tokens, step.estimated_output_tokens)
                if est > budget_remaining:
                    continue
                if tier_strength > failed_strength:
                    stronger.append(m)
                else:
                    same_tier.append(m)
        return stronger, same_tier

    def _escalate(
        self,
        step: Step,
        prompt: str,
        budget_remaining: float,
        ledger: CostLedger | None = None,
        trace: RoutingTrace | None = None,
        reason_code: EscalationReasonCode = EscalationReasonCode.MODEL_ERROR,
    ) -> StepResult | None:
        """Try re-running a failed step with alternative models.

        Escalates to stronger tiers first, then same tier.
        Models that failed in prior steps are automatically skipped.
        """
        if ledger is None:
            ledger = CostLedger()

        logger.info("Escalating step %d — trying alternative models", step.id)

        failed_model = self.registry.get_model(step.model_id)
        failed_tier = failed_model.tier if failed_model else ModelTier.BUDGET

        tried = {step.model_id} | self._failed_models
        candidates = self._get_fallback_candidates(tried, budget_remaining, step)

        # Try up to 5 alternative models
        for attempt_num, model in enumerate(candidates[:5], start=1):
            tried.add(model.id)

            if trace is not None:
                trace.escalations.append(EscalationRecord(
                    from_model=step.model_id,
                    from_tier=failed_tier,
                    to_model=model.id,
                    to_tier=model.tier,
                    reason_code=reason_code,
                    step_id=step.id,
                ))

            result = self._execute_step(
                step.id,
                model.id,
                prompt,
                budget_remaining,
                estimated_input_tokens=step.estimated_input_tokens,
            )
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
            if self._is_model_error(result):
                self._failed_models.add(model.id)
                logger.info("Model %s unavailable, trying next", model.id)
                continue
            # Non-model error (timeout, etc.) — stop trying
            break

        return None
