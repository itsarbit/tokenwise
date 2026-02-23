"""Router — picks the best model for a single query based on strategy.

Uses a two-stage pipeline:
  1. **Scenario detection** — detect required capabilities and estimate complexity
  2. **Strategy routing** — filter to capable models within budget, apply preference
"""

from __future__ import annotations

import re

from tokenwise.config import get_settings
from tokenwise.models import (
    EscalationPolicy,
    ModelInfo,
    ModelTier,
    RiskGateBlockedError,
    RoutingStrategy,
    RoutingTrace,
    TerminationState,
    TraceLevel,
)
from tokenwise.registry import ModelRegistry
from tokenwise.risk_gate import evaluate_risk

# Word-boundary keyword patterns for detecting query needs
_CODE_PATTERNS = [
    re.compile(r"\b(?:code|function|class|debug|refactor|implement|api|sql)\b", re.I),
    re.compile(r"\b(?:python|javascript|typescript|rust|golang|java|cpp)\b", re.I),
    re.compile(r"\b(?:bug|fix|compile|syntax|runtime|exception|traceback)\b", re.I),
]
_REASONING_PATTERNS = [
    re.compile(r"\b(?:reason|analyze|compare|evaluate|logic|proof|derive)\b", re.I),
    re.compile(r"step[- ]by[- ]step", re.I),
    re.compile(r"explain\s+why", re.I),
]
_MATH_PATTERNS = [
    re.compile(r"\b(?:calculate|math|equation|integral|derivative|solve|formula)\b", re.I),
]

# Token estimates per complexity level: (input_tokens, output_tokens)
_TOKEN_ESTIMATES: dict[str, tuple[int, int]] = {
    "simple": (500, 200),
    "moderate": (1000, 500),
    "complex": (2000, 1000),
}


def _detect_capabilities(query: str) -> list[str]:
    """Detect likely required capabilities from query text."""
    caps = []
    if any(p.search(query) for p in _CODE_PATTERNS):
        caps.append("code")
    if any(p.search(query) for p in _REASONING_PATTERNS):
        caps.append("reasoning")
    if any(p.search(query) for p in _MATH_PATTERNS):
        caps.append("math")
    return caps


def _estimate_complexity(query: str) -> str:
    """Estimate query complexity: 'simple', 'moderate', or 'complex'."""
    word_count = len(query.split())
    if word_count < 15:
        return "simple"
    if word_count < 60:
        return "moderate"
    return "complex"


class Router:
    """Selects the best model for a single query.

    Every route goes through a two-stage pipeline:

    **Stage 1 — Scenario detection:**
      Analyze the query to detect required capabilities (code, reasoning, math)
      and estimate complexity (simple, moderate, complex).

    **Stage 2 — Strategy routing:**
      Filter models by detected capability and budget ceiling, then apply
      the strategy preference (cheapest, best_quality, or balanced).
    """

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    @staticmethod
    def _detect_scenario(
        query: str, required_capability: str | None = None
    ) -> tuple[str | None, str]:
        """Detect primary capability and complexity from a query.

        Returns:
            (primary_capability, complexity)
        """
        detected_caps = _detect_capabilities(query)
        primary_cap = required_capability or (detected_caps[0] if detected_caps else None)
        complexity = _estimate_complexity(query)
        return primary_cap, complexity

    def route(
        self,
        query: str,
        strategy: RoutingStrategy | str = RoutingStrategy.BALANCED,
        budget: float | None = None,
        required_capability: str | None = None,
        budget_strict: bool = True,
    ) -> ModelInfo:
        """Pick the best model for a query.

        Args:
            query: The user query text.
            strategy: Routing strategy (cheapest, best_quality, balanced).
            budget: Optional max cost in USD — applied as a ceiling on all strategies.
            required_capability: Explicitly required capability (overrides detection).
            budget_strict: When True (default), raise ValueError if no model fits the
                budget. When False, fall through to the cheapest available model.

        Returns:
            The selected ModelInfo.

        Raises:
            ValueError: If no suitable model is found, or if budget is too tight
                and budget_strict is True.
        """
        if isinstance(strategy, str):
            strategy = RoutingStrategy(strategy)

        # ── Stage 1: Scenario detection ──────────────────────────────
        primary_cap, complexity = self._detect_scenario(query, required_capability)

        # ── Stage 2: Filter → Route ─────────────────────────────────
        # Filter by capability
        candidates = self.registry.find_models(capability=primary_cap)
        candidates = [m for m in candidates if m.input_price > 0]

        # Apply budget ceiling (if provided)
        if budget is not None:
            affordable = self._filter_by_budget(candidates, budget, complexity)
            if affordable:
                candidates = affordable
            elif budget_strict and candidates:
                # Nothing affordable and strict mode — raise with helpful info
                est_in, est_out = _TOKEN_ESTIMATES.get(complexity, (1000, 500))
                cheapest = min(candidates, key=lambda m: m.estimate_cost(est_in, est_out))
                cheapest_cost = cheapest.estimate_cost(est_in, est_out)
                raise ValueError(
                    f"Budget ${budget:.6f} is too tight. Cheapest option is "
                    f"{cheapest.id} at ~${cheapest_cost:.6f} estimated cost."
                )
            # If not strict, keep full set (best-effort ceiling)

        if not candidates:
            raise ValueError(f"No models found for capability={primary_cap}")

        # Apply strategy preference
        if strategy == RoutingStrategy.CHEAPEST:
            return self._route_cheapest(candidates)
        elif strategy == RoutingStrategy.BEST_QUALITY:
            return self._route_best_quality(candidates)
        else:  # balanced
            return self._route_balanced(candidates, complexity)

    def route_with_trace(
        self,
        query: str,
        strategy: RoutingStrategy | str = RoutingStrategy.BALANCED,
        budget: float | None = None,
        required_capability: str | None = None,
        budget_strict: bool = True,
    ) -> tuple[ModelInfo, RoutingTrace]:
        """Route a query and return both the model and a structured trace.

        Same logic as ``route()`` but additionally returns a ``RoutingTrace``
        with audit metadata. Runs the risk gate check when enabled.

        Raises:
            RiskGateBlockedError: If the risk gate blocks the query.
            ValueError: If no suitable model is found.
        """
        settings = get_settings()
        try:
            policy = EscalationPolicy(settings.escalation_policy)
        except ValueError:
            policy = EscalationPolicy.FLEXIBLE
        try:
            level = TraceLevel(settings.trace_level)
        except ValueError:
            level = TraceLevel.BASIC
        trace = RoutingTrace(escalation_policy=policy, trace_level=level)

        # Risk gate check
        risk_result = evaluate_risk(query, settings.risk_gate)
        if risk_result.blocked:
            trace.termination_state = TerminationState.NO_GO
            raise RiskGateBlockedError(reason=risk_result.reason, trace=trace)

        model = self.route(
            query=query,
            strategy=strategy,
            budget=budget,
            required_capability=required_capability,
            budget_strict=budget_strict,
        )

        trace.initial_model = model.id
        trace.initial_tier = model.tier
        trace.final_model = model.id
        trace.final_tier = model.tier
        trace.termination_state = TerminationState.COMPLETED
        if budget is not None:
            _, complexity = self._detect_scenario(query, required_capability)
            est_in, est_out = _TOKEN_ESTIMATES.get(complexity, (1000, 500))
            est_cost = model.estimate_cost(est_in, est_out)
            trace.budget_used = est_cost
            trace.budget_remaining = budget - est_cost

        return model, trace

    def _filter_by_budget(
        self, candidates: list[ModelInfo], budget: float, complexity: str
    ) -> list[ModelInfo]:
        """Filter candidates to those whose estimated cost fits within budget."""
        est_in, est_out = _TOKEN_ESTIMATES.get(complexity, (1000, 500))
        return [m for m in candidates if m.estimate_cost(est_in, est_out) <= budget]

    def _route_cheapest(self, candidates: list[ModelInfo]) -> ModelInfo:
        """Pick the cheapest model from the candidate set."""
        return min(candidates, key=lambda m: m.input_price)

    def _route_best_quality(self, candidates: list[ModelInfo]) -> ModelInfo:
        """Pick the highest-quality model: flagship → mid → budget.

        Within a tier, pick the most expensive (likely highest quality).
        """
        for tier in [ModelTier.FLAGSHIP, ModelTier.MID, ModelTier.BUDGET]:
            tier_models = [m for m in candidates if m.tier == tier]
            if tier_models:
                return max(tier_models, key=lambda m: m.input_price)
        return max(candidates, key=lambda m: m.input_price)

    def _route_balanced(self, candidates: list[ModelInfo], complexity: str) -> ModelInfo:
        """Match model tier to query complexity, cheapest within the target tier."""
        tier_map = {
            "simple": ModelTier.BUDGET,
            "moderate": ModelTier.MID,
            "complex": ModelTier.FLAGSHIP,
        }
        target_tier = tier_map.get(complexity, ModelTier.MID)

        tier_models = [m for m in candidates if m.tier == target_tier]
        if tier_models:
            return min(tier_models, key=lambda m: m.input_price)

        # Fallback: cheapest across all remaining candidates
        return min(candidates, key=lambda m: m.input_price)
