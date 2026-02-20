"""Router — picks the best model for a single query based on strategy."""

from __future__ import annotations

from tokenwise.models import ModelInfo, ModelTier, RoutingStrategy
from tokenwise.registry import ModelRegistry

# Simple heuristics for detecting query needs
_CODE_KEYWORDS = {"code", "function", "class", "debug", "refactor", "implement", "api", "sql",
                  "python", "javascript", "typescript", "rust", "golang", "java", "bug", "fix"}
_REASONING_KEYWORDS = {"reason", "think", "analyze", "compare", "evaluate", "step by step",
                       "explain why", "proof", "derive", "logic"}
_MATH_KEYWORDS = {"calculate", "math", "equation", "integral", "derivative", "solve", "formula"}


def _detect_capabilities(query: str) -> list[str]:
    """Detect likely required capabilities from query text."""
    query_lower = query.lower()
    caps = []
    if any(kw in query_lower for kw in _CODE_KEYWORDS):
        caps.append("code")
    if any(kw in query_lower for kw in _REASONING_KEYWORDS):
        caps.append("reasoning")
    if any(kw in query_lower for kw in _MATH_KEYWORDS):
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
    """Selects the best model for a single query."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def route(
        self,
        query: str,
        strategy: RoutingStrategy | str = RoutingStrategy.BALANCED,
        budget: float | None = None,
        required_capability: str | None = None,
    ) -> ModelInfo:
        """Pick the best model for a query.

        Args:
            query: The user query text.
            strategy: Routing strategy to use.
            budget: Maximum cost in USD for this query (approximate).
            required_capability: Explicitly required capability.

        Returns:
            The selected ModelInfo.

        Raises:
            ValueError: If no suitable model is found.
        """
        if isinstance(strategy, str):
            strategy = RoutingStrategy(strategy)

        # Detect capabilities from query
        detected_caps = _detect_capabilities(query)
        primary_cap = required_capability or (detected_caps[0] if detected_caps else None)

        complexity = _estimate_complexity(query)

        if strategy == RoutingStrategy.CHEAPEST:
            return self._route_cheapest(primary_cap)
        elif strategy == RoutingStrategy.BEST_QUALITY:
            return self._route_best_quality(primary_cap)
        elif strategy == RoutingStrategy.BUDGET_CONSTRAINED:
            if budget is None:
                raise ValueError("budget is required for budget_constrained strategy")
            return self._route_budget_constrained(primary_cap, budget, complexity)
        else:  # balanced
            return self._route_balanced(primary_cap, complexity)

    def _route_cheapest(self, capability: str | None) -> ModelInfo:
        model = self.registry.cheapest(capability)
        if model is None:
            raise ValueError(f"No models found for capability={capability}")
        return model

    def _route_best_quality(self, capability: str | None) -> ModelInfo:
        models = self.registry.find_models(capability=capability, tier=ModelTier.FLAGSHIP)
        if not models:
            models = self.registry.find_models(capability=capability)
        if not models:
            raise ValueError(f"No models found for capability={capability}")
        # Pick the most expensive flagship (likely highest quality)
        return max(models, key=lambda m: m.input_price)

    def _route_budget_constrained(
        self, capability: str | None, budget: float, complexity: str
    ) -> ModelInfo:
        # Estimate tokens: simple ~500 in / 200 out, moderate ~1000/500, complex ~2000/1000
        token_estimates = {
            "simple": (500, 200),
            "moderate": (1000, 500),
            "complex": (2000, 1000),
        }
        est_in, est_out = token_estimates.get(complexity, (1000, 500))

        # Find models where estimated cost fits budget
        candidates = self.registry.find_models(capability=capability)
        affordable = [
            m for m in candidates
            if m.estimate_cost(est_in, est_out) <= budget and m.input_price > 0
        ]
        if not affordable:
            # Fall back to cheapest available
            return self._route_cheapest(capability)

        # Among affordable models, pick the best tier available
        for tier in [ModelTier.FLAGSHIP, ModelTier.MID, ModelTier.BUDGET]:
            tier_models = [m for m in affordable if m.tier == tier]
            if tier_models:
                # Pick cheapest within the best affordable tier
                return min(tier_models, key=lambda m: m.input_price)

        return affordable[0]

    def _route_balanced(self, capability: str | None, complexity: str) -> ModelInfo:
        # Simple → budget tier, moderate → mid tier, complex → flagship
        tier_map = {
            "simple": ModelTier.BUDGET,
            "moderate": ModelTier.MID,
            "complex": ModelTier.FLAGSHIP,
        }
        target_tier = tier_map.get(complexity, ModelTier.MID)

        models = self.registry.find_models(capability=capability, tier=target_tier)
        if not models:
            # Fall back: try any tier
            models = self.registry.find_models(capability=capability)
        if not models:
            raise ValueError(f"No models found for capability={capability}")

        # Pick cheapest within the target tier
        paid = [m for m in models if m.input_price > 0]
        return paid[0] if paid else models[0]
