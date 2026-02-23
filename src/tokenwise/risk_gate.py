"""Risk gate — lightweight, rule-based check for dangerous or ambiguous queries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tokenwise.config import RiskGateConfig

_IRREVERSIBLE_PATTERNS = [
    re.compile(r"\bdelete\b.*\b(?:database|db|table|production|prod)\b", re.I),
    re.compile(r"\b(?:database|db|table|production|prod)\b.*\bdelete\b", re.I),
    re.compile(r"\brm\s+-rf\b", re.I),
    re.compile(r"\bdrop\b.*\btable\b", re.I),
    re.compile(r"\btable\b.*\bdrop\b", re.I),
    re.compile(r"\btruncate\b.*\btable\b", re.I),
    re.compile(r"\bformat\b.*\bdisk\b", re.I),
    re.compile(r"\bdestroy\b.*\b(?:infrastructure|cluster|server)\b", re.I),
]

_AMBIGUITY_PATTERNS = [
    re.compile(r"^(?:do it|go ahead|whatever|just do it|ok|yes|sure|yep)$", re.I),
    re.compile(r"^(?:make it work|fix it|handle it|do the thing)$", re.I),
]


def _compute_ambiguity_score(query: str) -> float:
    """Compute an ambiguity score from 0.0 (clear) to 1.0 (ambiguous).

    Based on word count and pattern matches.
    """
    words = query.strip().split()
    word_count = len(words)

    # Very short queries are inherently ambiguous
    if word_count <= 2:
        base = 0.8
    elif word_count <= 5:
        base = 0.5
    elif word_count <= 10:
        base = 0.3
    else:
        base = 0.1

    # Boost if matches ambiguity patterns
    pattern_boost = 0.0
    for pattern in _AMBIGUITY_PATTERNS:
        if pattern.search(query.strip()):
            pattern_boost = 0.3
            break

    return min(1.0, base + pattern_boost)


@dataclass
class RiskGateResult:
    """Result of a risk gate evaluation."""

    blocked: bool
    reason: str
    ambiguity_score: float


def evaluate_risk(query: str, config: RiskGateConfig) -> RiskGateResult:
    """Evaluate a query against the risk gate rules.

    Returns a RiskGateResult indicating whether the query is blocked.
    """
    if not config.enabled:
        return RiskGateResult(blocked=False, reason="", ambiguity_score=0.0)

    # Check irreversible patterns
    if config.irreversible_block:
        for pattern in _IRREVERSIBLE_PATTERNS:
            if pattern.search(query):
                return RiskGateResult(
                    blocked=True,
                    reason=f"Query matches irreversible operation pattern: {pattern.pattern}",
                    ambiguity_score=0.0,
                )

    # Check ambiguity
    ambiguity_score = _compute_ambiguity_score(query)
    if ambiguity_score >= config.ambiguity_threshold:
        return RiskGateResult(
            blocked=True,
            reason=f"Query is too ambiguous (score={ambiguity_score:.2f}, "
            f"threshold={config.ambiguity_threshold:.2f})",
            ambiguity_score=ambiguity_score,
        )

    return RiskGateResult(blocked=False, reason="", ambiguity_score=ambiguity_score)
