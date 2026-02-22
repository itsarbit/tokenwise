#!/usr/bin/env python3
"""Strategy-level Pareto benchmark — routing strategies compared on cost vs quality.

Runs 20 tasks (5 simple, 5 reasoning, 5 coding, 5 hard) across four strategies:
  1. Budget Only    — cheapest capable model, no escalation
  2. Mid Only       — mid-tier model, no escalation
  3. Flagship Only  — flagship model, no escalation
  4. TokenWise      — budget start with capability-aware escalation

Generates a scatter plot (assets/pareto.png) showing cost vs success rate.

Usage:
    uv sync --group benchmark
    uv run python benchmarks/strategy_pareto.py
    uv run python benchmarks/strategy_pareto.py --dry-run
    uv run python benchmarks/strategy_pareto.py --output assets/pareto.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ─── Tasks ──────────────────────────────────────────────────────────────

SIMPLE_TASKS = [
    "Write a haiku about distributed systems",
    "Explain the CAP theorem in three sentences",
    "What is the capital of Australia and why was it chosen?",
    "Summarize the difference between TCP and UDP in a bullet list",
    "Write a one-paragraph review of Python as a programming language",
]

REASONING_TASKS = [
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
    "How much does the ball cost? Show your reasoning step by step.",
    "You have 8 identical-looking balls and one is slightly heavier. Using a balance "
    "scale, what is the minimum number of weighings to guarantee finding the heavy one? "
    "Explain your strategy.",
    "If all roses are flowers, and some flowers fade quickly, can we conclude that some "
    "roses fade quickly? Explain using formal logic.",
    "Three people check into a hotel room costing $30. They each pay $10. The manager "
    "realizes the room costs $25 and sends $5 back with a bellboy who keeps $2, returning "
    "$1 to each guest. Each guest paid $9 (total $27) and the bellboy has $2, totaling $29. "
    "Where is the missing dollar?",
    "Alice is looking at Bob. Bob is looking at Charlie. Alice is married. Charlie is not "
    "married. Is a married person looking at an unmarried person? Answer and explain, even "
    "though Bob's marital status is unknown.",
]

CODING_TASKS = [
    "Write a Python function implementing binary search on a sorted list. Handle edge "
    "cases (empty list, element not found). Include type hints.",
    "Write a Python class implementing an LRU cache with O(1) get and put operations.",
    "Write a Python function that validates email addresses using regex. Include at "
    "least 5 test cases.",
    "Write a Python function to detect a cycle in a linked list using Floyd's tortoise "
    "and hare algorithm. Include the ListNode class.",
    "Write a Python generator function that yields Fibonacci numbers indefinitely. "
    "Show usage printing the first 20.",
]

HARD_TASKS = [
    "Design and implement a token bucket rate limiter in Python. Include the algorithm "
    "explanation and a working class with acquire() and try_acquire() methods.",
    "Compare merge sort and quicksort: time complexity (best/avg/worst), space complexity, "
    "stability. Then implement both in Python with type hints.",
    "Explain how HTTPS works end-to-end: DNS resolution, TCP handshake, TLS 1.3 handshake "
    "(including certificate verification), then HTTP request/response.",
    "Write a Python function that evaluates mathematical expression strings supporting "
    "+, -, *, /, parentheses, and correct operator precedence without using eval().",
    "Explain the Raft consensus algorithm: leader election, log replication, and the "
    "safety guarantee. Include pseudocode for the leader election.",
]

ALL_TASKS = SIMPLE_TASKS + REASONING_TASKS + CODING_TASKS + HARD_TASKS
TASK_CATEGORIES = ["simple"] * 5 + ["reasoning"] * 5 + ["coding"] * 5 + ["hard"] * 5

CATEGORY_CAPABILITIES: dict[str, list[str]] = {
    "simple": ["general"],
    "reasoning": ["reasoning"],
    "coding": ["code"],
    "hard": ["code", "reasoning"],
}

DEFAULT_BUDGET_MODEL = "openai/gpt-4.1-nano"
DEFAULT_MID_MODEL = "openai/gpt-4.1"
DEFAULT_FLAGSHIP_MODEL = "anthropic/claude-sonnet-4"

BUDGET_PER_TASK = 0.10


# ─── Answer validators ─────────────────────────────────────────────────
# Index into ALL_TASKS → validation function.
# Tasks without a validator use the default length check.

def _val_bat_ball(r: str) -> bool:
    """Correct answer: $0.05 (NOT $0.10)."""
    r = r.lower()
    return ("0.05" in r or "five cents" in r or "5 cents" in r) and "0.10" not in r.split("cost")[0][-20:] if "cost" in r else ("0.05" in r or "five cents" in r or "5 cents" in r)


def _val_8_balls(r: str) -> bool:
    """Correct answer: 2 weighings."""
    return "2" in r and "weigh" in r.lower()


def _val_roses(r: str) -> bool:
    """Correct answer: No — undistributed middle fallacy."""
    first_200 = r.lower()[:300]
    return any(w in first_200 for w in ["cannot conclude", "does not follow", "no,", "no.", "no ", "invalid", "fallacy"])


def _val_missing_dollar(r: str) -> bool:
    """Should identify the framing error (the $27 already includes the bellboy's $2)."""
    r_lower = r.lower()
    return any(w in r_lower for w in ["misleading", "error in", "wrong to add", "fallacy", "incorrect", "shouldn't add", "should not add", "flawed", "double.count", "double count", "already include"])


def _val_alice_bob(r: str) -> bool:
    """Correct answer: Yes, a married person IS looking at an unmarried person."""
    first_200 = r.lower()[:300]
    return "yes" in first_200


def _val_has_code(r: str) -> bool:
    """Check that response contains actual Python code."""
    return "def " in r or "class " in r


def _val_hard_substantial(r: str) -> bool:
    """Check for both explanation and code (>300 chars)."""
    return len(r.strip()) > 300 and ("def " in r or "class " in r or "```" in r or "algorithm" in r.lower())


# Map task index → validator (0-indexed into ALL_TASKS)
TASK_VALIDATORS: dict[int, Any] = {
    # Reasoning (indices 5-9)
    5: _val_bat_ball,
    6: _val_8_balls,
    7: _val_roses,
    8: _val_missing_dollar,
    9: _val_alice_bob,
    # Coding (indices 10-14)
    10: _val_has_code,
    11: _val_has_code,
    12: _val_has_code,
    13: _val_has_code,
    14: _val_has_code,
    # Hard (indices 15-19)
    15: _val_hard_substantial,
    16: _val_hard_substantial,
    17: _val_hard_substantial,
    18: _val_hard_substantial,
    19: _val_hard_substantial,
}


# ─── Data classes ───────────────────────────────────────────────────────


@dataclass
class TaskResult:
    task: str
    category: str
    success: bool
    cost: float
    model_used: str
    escalated: bool = False
    error: str | None = None


@dataclass
class StrategyResult:
    name: str
    color: str
    marker: str
    results: list[TaskResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return (
            sum(1 for r in self.results if r.success) / len(self.results)
            if self.results
            else 0.0
        )

    @property
    def avg_cost(self) -> float:
        return (
            sum(r.cost for r in self.results) / len(self.results)
            if self.results
            else 0.0
        )

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.results)


# ─── Runners ────────────────────────────────────────────────────────────


def _check_success(response: dict[str, Any], task_index: int = -1) -> bool:
    """Check if an LLM response is a meaningful success.

    Uses task-specific validators when available (reasoning correctness,
    code structure checks), otherwise falls back to a length check.
    """
    choices = response.get("choices", [])
    if not choices:
        return False
    content = choices[0].get("message", {}).get("content", "")
    if not content or len(content.strip()) < 20:
        return False
    validator = TASK_VALIDATORS.get(task_index)
    if validator:
        return bool(validator(content))
    return True


def _get_cost(model_id: str, response: dict[str, Any], registry: Any) -> float:
    """Calculate cost from response usage data."""
    usage = response.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    model_info = registry.get_model(model_id)
    if model_info:
        return float(model_info.estimate_cost(input_tokens, output_tokens))
    return 0.0


def run_fixed_tier(
    model_id: str,
    tasks: list[str],
    categories: list[str],
    resolver: Any,
    registry: Any,
) -> list[TaskResult]:
    """Run all tasks with a single model, no escalation."""
    results: list[TaskResult] = []
    provider, model_name = resolver.resolve(model_id)

    for i, (task, cat) in enumerate(zip(tasks, categories)):
        print(f"    Task {i + 1}/{len(tasks)} [{cat:>9}] ", end="", flush=True)
        try:
            resp = provider.chat_completion(
                model=model_name,
                messages=[{"role": "user", "content": task}],
                max_tokens=1024,
                timeout=90.0,
            )
            success = _check_success(resp, task_index=i)
            cost = _get_cost(model_id, resp, registry)
            print(f"{'ok' if success else 'FAIL'} (${cost:.6f})")
            results.append(
                TaskResult(
                    task=task,
                    category=cat,
                    success=success,
                    cost=cost,
                    model_used=model_id,
                )
            )
        except Exception as e:
            print(f"ERROR ({e})")
            results.append(
                TaskResult(
                    task=task,
                    category=cat,
                    success=False,
                    cost=0.0,
                    model_used=model_id,
                    error=str(e),
                )
            )

    return results


def run_escalation(
    tasks: list[str],
    categories: list[str],
    tier_models: list[str],
    resolver: Any,
    registry: Any,
) -> list[TaskResult]:
    """Run tasks with quality-aware escalation: budget → mid → flagship.

    For each task, try the cheapest model first. If the answer fails
    validation, escalate to the next tier. Cost includes all attempts
    (wasted + successful).
    """
    results: list[TaskResult] = []

    for i, (task, cat) in enumerate(zip(tasks, categories)):
        print(f"    Task {i + 1}/{len(tasks)} [{cat:>9}] ", end="", flush=True)

        total_cost = 0.0
        success = False
        final_model = tier_models[0]
        escalated = False

        for tier_idx, model_id in enumerate(tier_models):
            provider, model_name = resolver.resolve(model_id)
            try:
                resp = provider.chat_completion(
                    model=model_name,
                    messages=[{"role": "user", "content": task}],
                    max_tokens=1024,
                    timeout=90.0,
                )
                cost = _get_cost(model_id, resp, registry)
                total_cost += cost
                passed = _check_success(resp, task_index=i)

                if passed:
                    success = True
                    final_model = model_id
                    escalated = tier_idx > 0
                    break
                # Answer failed validation — escalate to next tier
                final_model = model_id
            except Exception:
                # LLM error — escalate to next tier
                pass

        esc_tag = " [escalated]" if escalated else ""
        print(
            f"{'ok' if success else 'FAIL'} "
            f"(${total_cost:.6f}) {final_model}{esc_tag}"
        )
        results.append(
            TaskResult(
                task=task,
                category=cat,
                success=success,
                cost=total_cost,
                model_used=final_model,
                escalated=escalated,
            )
        )

    return results

    return results


# ─── CSV ────────────────────────────────────────────────────────────────


def save_csv(strategies: dict[str, StrategyResult], csv_path: str) -> None:
    """Save per-task results to CSV."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["strategy", "task", "category", "success", "cost", "model_used", "escalated", "error"]
        )
        for name, sr in strategies.items():
            for tr in sr.results:
                writer.writerow(
                    [
                        name,
                        tr.task[:60],
                        tr.category,
                        tr.success,
                        f"{tr.cost:.8f}",
                        tr.model_used,
                        tr.escalated,
                        tr.error or "",
                    ]
                )

    print(f"Results saved to {path}")


# ─── Plot ───────────────────────────────────────────────────────────────


def plot_pareto(strategies: dict[str, StrategyResult], output_path: str) -> None:
    """Generate the cost–quality frontier scatter plot."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("\nmatplotlib not installed. Run: uv sync --group benchmark")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for name, sr in strategies.items():
        if not sr.results:
            continue

        x = sr.avg_cost
        y = sr.success_rate * 100
        is_esc = name == "TokenWise Escalation"

        ax.scatter(
            x,
            y,
            c=sr.color,
            marker=sr.marker,
            s=400 if is_esc else 150,
            zorder=10 if is_esc else 5,
            edgecolors="black" if is_esc else "white",
            linewidths=2 if is_esc else 1,
        )
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(14, 8) if is_esc else (10, 6),
            fontsize=11 if is_esc else 10,
            fontweight="bold" if is_esc else "normal",
            color=sr.color,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Average Cost per Task (USD, log scale)", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title(
        "Cost\u2013Quality Frontier: Routing Strategies",
        fontsize=14,
        fontweight="bold",
    )

    min_rate = min(sr.success_rate * 100 for sr in strategies.values() if sr.results)
    ax.set_ylim(max(0, min_rate - 15), 105)
    ax.grid(True, alpha=0.3, linestyle="--")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.4f}"))

    fig.text(
        0.5,
        0.02,
        "\u2190 Cheaper                                                     "
        "   More Expensive \u2192",
        ha="center",
        fontsize=9,
        color="gray",
        style="italic",
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    print(f"Plot saved to {path}")


# ─── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TokenWise strategy-level Pareto benchmark"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show plan without executing"
    )
    parser.add_argument(
        "--output", default="assets/pareto.png", help="Output plot path"
    )
    parser.add_argument(
        "--csv",
        default="benchmarks/strategy_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--budget-model", default=DEFAULT_BUDGET_MODEL, help="Budget tier model"
    )
    parser.add_argument(
        "--mid-model", default=DEFAULT_MID_MODEL, help="Mid tier model"
    )
    parser.add_argument(
        "--flagship-model",
        default=DEFAULT_FLAGSHIP_MODEL,
        help="Flagship tier model",
    )
    args = parser.parse_args()

    print("TokenWise Strategy Pareto Benchmark")
    print("=" * 50)
    print(
        f"\nTasks: {len(ALL_TASKS)} "
        f"({len(SIMPLE_TASKS)} simple, {len(REASONING_TASKS)} reasoning, "
        f"{len(CODING_TASKS)} coding, {len(HARD_TASKS)} hard)"
    )
    print(f"Budget per task: ${BUDGET_PER_TASK}")
    print("\nStrategies:")
    print(f"  1. Budget Only           \u2192 {args.budget_model}")
    print(f"  2. Mid Only              \u2192 {args.mid_model}")
    print(f"  3. Flagship Only         \u2192 {args.flagship_model}")
    print(
        f"  4. TokenWise Escalation  \u2192 {args.budget_model} \u2192 mid \u2192 flagship"
    )
    print(f"\nTotal LLM calls: ~{len(ALL_TASKS) * 4}")

    if args.dry_run:
        print("\n--dry-run: exiting without executing.")
        sys.exit(0)

    from tokenwise.providers import ProviderResolver
    from tokenwise.registry import ModelRegistry

    registry = ModelRegistry()
    registry.load_from_openrouter()
    resolver = ProviderResolver()

    strategies: dict[str, StrategyResult] = {}

    # 1. Budget Only
    print(f"\n{'─' * 50}")
    print(f"  Strategy: Budget Only ({args.budget_model})")
    sr_budget = StrategyResult(name="Budget Only", color="#2ecc71", marker="o")
    sr_budget.results = run_fixed_tier(
        args.budget_model, ALL_TASKS, TASK_CATEGORIES, resolver, registry
    )
    strategies["Budget Only"] = sr_budget
    print(
        f"  \u2192 {sr_budget.success_rate:.0%} success, "
        f"avg ${sr_budget.avg_cost:.6f}/task"
    )

    # 2. Mid Only
    print(f"\n{'─' * 50}")
    print(f"  Strategy: Mid Only ({args.mid_model})")
    sr_mid = StrategyResult(name="Mid Only", color="#f39c12", marker="o")
    sr_mid.results = run_fixed_tier(
        args.mid_model, ALL_TASKS, TASK_CATEGORIES, resolver, registry
    )
    strategies["Mid Only"] = sr_mid
    print(
        f"  \u2192 {sr_mid.success_rate:.0%} success, "
        f"avg ${sr_mid.avg_cost:.6f}/task"
    )

    # 3. Flagship Only
    print(f"\n{'─' * 50}")
    print(f"  Strategy: Flagship Only ({args.flagship_model})")
    sr_flagship = StrategyResult(
        name="Flagship Only", color="#e74c3c", marker="o"
    )
    sr_flagship.results = run_fixed_tier(
        args.flagship_model, ALL_TASKS, TASK_CATEGORIES, resolver, registry
    )
    strategies["Flagship Only"] = sr_flagship
    print(
        f"  \u2192 {sr_flagship.success_rate:.0%} success, "
        f"avg ${sr_flagship.avg_cost:.6f}/task"
    )

    # 4. TokenWise Escalation
    print(f"\n{'─' * 50}")
    print(f"  Strategy: TokenWise Escalation (start: {args.budget_model})")
    sr_esc = StrategyResult(
        name="TokenWise Escalation", color="#3498db", marker="*"
    )
    tier_models = [args.budget_model, args.mid_model, args.flagship_model]
    sr_esc.results = run_escalation(
        ALL_TASKS, TASK_CATEGORIES, tier_models, resolver, registry
    )
    strategies["TokenWise Escalation"] = sr_esc
    print(
        f"  \u2192 {sr_esc.success_rate:.0%} success, "
        f"avg ${sr_esc.avg_cost:.6f}/task"
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"{'Strategy':<25} {'Success':>8} {'Avg Cost':>12} {'Total Cost':>12}")
    print(f"{'─' * 60}")
    for name, sr in strategies.items():
        print(
            f"{name:<25} {sr.success_rate:>7.0%} "
            f"${sr.avg_cost:>11.6f} ${sr.total_cost:>11.6f}"
        )

    total_spent = sum(sr.total_cost for sr in strategies.values())
    print(f"{'─' * 60}")
    print(f"{'Total benchmark cost':<25} {'':>8} {'':>12} ${total_spent:>11.6f}")

    save_csv(strategies, args.csv)
    plot_pareto(strategies, args.output)


if __name__ == "__main__":
    main()
