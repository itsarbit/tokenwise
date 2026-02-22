#!/usr/bin/env python3
"""Pareto benchmark — cost vs quality scatter plot across model tiers.

Runs a fixed set of short tasks through TokenWise's executor at different
budget tiers, collects cost and success metrics, saves results to CSV, and
generates a Pareto-front scatter plot.

Usage:
    uv run python benchmarks/pareto.py                # run benchmark + plot
    uv run python benchmarks/pareto.py --dry-run      # show plan without executing
    uv run python benchmarks/pareto.py --output out.png  # custom plot path
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

TASKS = [
    "Write a haiku about distributed systems",
    "Write a Python function that checks if a string is a palindrome",
    "What is the derivative of x^3 * sin(x)?",
    "Explain the CAP theorem in three sentences",
    "Summarize the benefits and drawbacks of microservices in a bullet list",
]

# Budget tiers to test each model at
BUDGET_PER_TASK = 0.05


@dataclass
class ModelResult:
    model_id: str
    tasks_run: int = 0
    successes: int = 0
    total_cost: float = 0.0
    costs: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.tasks_run if self.tasks_run else 0.0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.tasks_run if self.tasks_run else 0.0


def discover_models() -> list[str]:
    """Discover budget and mid tier models from the registry."""
    from tokenwise.models import ModelTier
    from tokenwise.registry import ModelRegistry

    registry = ModelRegistry()
    registry.load_from_openrouter()

    models = []
    for tier in [ModelTier.BUDGET, ModelTier.MID]:
        tier_models = registry.find_models(tier=tier, max_input_price=5.0)
        for m in tier_models[:3]:
            models.append(m.id)
    return models


def run_benchmark(model_ids: list[str]) -> dict[str, ModelResult]:
    """Execute all tasks on each model and collect metrics."""
    from tokenwise.executor import Executor
    from tokenwise.models import Plan, Step
    from tokenwise.registry import ModelRegistry

    registry = ModelRegistry()
    registry.load_from_openrouter()
    executor = Executor(registry=registry)

    results: dict[str, ModelResult] = {}

    for model_id in model_ids:
        mr = ModelResult(model_id=model_id)
        model = registry.get_model(model_id)
        if not model:
            print(f"  Skipping {model_id} — not in registry")
            continue

        print(f"\n  Model: {model_id} (${model.input_price}/M in, ${model.output_price}/M out)")

        for i, task in enumerate(TASKS):
            est_cost = model.estimate_cost(200, 300)
            step = Step(
                id=1,
                description=task,
                model_id=model_id,
                estimated_input_tokens=200,
                estimated_output_tokens=300,
                estimated_cost=est_cost,
            )
            plan = Plan(
                task=task,
                steps=[step],
                total_estimated_cost=est_cost,
                budget=BUDGET_PER_TASK,
            )

            result = executor.execute(plan)
            mr.tasks_run += 1
            mr.total_cost += result.total_cost
            mr.costs.append(result.total_cost)
            if result.success:
                mr.successes += 1

            status = "ok" if result.success else "FAIL"
            print(f"    Task {i + 1}: {status} (${result.total_cost:.6f})")

        results[model_id] = mr
        print(f"  => {mr.successes}/{mr.tasks_run} passed, avg ${mr.avg_cost:.6f}/task")

    return results


def save_csv(results: dict[str, ModelResult], csv_path: str) -> None:
    """Save benchmark results to CSV."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_id",
                "tasks_run",
                "successes",
                "success_rate",
                "total_cost",
                "avg_cost",
            ]
        )
        for model_id, mr in results.items():
            writer.writerow(
                [
                    model_id,
                    mr.tasks_run,
                    mr.successes,
                    f"{mr.success_rate:.4f}",
                    f"{mr.total_cost:.8f}",
                    f"{mr.avg_cost:.8f}",
                ]
            )

    print(f"Results saved to {path}")


def plot_pareto(results: dict[str, ModelResult], output_path: str) -> None:
    """Generate a Pareto-front scatter plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed. Install with: uv sync --group benchmark")
        print("Skipping plot generation.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_id, mr in results.items():
        if mr.tasks_run == 0:
            continue
        short_name = model_id.split("/")[-1] if "/" in model_id else model_id
        ax.scatter(mr.avg_cost * 1000, mr.success_rate * 100, s=100, zorder=5)
        ax.annotate(
            short_name,
            (mr.avg_cost * 1000, mr.success_rate * 100),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )

    ax.set_xlabel("Avg Cost per Task ($ x 1000)", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title("TokenWise Pareto Benchmark — Cost vs Quality", fontsize=13)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TokenWise Pareto benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument(
        "--output",
        default="benchmarks/pareto.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--csv",
        default="benchmarks/results.csv",
        help="Output CSV path",
    )
    parser.add_argument("--models", nargs="*", help="Specific model IDs to benchmark")
    args = parser.parse_args()

    print("TokenWise Pareto Benchmark")
    print("=" * 40)

    if args.models:
        model_ids = args.models
    else:
        print("\nDiscovering models from OpenRouter...")
        model_ids = discover_models()

    print(f"\nModels: {len(model_ids)}")
    for m in model_ids:
        print(f"  - {m}")
    print(f"Tasks: {len(TASKS)}")
    print(f"Budget per task: ${BUDGET_PER_TASK}")

    if args.dry_run:
        print("\n--dry-run: would run the above benchmark. Exiting.")
        sys.exit(0)

    print("\nRunning benchmark...")
    results = run_benchmark(model_ids)

    print("\n" + "=" * 40)
    print("Summary")
    print("=" * 40)
    for model_id, mr in results.items():
        print(f"  {model_id}: {mr.success_rate:.0%} success, avg ${mr.avg_cost:.6f}/task")

    save_csv(results, args.csv)
    plot_pareto(results, args.output)


if __name__ == "__main__":
    main()
