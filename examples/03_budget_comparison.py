#!/usr/bin/env python3
"""Example 3: Budget Impact Visualization.

Plans the same task at 3 budget levels and shows how model tier assignments
change. Planning-only (no execution), so it's fast and cheap to run.

Requires OPENROUTER_API_KEY. Estimated cost: $0.01–$0.05 per run.

Usage:
    uv run python examples/03_budget_comparison.py
    uv run python examples/03_budget_comparison.py --task "Build a web scraper with error handling"
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from tokenwise import Planner

console = Console()

DEFAULT_TASK = (
    "Build a Python web scraper that extracts product data from an e-commerce site, "
    "stores results in a SQLite database, and generates a summary report with charts."
)

BUDGET_LEVELS = [
    (0.05, "Tight ($0.05)"),
    (0.25, "Moderate ($0.25)"),
    (1.00, "Generous ($1.00)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TokenWise Budget Comparison demo")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task to plan at multiple budgets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    console.print("\n[bold cyan]TokenWise — Budget Impact Comparison[/bold cyan]\n")
    console.print(f"Task: [italic]{args.task}[/italic]\n")

    planner = Planner()
    plans = []

    for budget, label in BUDGET_LEVELS:
        console.print(f"Planning at budget {label}...", end=" ")
        plan = planner.plan(task=args.task, budget=budget)
        plans.append((label, budget, plan))
        console.print(f"[green]{len(plan.steps)} steps, ${plan.total_estimated_cost:.4f}[/green]")

    console.print()

    # Detailed comparison table
    for label, budget, plan in plans:
        table = Table(title=f"Budget: {label}", show_lines=True)
        table.add_column("#", style="bold", width=3)
        table.add_column("Step", max_width=40)
        table.add_column("Model", max_width=28)
        table.add_column("Tier", width=10)
        table.add_column("Est. Cost", justify="right")

        for step in plan.steps:
            # Look up model tier from registry
            model = planner.registry.get_model(step.model_id)
            tier = model.tier.value if model else "?"
            table.add_row(
                str(step.id),
                step.description,
                step.model_id,
                tier,
                f"${step.estimated_cost:.4f}",
            )

        console.print(table)
        status = "[green]within budget" if plan.is_within_budget() else "[red]over budget"
        console.print(f"  Total: ${plan.total_estimated_cost:.4f} / ${budget:.2f} ({status}[/])\n")

    # Summary comparison
    summary = Table(title="Summary: Budget vs. Model Tiers", show_lines=True)
    summary.add_column("Budget Level")
    summary.add_column("Steps", justify="right")
    summary.add_column("Flagship", justify="right")
    summary.add_column("Mid", justify="right")
    summary.add_column("Budget Tier", justify="right")
    summary.add_column("Est. Total", justify="right")
    summary.add_column("Fits?", justify="center")

    for label, budget, plan in plans:
        tier_counts = {"flagship": 0, "mid": 0, "budget": 0}
        for step in plan.steps:
            model = planner.registry.get_model(step.model_id)
            if model:
                tier_counts[model.tier.value] = tier_counts.get(model.tier.value, 0) + 1

        fits = "[green]Yes" if plan.is_within_budget() else "[red]No"
        summary.add_row(
            label,
            str(len(plan.steps)),
            str(tier_counts.get("flagship", 0)),
            str(tier_counts.get("mid", 0)),
            str(tier_counts.get("budget", 0)),
            f"${plan.total_estimated_cost:.4f}",
            fits,
        )

    console.print(summary)
    console.print(
        "\n[dim]Higher budgets allow flagship models for complex steps; "
        "tight budgets force downgrades to cheaper tiers.[/dim]\n"
    )


if __name__ == "__main__":
    main()
