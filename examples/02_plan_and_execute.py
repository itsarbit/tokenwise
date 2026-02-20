#!/usr/bin/env python3
"""Example 2: Full Plan & Execute Pipeline.

Demonstrates the Planner (LLM decomposition) and Executor (step-by-step
execution with cost tracking and escalation).

Requires OPENROUTER_API_KEY. Estimated cost: $0.05–$0.20 per run.

Usage:
    uv run python examples/02_plan_and_execute.py
    uv run python examples/02_plan_and_execute.py --task "Write a CLI calculator in Python"
    uv run python examples/02_plan_and_execute.py --budget 0.10
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tokenwise import Executor, Planner

console = Console()

DEFAULT_TASK = (
    "Design a REST API for a task management app with user authentication. "
    "Include endpoint definitions, a database schema, and example request/response payloads."
)
DEFAULT_BUDGET = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TokenWise Plan & Execute demo")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task to decompose and execute")
    parser.add_argument(
        "--budget", type=float, default=DEFAULT_BUDGET, help="Budget in USD (default: 0.50)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    console.print("\n[bold cyan]TokenWise — Plan & Execute Pipeline[/bold cyan]\n")
    console.print(Panel(args.task, title="Task", border_style="blue"))
    console.print(f"Budget: [bold green]${args.budget:.2f}[/bold green]\n")

    # --- Phase 1: Planning ---
    console.print("[bold yellow]Phase 1: Planning...[/bold yellow]")
    planner = Planner()
    plan = planner.plan(task=args.task, budget=args.budget)

    plan_table = Table(title=f"Execution Plan ({len(plan.steps)} steps)", show_lines=True)
    plan_table.add_column("#", style="bold", width=3)
    plan_table.add_column("Step Description", max_width=50)
    plan_table.add_column("Model", max_width=28)
    plan_table.add_column("Est. Tokens", justify="right")
    plan_table.add_column("Est. Cost", justify="right")

    for step in plan.steps:
        plan_table.add_row(
            str(step.id),
            step.description,
            step.model_id,
            f"{step.estimated_input_tokens}+{step.estimated_output_tokens}",
            f"${step.estimated_cost:.4f}",
        )

    console.print(plan_table)
    budget_status = "[green]within budget" if plan.is_within_budget() else "[red]over budget"
    console.print(
        f"\nTotal estimated cost: [bold]${plan.total_estimated_cost:.4f}[/bold]"
        f" ({budget_status}[/])\n"
    )

    # --- Phase 2: Execution ---
    console.print("[bold yellow]Phase 2: Executing...[/bold yellow]\n")
    executor = Executor()
    result = executor.execute(plan)

    result_table = Table(title="Execution Results", show_lines=True)
    result_table.add_column("#", style="bold", width=3)
    result_table.add_column("Model Used", max_width=28)
    result_table.add_column("Tokens (in/out)", justify="right")
    result_table.add_column("Actual Cost", justify="right")
    result_table.add_column("Status", justify="center")

    for sr in result.step_results:
        status = "[green]OK" if sr.success else "[red]FAIL"
        if sr.escalated:
            status += " [yellow](escalated)"
        result_table.add_row(
            str(sr.step_id),
            sr.model_id,
            f"{sr.input_tokens}/{sr.output_tokens}",
            f"${sr.actual_cost:.4f}",
            status,
        )

    console.print(result_table)

    # --- Summary ---
    overall = "[bold green]SUCCESS" if result.success else "[bold red]FAILED"
    console.print(f"\nOverall: {overall}[/]")
    console.print(f"Total cost:  [bold]${result.total_cost:.4f}[/bold]")
    console.print(
        f"Budget remaining: [bold]${result.budget_remaining:.4f}[/bold] of ${result.budget:.2f}\n"
    )

    if result.final_output:
        console.print(Panel(result.final_output[:2000], title="Final Output", border_style="green"))
    else:
        console.print("[red]No output was produced.[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
