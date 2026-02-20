#!/usr/bin/env python3
"""Example 1: Routing Strategy Comparison.

Demonstrates the Router with all 4 strategies and capability detection.
No LLM calls are made — only registry lookups — so this is free to run.

Usage:
    uv run python examples/01_routing_strategies.py
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from tokenwise.models import RoutingStrategy
from tokenwise.registry import ModelRegistry
from tokenwise.router import Router

console = Console()

# Four diverse queries that exercise different capability detection paths
QUERIES = [
    ("Simple greeting", "Say hello in French"),
    ("Code task", "Write a Python function to parse CSV files and handle edge cases"),
    ("Math problem", "Calculate the integral of x^2 * e^x from 0 to infinity"),
    (
        "Long analytical question",
        "Compare and contrast microservices vs monolithic architecture. "
        "Analyze the trade-offs in terms of scalability, development velocity, "
        "operational complexity, and team organization. Provide a step-by-step "
        "framework for deciding which approach to use for a new project, "
        "considering factors like team size, expected traffic, and deployment "
        "constraints.",
    ),
]

STRATEGIES = [
    RoutingStrategy.CHEAPEST,
    RoutingStrategy.BALANCED,
    RoutingStrategy.BEST_QUALITY,
    RoutingStrategy.BUDGET_CONSTRAINED,
]


def main() -> None:
    console.print("\n[bold cyan]TokenWise — Routing Strategy Comparison[/bold cyan]\n")

    # Load the model registry (fetches from OpenRouter API)
    registry = ModelRegistry()
    console.print("Loading model registry from OpenRouter...", end=" ")
    count = registry.load_from_openrouter()
    console.print(f"[green]{count} models loaded.[/green]\n")

    router = Router(registry)

    # Build a Rich table: rows = queries, columns = strategies
    table = Table(title="Model Selected per Query × Strategy", show_lines=True)
    table.add_column("Query", style="bold", max_width=30)
    for strategy in STRATEGIES:
        table.add_column(strategy.value, max_width=28)

    for label, query in QUERIES:
        row = [label]
        for strategy in STRATEGIES:
            budget = 0.10 if strategy == RoutingStrategy.BUDGET_CONSTRAINED else None
            try:
                model = router.route(query, strategy=strategy, budget=budget)
                cell = f"[bold]{model.id}[/bold]\n"
                cell += f"Tier: {model.tier.value}\n"
                cell += f"${model.input_price:.2f}/M in"
            except ValueError as e:
                cell = f"[red]Error: {e}[/red]"
            row.append(cell)
        table.add_row(*row)

    console.print(table)

    # Show detected capabilities for each query
    console.print("\n[bold]Capability Detection:[/bold]")
    from tokenwise.router import _detect_capabilities, _estimate_complexity

    for label, query in QUERIES:
        caps = _detect_capabilities(query) or ["general"]
        complexity = _estimate_complexity(query)
        console.print(f"  {label}: caps={caps}, complexity={complexity}")

    console.print("\n[dim]No LLM calls were made — all routing is local.[/dim]\n")


if __name__ == "__main__":
    main()
