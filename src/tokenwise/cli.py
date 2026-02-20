"""Typer CLI for TokenWise."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from tokenwise.executor import Executor
from tokenwise.planner import Planner
from tokenwise.registry import ModelRegistry
from tokenwise.router import Router

app = typer.Typer(
    name="tokenwise",
    help="TokenWise — Intelligent LLM Task Planner",
    no_args_is_help=True,
)
console = Console()


@app.command()
def models(
    capability: str | None = typer.Option(None, "--capability", "-c", help="Filter by capability"),
    tier: str | None = typer.Option(
        None, "--tier", "-t", help="Filter by tier (flagship/mid/budget)"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max models to display"),
) -> None:
    """List available models and pricing."""
    registry = ModelRegistry()
    try:
        registry.ensure_loaded()
        console.print(f"[green]Loaded {len(registry.models)} models[/green]\n")
    except Exception as e:
        console.print(f"[red]Failed to load models: {e}[/red]")
        raise typer.Exit(1)

    from tokenwise.models import ModelTier

    all_models = registry.find_models(
        capability=capability,
        tier=ModelTier(tier) if tier else None,
    )

    table = Table(title="Available Models")
    table.add_column("Model ID", style="cyan", max_width=40)
    table.add_column("Tier", style="magenta")
    table.add_column("Input $/M", justify="right", style="green")
    table.add_column("Output $/M", justify="right", style="green")
    table.add_column("Context", justify="right")
    table.add_column("Capabilities", style="yellow")

    for m in all_models[:limit]:
        table.add_row(
            m.id,
            m.tier.value,
            f"${m.input_price:.2f}",
            f"${m.output_price:.2f}",
            f"{m.context_window:,}",
            ", ".join(m.capabilities) if m.capabilities else "-",
        )

    console.print(table)
    if len(all_models) > limit:
        console.print(
            f"\n[dim]Showing {limit} of {len(all_models)} models."
            " Use --limit to see more.[/dim]"
        )


@app.command()
def route(
    query: str = typer.Argument(help="The query to route"),
    strategy: str = typer.Option(
        "balanced", "--strategy", "-s",
        help="Routing strategy: cheapest, best_quality, balanced, budget_constrained",
    ),
    budget: float | None = typer.Option(
        None, "--budget", "-b", help="Budget in USD (for budget_constrained)"
    ),
    capability: str | None = typer.Option(None, "--capability", "-c", help="Required capability"),
) -> None:
    """Route a single query to the best model."""
    registry = ModelRegistry()
    router = Router(registry)

    try:
        model = router.route(
            query=query,
            strategy=strategy,
            budget=budget,
            required_capability=capability,
        )
    except ValueError as e:
        console.print(f"[red]Routing failed: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold green]Selected model:[/bold green] {model.id}")
    console.print(f"  Provider: {model.provider}")
    console.print(f"  Tier: {model.tier.value}")
    console.print(f"  Input price: ${model.input_price:.2f}/M tokens")
    console.print(f"  Output price: ${model.output_price:.2f}/M tokens")
    console.print(f"  Context window: {model.context_window:,} tokens")
    if model.capabilities:
        console.print(f"  Capabilities: {', '.join(model.capabilities)}")


@app.command()
def plan(
    task: str = typer.Argument(help="The task to plan"),
    budget: float = typer.Option(1.0, "--budget", "-b", help="Budget in USD"),
    execute: bool = typer.Option(
        False, "--execute", "-x", help="Execute the plan after creating it"
    ),
) -> None:
    """Plan a complex task — decompose into steps with model assignments."""
    registry = ModelRegistry()
    planner = Planner(registry=registry)

    with console.status("[bold green]Planning task..."):
        try:
            task_plan = planner.plan(task=task, budget=budget)
        except Exception as e:
            console.print(f"[red]Planning failed: {e}[/red]")
            raise typer.Exit(1)

    # Display the plan
    console.print(f"\n[bold]Plan for:[/bold] {task_plan.task}")
    console.print(f"[bold]Budget:[/bold] ${task_plan.budget:.2f}")
    console.print(f"[bold]Estimated cost:[/bold] ${task_plan.total_estimated_cost:.4f}")
    within = "[green]Yes[/green]" if task_plan.is_within_budget() else "[red]No[/red]"
    console.print(f"[bold]Within budget:[/bold] {within}\n")

    table = Table(title="Execution Steps")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Description", max_width=50)
    table.add_column("Model", style="magenta")
    table.add_column("Est. Cost", justify="right", style="green")
    table.add_column("Depends On", style="dim")

    for step in task_plan.steps:
        deps = ", ".join(str(d) for d in step.depends_on) if step.depends_on else "-"
        table.add_row(
            str(step.id),
            step.description,
            step.model_id,
            f"${step.estimated_cost:.4f}",
            deps,
        )

    console.print(table)

    if execute:
        console.print("\n[bold]Executing plan...[/bold]\n")
        executor = Executor(registry=registry)
        result = executor.execute(task_plan)

        status = "[green]Success[/green]" if result.success else "[red]Failed[/red]"
        console.print(f"\n[bold]Status:[/bold] {status}")
        console.print(f"[bold]Total cost:[/bold] ${result.total_cost:.4f}")
        console.print(f"[bold]Budget remaining:[/bold] ${result.budget_remaining:.4f}")

        if result.final_output:
            console.print(f"\n[bold]Output:[/bold]\n{result.final_output[:2000]}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
) -> None:
    """Start the OpenAI-compatible proxy server."""
    import uvicorn

    console.print(f"[bold green]Starting TokenWise proxy on {host}:{port}[/bold green]")
    console.print("[dim]OpenAI-compatible endpoint: POST /v1/chat/completions[/dim]\n")

    uvicorn.run(
        "tokenwise.proxy:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.callback()
def main() -> None:
    """TokenWise — Intelligent LLM Task Planner."""


if __name__ == "__main__":
    app()
