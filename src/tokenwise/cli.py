"""Typer CLI for TokenWise."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tokenwise.config import get_settings
from tokenwise.executor import Executor
from tokenwise.ledger_store import LedgerStore
from tokenwise.models import RiskGateBlockedError
from tokenwise.planner import Planner
from tokenwise.registry import ModelRegistry
from tokenwise.router import Router

app = typer.Typer(
    name="tokenwise",
    help="TokenWise — Intelligent LLM Task Planner",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
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

    _tier_rank = {ModelTier.FLAGSHIP: 0, ModelTier.MID: 1, ModelTier.BUDGET: 2}
    all_models = sorted(
        registry.find_models(
            capability=capability,
            tier=ModelTier(tier) if tier else None,
        ),
        key=lambda m: (_tier_rank.get(m.tier, 9), -m.input_price),
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
            f"\n[dim]Showing {limit} of {len(all_models)} models. Use --limit to see more.[/dim]"
        )


@app.command()
def route(
    query: str = typer.Argument(help="The query to route"),
    strategy: str = typer.Option(
        "balanced",
        "--strategy",
        "-s",
        help="Routing strategy: cheapest, best_quality, balanced",
    ),
    budget: float | None = typer.Option(
        None, "--budget", "-b", help="Max budget in USD (applies to all strategies)"
    ),
    capability: str | None = typer.Option(None, "--capability", "-c", help="Required capability"),
) -> None:
    """Route a single query to the best model."""
    registry = ModelRegistry()
    router = Router(registry)

    try:
        model, trace = router.route_with_trace(
            query=query,
            strategy=strategy,
            budget=budget,
            required_capability=capability,
        )
    except RiskGateBlockedError as e:
        console.print(f"[red]Risk gate blocked: {e.reason}[/red]")
        if e.trace:
            console.print(f"  Request ID: {e.trace.request_id}")
            console.print(f"  Termination: {e.trace.termination_state.value}")
        raise typer.Exit(1)
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
    console.print(f"\n[dim]  Request ID: {trace.request_id}[/dim]")
    console.print(f"[dim]  Termination: {trace.termination_state.value}[/dim]")


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
    if task_plan.decomposition_source == "fallback":
        console.print(
            f"[yellow]Warning: LLM decomposition failed, using single-step fallback. "
            f"({task_plan.decomposition_error or 'unknown error'})[/yellow]"
        )
    console.print(f"[bold]Budget:[/bold] ${task_plan.budget:.2f}")
    console.print(f"[bold]Estimated cost:[/bold] ${task_plan.total_estimated_cost:.4f}")
    if task_plan.planner_cost > 0:
        console.print(f"[bold]Planner cost:[/bold] ${task_plan.planner_cost:.4f}")
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

    # Show parallelism info
    independent = sum(1 for s in task_plan.steps if not s.depends_on)
    if independent > 1:
        console.print(f"\n[dim]{independent} steps can run in parallel (no dependencies)[/dim]")

    if execute:
        console.print("\n[bold]Executing plan...[/bold]\n")
        executor = Executor(registry=registry)
        result = executor.execute(task_plan)

        status = "[green]Success[/green]" if result.success else "[red]Failed[/red]"
        console.print(f"\n[bold]Status:[/bold] {status}")
        console.print(f"[bold]Total cost:[/bold] ${result.total_cost:.4f}")
        console.print(f"[bold]Budget remaining:[/bold] ${result.budget_remaining:.4f}")

        if result.skipped_steps:
            console.print(
                f"\n[yellow]Skipped {len(result.skipped_steps)} step(s) "
                f"due to budget exhaustion:[/yellow]"
            )
            for s in result.skipped_steps:
                console.print(f"  [dim]Step {s.id}: {s.description}[/dim]")

        if result.ledger.entries:
            ledger_table = Table(title="Cost Breakdown")
            ledger_table.add_column("Reason", style="cyan")
            ledger_table.add_column("Model", style="magenta")
            ledger_table.add_column("In Tokens", justify="right")
            ledger_table.add_column("Out Tokens", justify="right")
            ledger_table.add_column("Cost", justify="right", style="green")
            ledger_table.add_column("OK?", justify="center")

            for entry in result.ledger.entries:
                ledger_table.add_row(
                    entry.reason,
                    entry.model_id,
                    str(entry.input_tokens),
                    str(entry.output_tokens),
                    f"${entry.cost:.6f}",
                    "[green]Y[/green]" if entry.success else "[red]N[/red]",
                )

            console.print()
            console.print(ledger_table)
            if result.ledger.wasted_cost > 0:
                console.print(
                    f"[yellow]Wasted cost (failed attempts): "
                    f"${result.ledger.wasted_cost:.6f}[/yellow]"
                )

        if result.final_output:
            console.print(f"\n[bold]Output:[/bold]\n{result.final_output[:2000]}")

        # Display routing trace summary
        if result.routing_trace:
            rt = result.routing_trace
            console.print("\n[bold]Routing Trace:[/bold]")
            console.print(f"  Request ID: {rt.request_id}")
            console.print(f"  Initial model: {rt.initial_model} ({rt.initial_tier.value})")
            console.print(f"  Final model: {rt.final_model} ({rt.final_tier.value})")
            console.print(f"  Termination: {rt.termination_state.value}")
            console.print(f"  Escalation policy: {rt.escalation_policy.value}")
            if rt.escalations:
                esc_table = Table(title="Escalations")
                esc_table.add_column("Step", justify="right", style="cyan")
                esc_table.add_column("From", style="magenta")
                esc_table.add_column("To", style="green")
                esc_table.add_column("Reason", style="yellow")
                for esc in rt.escalations:
                    esc_table.add_row(
                        str(esc.step_id or "-"),
                        f"{esc.from_model} ({esc.from_tier.value})",
                        f"{esc.to_model} ({esc.to_tier.value})",
                        esc.reason_code.value,
                    )
                console.print(esc_table)

        # Persist to ledger store
        settings = get_settings()
        store_path = Path(settings.ledger_path) if settings.ledger_path else None
        store = LedgerStore(path=store_path)
        store.save(
            task=task_plan.task,
            ledger=result.ledger,
            budget=task_plan.budget,
            success=result.success,
            planner_cost=task_plan.planner_cost,
            trace=result.routing_trace,
        )
        console.print(f"\n[dim]Saved to ledger: {store.path}[/dim]")


@app.command()
def ledger(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of recent entries to show"),
    summary: bool = typer.Option(False, "--summary", "-s", help="Show aggregate spend statistics"),
) -> None:
    """Show persistent spend history across sessions."""
    settings = get_settings()
    store_path = Path(settings.ledger_path) if settings.ledger_path else None
    store = LedgerStore(path=store_path)

    if summary:
        stats = store.summary()
        console.print("\n[bold]Spend Summary[/bold]")
        console.print(f"  Total tasks:    {stats['num_tasks']}")
        console.print(f"  Succeeded:      {stats['num_succeeded']}")
        console.print(f"  Failed:         {stats['num_failed']}")
        console.print(f"  Total spend:    ${stats['total_spend']:.4f}")
        console.print(f"  Planner spend:  ${stats['total_planner_spend']:.4f}")
        console.print(f"  Wasted spend:   ${stats['total_wasted']:.4f}")
        return

    records = store.load(limit=limit)
    if not records:
        console.print("[dim]No ledger entries found.[/dim]")
        return

    table = Table(title="Spend History")
    table.add_column("Timestamp", style="dim", max_width=20)
    table.add_column("Task", max_width=40)
    table.add_column("Cost", justify="right", style="green")
    table.add_column("Budget", justify="right")
    table.add_column("OK?", justify="center")

    for r in records:
        ts = r.get("timestamp", "?")[:19]
        task_str = r.get("task", "?")
        if len(task_str) > 40:
            task_str = task_str[:37] + "..."
        cost = r.get("total_cost", 0.0)
        budget_val = r.get("budget", 0.0)
        ok = "[green]Y[/green]" if r.get("success") else "[red]N[/red]"
        table.add_row(ts, task_str, f"${cost:.4f}", f"${budget_val:.2f}", ok)

    console.print(table)


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
