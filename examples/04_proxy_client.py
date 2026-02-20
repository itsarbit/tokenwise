#!/usr/bin/env python3
"""Example 4: OpenAI-Compatible Proxy.

Starts the TokenWise proxy in a background thread, then sends requests via
raw httpx (proving any HTTP client works — no OpenAI SDK needed).

Demonstrates:
  - /v1/models endpoint
  - /v1/chat/completions with model="auto" (balanced routing)
  - Cheapest strategy via tokenwise options
  - Budget-constrained routing via tokenwise options

Requires OPENROUTER_API_KEY. Estimated cost: $0.01–$0.05 per run.

Usage:
    uv run python examples/04_proxy_client.py
"""

from __future__ import annotations

import threading
import time

import httpx
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

PROXY_HOST = "127.0.0.1"
PROXY_PORT = 8321  # Non-default port to avoid conflicts
BASE_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"


def start_proxy() -> threading.Thread:
    """Start the TokenWise proxy server in a daemon thread."""
    config = uvicorn.Config(
        "tokenwise.proxy:app",
        host=PROXY_HOST,
        port=PROXY_PORT,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for the server to be ready
    for _ in range(30):
        try:
            resp = httpx.get(f"{BASE_URL}/health", timeout=2.0)
            if resp.status_code == 200:
                return thread
        except httpx.ConnectError:
            pass
        time.sleep(0.3)

    raise RuntimeError("Proxy server failed to start")


def demo_list_models(client: httpx.Client) -> None:
    """Show the /v1/models endpoint."""
    console.print("[bold yellow]1. GET /v1/models[/bold yellow]")
    resp = client.get(f"{BASE_URL}/v1/models")
    resp.raise_for_status()
    data = resp.json()

    models = data.get("data", [])
    table = Table(title=f"Available Models ({len(models)} total, showing first 10)")
    table.add_column("Model ID", max_width=40)
    table.add_column("Provider")

    for m in models[:10]:
        table.add_row(m["id"], m.get("owned_by", ""))

    console.print(table)
    console.print()


def demo_auto_routing(client: httpx.Client) -> None:
    """Send a request with model="auto" (balanced routing)."""
    console.print('[bold yellow]2. POST /v1/chat/completions — model="auto"[/bold yellow]')

    resp = client.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Explain what a hash table is in 2 sentences."}
            ],
            "temperature": 0.7,
            "max_tokens": 200,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()

    model_used = data.get("model", "?")
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

    console.print(f"  Model used: [bold]{model_used}[/bold]")
    console.print(f"  Tokens: {tokens}")
    console.print(Panel(content.strip(), title="Response", border_style="green"))
    console.print()


def demo_cheapest_strategy(client: httpx.Client) -> None:
    """Send a request with the cheapest strategy via tokenwise options."""
    console.print('[bold yellow]3. POST /v1/chat/completions — strategy: "cheapest"[/bold yellow]')

    resp = client.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "temperature": 0.0,
            "max_tokens": 50,
            "tokenwise": {
                "strategy": "cheapest",
            },
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()

    model_used = data.get("model", "?")
    content = data["choices"][0]["message"]["content"]

    console.print(f"  Model used: [bold]{model_used}[/bold] (should be cheapest available)")
    console.print(Panel(content.strip(), title="Response", border_style="green"))
    console.print()


def demo_budget_constrained(client: httpx.Client) -> None:
    """Send a request with budget constraint via tokenwise options."""
    console.print(
        '[bold yellow]4. POST /v1/chat/completions — strategy: "budget_constrained"[/bold yellow]'
    )

    resp = client.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": "Write a Python function that checks if a string is a palindrome.",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 300,
            "tokenwise": {
                "strategy": "budget_constrained",
                "budget": 0.01,
            },
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()

    model_used = data.get("model", "?")
    content = data["choices"][0]["message"]["content"]

    console.print(f"  Model used: [bold]{model_used}[/bold] (best model within $0.01 budget)")
    console.print(Panel(content.strip(), title="Response", border_style="green"))


def main() -> None:
    console.print("\n[bold cyan]TokenWise — OpenAI-Compatible Proxy Demo[/bold cyan]\n")

    console.print(f"Starting proxy at {BASE_URL}...", end=" ")
    start_proxy()
    console.print("[green]ready![/green]\n")

    with httpx.Client() as client:
        demo_list_models(client)
        demo_auto_routing(client)
        demo_cheapest_strategy(client)
        demo_budget_constrained(client)

    console.print("\n[dim]Proxy runs in a daemon thread and exits with this process.[/dim]\n")


if __name__ == "__main__":
    main()
