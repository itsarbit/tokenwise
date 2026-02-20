# TokenWise Examples

Runnable scripts demonstrating each core feature of TokenWise.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (or pip)
- An OpenRouter API key — get one at <https://openrouter.ai/keys>

```bash
export OPENROUTER_API_KEY="sk-or-..."
cd /path/to/tokenwise
```

## Examples

### 01 — Routing Strategy Comparison

Routes 4 diverse queries through all 4 strategies and displays a comparison table.
**Free to run** — no LLM calls, only registry lookups.

```bash
uv run python examples/01_routing_strategies.py
```

### 02 — Plan & Execute Pipeline

The headline demo. Decomposes a complex task into steps via the Planner, then
executes each step with cost tracking and automatic escalation on failure.

```bash
uv run python examples/02_plan_and_execute.py
uv run python examples/02_plan_and_execute.py --task "Write a CLI calculator in Python"
uv run python examples/02_plan_and_execute.py --budget 0.10
```

Estimated cost: **$0.05–$0.20** per run.

### 03 — Budget Impact Visualization

Plans the same task at 3 budget levels ($0.05, $0.25, $1.00) and shows how
model tier assignments change. Planning-only (no execution).

```bash
uv run python examples/03_budget_comparison.py
uv run python examples/03_budget_comparison.py --task "Build a web scraper with error handling"
```

Estimated cost: **$0.01–$0.05** per run.

### 04 — OpenAI-Compatible Proxy

Starts the proxy server in a background thread, then sends requests via raw
`httpx` — proving any HTTP client works (no OpenAI SDK needed). Demonstrates
auto-routing, cheapest strategy, and budget-constrained routing.

```bash
uv run python examples/04_proxy_client.py
```

Estimated cost: **$0.01–$0.05** per run.

## Total Cost

All examples combined: **< $1.00**.
