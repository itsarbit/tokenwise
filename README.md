<p align="center">
  <img src="assets/logo.png" alt="TokenWise" width="540">
</p>

<h1 align="center">TokenWise</h1>

<p align="center">
  <a href="https://github.com/itsarbit/tokenwise/actions/workflows/ci.yml"><img src="https://github.com/itsarbit/tokenwise/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  <a href="https://pypi.org/project/tokenwise/"><img src="https://img.shields.io/pypi/v/tokenwise" alt="PyPI"></a>
</p>

<p align="center"><strong>Intelligent LLM Task Planner</strong> — decompose tasks, route to optimal models, enforce budgets.</p>

Existing LLM routers (RouteLLM, LLMRouter, Not Diamond) only do single-query routing: pick one model per request. TokenWise goes further — it **plans**: decomposes complex tasks into subtasks, assigns the right model to each step based on cost/quality/capability, enforces a token budget, and retries with a stronger model on failure.

> **Note:** TokenWise v0.1 uses [OpenRouter](https://openrouter.ai) as its model gateway. All model discovery, pricing, and LLM calls go through the OpenRouter API. You will need an OpenRouter API key to use TokenWise. Support for direct provider APIs (OpenAI, Anthropic, etc.) is planned for a future release.

## Features

- **Budget-aware planning** — "I have $0.50, get this done" → planner picks the cheapest viable path
- **Task decomposition** — Break complex tasks into subtasks, each routed to the right model
- **Model registry** — Knows model capabilities, prices, context windows (fetched from [OpenRouter](https://openrouter.ai))
- **Simple routing** — For single queries, pick the best model based on cost/quality preference
- **OpenAI-compatible proxy** — Drop-in replacement so existing apps benefit without code changes
- **CLI** — `tokenwise plan`, `tokenwise route`, `tokenwise serve`, `tokenwise models`

## How It Works

```
┌───────────────────────────────────────────────────────┐
│                       TokenWise                       │
│                                                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │   Router   │  │  Planner   │  │  Executor  │       │
│  │            │  │            │  │            │       │
│  │  Picks 1   │  │  Breaks    │  │  Runs the  │       │
│  │  model per │  │  task into │  │  plan,     │       │
│  │  query     │  │  steps +   │  │  tracks    │       │
│  │            │  │  assigns   │  │  spend,    │       │
│  │            │  │  models    │  │  retries   │       │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘       │
│        │               │               │              │
│        └───────────────┼───────────────┘              │
│                        ▼                              │
│            ┌──────────────┐                           │
│            │   Registry   │  ← metadata + pricing     │
│            └──────┬───────┘                           │
│                   ▼                                   │
│            ┌──────────────┐                           │
│            │  OpenRouter  │  ← all LLM calls          │
│            └──────────────┘                           │
└───────────────────────────────────────────────────────┘
```

**Router** picks the best model for a single query using one of four strategies:
- `cheapest` — lowest cost model that fits the requirement
- `best_quality` — flagship-tier model
- `balanced` — matches model tier to query complexity
- `budget_constrained` — best model that fits within a dollar budget

**Planner** decomposes a complex task into subtasks using a cheap LLM, then assigns the optimal model to each step within your budget. If the plan exceeds budget, it automatically downgrades expensive steps.

**Executor** runs a plan step by step, tracks actual token usage and cost, and escalates to a stronger model if a step fails.

## Requirements

- Python >= 3.10
- An [OpenRouter](https://openrouter.ai) API key

## Install

```bash
# With uv (recommended)
uv add tokenwise

# With pip
pip install tokenwise
```

## Quick Start

### 1. Set your OpenRouter API key

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### 2. CLI usage

```bash
# List available models and pricing
tokenwise models

# Route a single query to the best model
tokenwise route "Write a haiku about Python"

# Route with a specific strategy
tokenwise route "Debug this segfault" --strategy cheapest

# Plan a complex task with a budget
tokenwise plan "Build a REST API for a todo app" --budget 0.50

# Plan and execute immediately
tokenwise plan "Write unit tests for auth module" --budget 0.25 --execute

# Start the OpenAI-compatible proxy server
tokenwise serve --port 8000
```

### 3. Python API

```python
from tokenwise import Router, Planner
from tokenwise.executor import Executor

# Simple routing — picks the best model, returns ModelInfo
router = Router()
model = router.route("Explain quantum computing", strategy="balanced")
print(f"Use model: {model.id} (${model.input_price}/M input tokens)")

# Task planning with budget
planner = Planner()
plan = planner.plan(
    task="Build a REST API for a todo app",
    budget=0.50,
)
print(f"Plan: {len(plan.steps)} steps, estimated ${plan.total_estimated_cost:.4f}")

# Execute the plan
executor = Executor()
result = executor.execute(plan)
print(f"Done! Cost: ${result.total_cost:.4f}, success: {result.success}")
```

### 4. OpenAI-compatible proxy

Start the proxy, then point any OpenAI-compatible client at it:

```bash
tokenwise serve --port 8000
```

```python
from openai import OpenAI

# Point at TokenWise proxy — it routes automatically
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="auto",  # TokenWise picks the best model
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Routing Strategies

| Strategy | When to Use | How It Works |
|---|---|---|
| `cheapest` | Minimize cost | Picks the lowest-price model with the required capability |
| `best_quality` | Maximize quality | Picks the most capable flagship model |
| `balanced` | Default | Matches model tier to query complexity (short→budget, long→flagship) |
| `budget_constrained` | Hard budget limit | Picks the best-tier model that fits within the dollar budget |

## Configuration

TokenWise reads configuration from environment variables and an optional config file (`~/.config/tokenwise/config.yaml`).

| Variable | Description | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | **Required.** Your OpenRouter API key | — |
| `OPENROUTER_BASE_URL` | OpenRouter API base URL | `https://openrouter.ai/api/v1` |
| `TOKENWISE_DEFAULT_STRATEGY` | Default routing strategy | `balanced` |
| `TOKENWISE_DEFAULT_BUDGET` | Default budget in USD | `1.00` |
| `TOKENWISE_PLANNER_MODEL` | Model used for task decomposition | `openai/gpt-4.1-mini` |
| `TOKENWISE_PROXY_HOST` | Proxy server bind host | `127.0.0.1` |
| `TOKENWISE_PROXY_PORT` | Proxy server bind port | `8000` |
| `TOKENWISE_CACHE_TTL` | Model registry cache TTL (seconds) | `3600` |
| `TOKENWISE_LOCAL_MODELS` | Path to local models YAML for offline use | — |

### Config file example

```yaml
# ~/.config/tokenwise/config.yaml
default_strategy: balanced
default_budget: 0.50
planner_model: openai/gpt-4.1-mini
```

## Architecture

```
src/tokenwise/
├── models.py      # Pydantic data models (ModelInfo, Plan, Step, etc.)
├── config.py      # Settings from env vars and config file
├── registry.py    # ModelRegistry — fetches/caches models from OpenRouter
├── router.py      # Router — picks best model for a single query
├── planner.py     # Planner — decomposes tasks, assigns models per step
├── executor.py    # Executor — runs plans, tracks spend, escalates on failure
├── cli.py         # Typer CLI (models, route, plan, serve)
└── proxy.py       # FastAPI OpenAI-compatible proxy server
```

## Known Limitations (v0.1)

- **OpenRouter-only** — all LLM calls go through OpenRouter. Direct provider APIs are not yet supported.
- **Capability detection is heuristic** — model capabilities are inferred from model names, not from a canonical source.
- **Linear execution** — plan steps run sequentially; parallel step execution is not yet implemented.
- **Planner cost not budgeted** — the LLM call used to decompose the task is not deducted from the user's budget.

## Development

```bash
git clone https://github.com/itsarbit/tokenwise.git
cd tokenwise
uv sync
uv run pytest
uv run ruff check src/ tests/
uv run mypy src/
```

## License

MIT
