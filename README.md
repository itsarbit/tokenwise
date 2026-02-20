# TokenWise

**Intelligent LLM Task Planner** — decompose tasks, route to optimal models, enforce budgets.

Existing LLM routers only do single-query routing: pick one model per request. TokenWise goes further — it **plans**: decomposes complex tasks into subtasks, assigns the right model to each step based on cost/quality/capability, enforces a token budget, and retries with a stronger model on failure.

## Features

- **Budget-aware planning** — "I have $0.50, get this done" → planner picks the cheapest viable path
- **Task decomposition** — Break complex tasks into subtasks, each routed to the right model
- **Model registry** — Knows model capabilities, prices, context windows (via OpenRouter)
- **Simple routing** — For single queries, pick the best model based on cost/quality preference
- **OpenAI-compatible proxy** — Drop-in replacement so existing apps benefit without code changes
- **CLI** — `tokenwise plan`, `tokenwise route`, `tokenwise serve`

## Install

```bash
# With uv (recommended)
uv add tokenwise

# With pip
pip install tokenwise
```

## Quick Start

### Set your API key

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### CLI usage

```bash
# List available models and pricing
tokenwise models

# Route a single query to the best model
tokenwise route "Write a haiku about Python"

# Plan a complex task with a budget
tokenwise plan "Build a REST API for a todo app" --budget 0.50

# Start the OpenAI-compatible proxy server
tokenwise serve --port 8000
```

### Python API

```python
from tokenwise import Router, Planner

# Simple routing
router = Router()
model = router.route("Explain quantum computing", strategy="balanced")

# Task planning with budget
planner = Planner()
plan = planner.plan(
    task="Build a REST API for a todo app",
    budget=0.50,
)
result = plan.execute()
```

## Configuration

TokenWise reads configuration from environment variables and an optional config file (`~/.config/tokenwise/config.yaml`).

| Variable | Description | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API key | — |
| `TOKENWISE_DEFAULT_STRATEGY` | Default routing strategy | `balanced` |
| `TOKENWISE_DEFAULT_BUDGET` | Default budget in USD | `1.00` |
| `TOKENWISE_PLANNER_MODEL` | Model used for task planning | `openai/gpt-4.1-mini` |

## Development

```bash
git clone https://github.com/tokenwise/tokenwise.git
cd tokenwise
uv sync
uv run pytest
uv run ruff check src/
```

## License

MIT
