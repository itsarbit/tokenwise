<p align="center">
  <img src="assets/logo.png" alt="TokenWise" width="540">
</p>

<h1 align="center">TokenWise</h1>

<p align="center">
  <a href="https://github.com/itsarbit/tokenwise/actions/workflows/ci.yml"><img src="https://github.com/itsarbit/tokenwise/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  <a href="https://pypi.org/project/tokenwise-llm/"><img src="https://img.shields.io/pypi/v/tokenwise-llm" alt="PyPI"></a>
</p>

<p align="center">Production-grade LLM routing with budget ceilings, tiered escalation, and multi-provider failover.</p>

---

TokenWise is not just a model picker.

It is a lightweight control layer for LLM systems that need:

- **Strict budget enforcement** — hard cost ceilings that fail fast, never silently overspend
- **Capability-aware routing** — routes and fallbacks filtered by what the task actually needs (code, reasoning, math)
- **Deterministic escalation** — budget → mid → flagship, never downward
- **Task decomposition** — break complex work into subtasks, each routed to the right model
- **Multi-provider failover** — OpenRouter, OpenAI, Anthropic, Google — with a shared connection pool
- **An OpenAI-compatible proxy** — drop-in replacement for any existing SDK

Modern LLM applications are production systems. Production systems need guardrails. TokenWise provides those guardrails.

## Why TokenWise Exists

Most LLM routers do one thing: pick a model per request. That is not enough for real systems.

In production, you need a hard budget ceiling per task. You need tiered escalation that tries stronger models when weaker ones fail. You need provider failover. You need capability-aware routing that knows a coding task should not fall back to a model that cannot code. You need deterministic behavior you can reason about.

TokenWise treats routing as infrastructure — not a convenience feature.

> **Note:** TokenWise uses [OpenRouter](https://openrouter.ai) as the default model gateway for model discovery and routing. You can also use direct provider APIs (OpenAI, Anthropic, Google) by setting the corresponding API keys — when a direct key is available, requests for that provider bypass OpenRouter automatically.

## Comparison

| Feature | TokenWise | [RouteLLM](https://github.com/lm-sys/RouteLLM) | [LiteLLM](https://github.com/BerriAI/litellm) | [Not Diamond](https://notdiamond.ai) | [Martian](https://withmartian.com) | [Portkey](https://portkey.ai) | [OpenRouter](https://openrouter.ai) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Task decomposition | **Yes** | - | - | - | - | - | - |
| Strict budget ceiling | **Yes** | - | Yes | - | Per-request | Yes | Yes |
| Tier-based escalation | **Yes** | - | Yes | - | - | Yes | - |
| Capability-aware fallback | **Yes** | - | - | Partial | Yes | Partial | Partial |
| Cost ledger | **Yes** | - | Yes | - | - | Yes | Dashboard |
| OpenAI-compatible proxy | **Yes** | Yes | Yes | Yes | Yes | Yes | Yes |
| CLI | **Yes** | - | Yes | - | - | - | - |
| Python API | **Yes** | Yes | Yes | Yes | Via OpenAI SDK | Yes | Yes |
| Self-hosted / open source | **Yes** | Yes | Yes | - | - | Gateway only | - |

**Key differentiator:** TokenWise is the only router that also **plans** — it decomposes a complex task into subtasks, assigns the optimal model to each step within your budget, tracks spend across attempts with a structured cost ledger, and escalates to stronger models on failure. Every other tool on this list only routes individual queries.

---

## Core Features

### Budget-Aware Routing

Enforce a strict maximum cost per request or workflow. If no model fits within the ceiling, TokenWise fails fast. No silent overspending.

```python
router = Router()
model = router.route("Debug this segfault", strategy="best_quality", budget=0.05)
# Raises ValueError if nothing fits — never silently exceeds the limit
```

### Tiered Escalation

Three model tiers: **budget**, **mid**, **flagship**.

If a model fails, TokenWise escalates strictly upward. It never downgrades. Escalation preserves required capabilities — a failed code model is replaced by a stronger code model, not a generic one.

### Capability-Aware Selection

Routing considers capabilities: `code`, `reasoning`, `math`, `general`.

Fallback never selects a model that cannot perform the required task. Capabilities are tracked per step, not inferred at retry time.

### Task Decomposition

Break complex tasks into subtasks. Each step gets the right model at the right price.

```python
planner = Planner()
plan = planner.plan("Build a REST API for a todo app", budget=0.50)
# 4 steps, each with the cheapest viable model for its capability
```

### Cost Ledger

Every LLM call — successful or failed — is recorded in a structured `CostLedger`. See exactly where your money went across attempts and escalations.

### Multi-Provider Failover

Supports OpenRouter, OpenAI, Anthropic, and Google. Direct API keys bypass OpenRouter automatically. The proxy shares a single `httpx.AsyncClient` across all providers for connection pooling.

---

## Install

```bash
pip install tokenwise-llm
```

## Quick Start

### 1. Set your API key

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### 2. Use it

**CLI:**

```bash
# Route a query
tokenwise route "Write a haiku about Python"

# Route with budget ceiling
tokenwise route "Debug this segfault" --strategy best_quality --budget 0.05

# Plan and execute a complex task
tokenwise plan "Build a REST API for a todo app" --budget 0.50 --execute

# Start the OpenAI-compatible proxy
tokenwise serve --port 8000

# List models and pricing
tokenwise models
```

**Python API:**

```python
from tokenwise import Router, Planner
from tokenwise.executor import Executor

# Route a single query — detects scenario, picks best model within budget
router = Router()
model = router.route("Explain quantum computing", strategy="balanced", budget=0.10)
print(f"Use model: {model.id} (${model.input_price}/M input tokens)")

# Plan a complex task
planner = Planner()
plan = planner.plan(task="Build a REST API for a todo app", budget=0.50)
print(f"Plan: {len(plan.steps)} steps, estimated ${plan.total_estimated_cost:.4f}")

# Execute the plan — tracks spend, escalates on failure
executor = Executor()
result = executor.execute(plan)
print(f"Done! Cost: ${result.total_cost:.4f}, success: {result.success}")
```

**OpenAI-compatible proxy:**

```bash
tokenwise serve --port 8000
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="auto",  # TokenWise picks the best model
    messages=[{"role": "user", "content": "Hello!"}],
)
```

---

## How It Works

```
┌───────────────────────────────────────────────────────┐
│                       TokenWise                       │
│                                                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │   Router   │  │  Planner   │  │  Executor  │       │
│  │            │  │            │  │            │       │
│  │  1. Detect │  │  Breaks    │  │  Runs the  │       │
│  │  scenario  │  │  task into │  │  plan,     │       │
│  │  2. Route  │  │  steps +   │  │  tracks    │       │
│  │  within    │  │  assigns   │  │  spend,    │       │
│  │  budget    │  │  models    │  │  retries   │       │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘       │
│        │               │               │              │
│        └───────────────┼───────────────┘              │
│                        ▼                              │
│          ┌──────────────────────────┐                 │
│          │    ProviderResolver      │  ← LLM calls    │
│          │                          │                 │
│          │  OpenAI    · Anthropic   │                 │
│          │  Google    · OpenRouter  │                 │
│          └──────────────────────────┘                 │
│                                                       │
│            ┌──────────────┐                           │
│            │   Registry   │  ← metadata + pricing     │
│            └──────────────┘                           │
└───────────────────────────────────────────────────────┘
```

**Router** uses a two-stage pipeline for every request:

```
               ┌───────────────────┐      ┌────────────────────┐
 query ──────▶ │  1. Detect        │─────▶│  2. Route          │──────▶ model
               │     Scenario      │      │     with Strategy  │
               │                   │      │                    │
               │  · capabilities   │      │  · filter budget   │
               │    (code, reason, │      │  · cheapest /      │
               │     math)         │      │    balanced /      │
               │  · complexity     │      │    best_quality    │
               │    (simple → hard)│      │                    │
               └───────────────────┘      └────────────────────┘
```

**Router** separates *understanding what the query needs* from *choosing how to spend*. Budget is a universal parameter — not a strategy. By default, the router enforces the budget as a hard ceiling: if no model fits, it raises an error instead of silently exceeding the limit.

**Planner** decomposes a complex task into subtasks using a cheap LLM, then assigns the optimal model to each step within your budget. If the plan exceeds budget, it automatically downgrades expensive steps.

**Executor** runs a plan step by step, tracks actual token usage and cost via a `CostLedger`, and escalates to a stronger model if a step fails. Escalation tries stronger tiers first (flagship before mid) and filters by the step's required capabilities.

## Routing Strategies

| Strategy | When to Use | How It Works |
|---|---|---|
| `cheapest` | Minimize cost | Picks the lowest-price capable model |
| `best_quality` | Maximize quality | Picks the best flagship-tier capable model |
| `balanced` | Default | Matches model tier to query complexity (short→budget, long→flagship) |

All strategies enforce the budget as a hard ceiling. Pass `budget_strict=False` in the Python API to fall back to best-effort behavior.

## Configuration

TokenWise reads configuration from environment variables and an optional config file (`~/.config/tokenwise/config.yaml`).

| Variable | Required | Description | Default |
|---|---|---|---|
| `OPENROUTER_API_KEY` | **Yes** | OpenRouter API key (model discovery + fallback for LLM calls) | — |
| `OPENAI_API_KEY` | Optional | Direct OpenAI API key; falls back to OpenRouter if not set | — |
| `ANTHROPIC_API_KEY` | Optional | Direct Anthropic API key; falls back to OpenRouter if not set | — |
| `GOOGLE_API_KEY` | Optional | Direct Google AI API key; falls back to OpenRouter if not set | — |
| `OPENROUTER_BASE_URL` | Optional | OpenRouter API base URL | `https://openrouter.ai/api/v1` |
| `TOKENWISE_DEFAULT_STRATEGY` | Optional | Default routing strategy | `balanced` |
| `TOKENWISE_DEFAULT_BUDGET` | Optional | Default budget in USD | `1.00` |
| `TOKENWISE_PLANNER_MODEL` | Optional | Model used for task decomposition | `openai/gpt-4.1-mini` |
| `TOKENWISE_PROXY_HOST` | Optional | Proxy server bind host | `127.0.0.1` |
| `TOKENWISE_PROXY_PORT` | Optional | Proxy server bind port | `8000` |
| `TOKENWISE_CACHE_TTL` | Optional | Model registry cache TTL (seconds) | `3600` |
| `TOKENWISE_LOCAL_MODELS` | Optional | Path to local models YAML for offline use | — |

```yaml
# ~/.config/tokenwise/config.yaml
default_strategy: balanced
default_budget: 0.50
planner_model: openai/gpt-4.1-mini
```

## Architecture

```
src/tokenwise/
├── models.py          # Pydantic data models (ModelInfo, Plan, Step, etc.)
├── config.py          # Settings from env vars and config file
├── registry.py        # ModelRegistry — fetches/caches models from OpenRouter
├── router.py          # Router — two-stage pipeline: scenario → strategy
├── planner.py         # Planner — decomposes tasks, assigns models per step
├── executor.py        # Executor — runs plans, tracks spend, escalates on failure
├── cli.py             # Typer CLI (models, route, plan, serve)
├── proxy.py           # FastAPI OpenAI-compatible proxy server
├── providers/         # LLM provider adapters
│   ├── openrouter.py  #   OpenRouter (default, routes via openrouter.ai)
│   ├── openai.py      #   Direct OpenAI API
│   ├── anthropic.py   #   Direct Anthropic Messages API
│   ├── google.py      #   Direct Google Gemini API
│   └── resolver.py    #   Maps model IDs → provider instances
└── data/
    └── model_capabilities.json  # Curated model family → capabilities mapping
```

## Philosophy

LLM systems should be treated like distributed systems.

That means clear failure semantics, explicit cost ceilings, predictable escalation, and observability. TokenWise is designed with that philosophy.

## Known Limitations (v0.3)

- **Linear execution** — plan steps run sequentially; parallel step execution is not yet implemented.
- **Planner cost not budgeted** — the LLM call used to decompose the task is not deducted from the user's budget.
- **No persistent spend tracking** — the `CostLedger` lives in memory for a single plan execution; there is no cross-session spend history yet.

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
