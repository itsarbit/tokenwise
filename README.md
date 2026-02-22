<p align="center">
  <img src="assets/logo.png" alt="TokenWise" width="540">
</p>

<h1 align="center">TokenWise</h1>

<p align="center">
  <a href="https://github.com/itsarbit/tokenwise/actions/workflows/ci.yml"><img src="https://github.com/itsarbit/tokenwise/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  <a href="https://pypi.org/project/tokenwise-llm/"><img src="https://img.shields.io/pypi/v/tokenwise-llm?v=1" alt="PyPI"></a>
</p>

<p align="center">
Production-grade LLM routing with budget ceilings,
tiered escalation, and multi-provider failover.
</p>

---

TokenWise is not just a model picker.

It is a lightweight control layer for LLM systems that need:

- **Strict budget enforcement** — hard cost ceilings that fail
  fast, never silently overspend
- **Capability-aware routing** — routes and fallbacks filtered
  by what the task actually needs (code, reasoning, math)
- **Deterministic escalation** — budget to mid to flagship,
  never downward
- **Task decomposition** — break complex work into subtasks,
  each routed to the right model
- **Multi-provider failover** — OpenRouter, OpenAI, Anthropic,
  Google — with a shared connection pool
- **An OpenAI-compatible proxy** — drop-in replacement for any
  existing SDK

Modern LLM applications are production systems.
Production systems need guardrails.
TokenWise provides those guardrails.

## Why TokenWise Exists

Most LLM routers do one thing: pick a model per request.
That is not enough for real systems.

In production, you need a hard budget ceiling per task.
You need tiered escalation that tries stronger models when
weaker ones fail. You need provider failover. You need
capability-aware routing that knows a coding task should not
fall back to a model that cannot code. You need deterministic
behavior you can reason about.

TokenWise treats routing as infrastructure — not a convenience
feature.

> **Note:** TokenWise uses [OpenRouter](https://openrouter.ai)
> as the default model gateway for model discovery and routing.
> You can also use direct provider APIs (OpenAI, Anthropic,
> Google) by setting the corresponding API keys — when a direct
> key is available, requests for that provider bypass OpenRouter
> automatically.

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

**What these terms mean in TokenWise's context:**

- **Task decomposition** — breaks a complex prompt into multiple
  LLM steps, each assigned to a different model. Not just model
  selection per request.
- **Strict budget ceiling** — hard cap on total USD spend;
  execution stops rather than overshooting. Some tools offer
  per-request limits but not cross-step budgets.
- **Tier-based escalation** — on failure, retries with a
  stronger-tier model (budget, mid, flagship), never downward.
- **Capability-aware fallback** — fallback candidates are
  filtered by required capabilities (code, reasoning, math),
  not just price or tier.
- **Cost ledger** — structured per-call log of model, tokens,
  cost, and success/failure — including failed attempts and
  escalations.

Note: some competitors may partially cover these features.
The table reflects our understanding as of February 2026;
corrections welcome via
[issues](https://github.com/itsarbit/tokenwise/issues).

---

## Core Features

### Budget-Aware Routing

Enforce a strict maximum cost per request or workflow. If no
model fits within the ceiling, TokenWise fails fast. No silent
overspending.

```python
router = Router()
model = router.route(
    "Debug this segfault",
    strategy="best_quality",
    budget=0.05,
)
# Raises ValueError if nothing fits
```

### Tiered Escalation

Three model tiers: **budget**, **mid**, **flagship**.

If a model fails, TokenWise escalates strictly upward. It never
downgrades. Escalation preserves required capabilities — a
failed code model is replaced by a stronger code model, not a
generic one.

### Capability-Aware Selection

Routing considers capabilities: `code`, `reasoning`, `math`,
`general`.

Fallback never selects a model that cannot perform the required
task. Capabilities are tracked per step, not inferred at retry
time.

### Task Decomposition

Break complex tasks into subtasks. Each step gets the right
model at the right price.

```python
planner = Planner()
plan = planner.plan(
    "Build a REST API for a todo app",
    budget=0.50,
)
# 4 steps, each with the cheapest viable model
```

### Cost Ledger

All LLM calls are recorded in a structured `CostLedger`,
including failed attempts and escalations. See exactly where
your money went.

### Multi-Provider Failover

Supports OpenRouter, OpenAI, Anthropic, and Google. Direct API
keys bypass OpenRouter automatically. The proxy shares a single
`httpx.AsyncClient` across all providers for connection pooling.

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
tokenwise route "Debug this segfault" \
  --strategy best_quality --budget 0.05

# Plan and execute a complex task
tokenwise plan "Build a REST API for a todo app" \
  --budget 0.50 --execute

# View spend history
tokenwise ledger
tokenwise ledger --summary

# Start the OpenAI-compatible proxy
tokenwise serve --port 8000

# List models and pricing
tokenwise models
```

**Python API:**

```python
from tokenwise import Router, Planner
from tokenwise.executor import Executor

# Route a single query
router = Router()
model = router.route(
    "Explain quantum computing",
    strategy="balanced",
    budget=0.10,
)
print(f"Use model: {model.id} "
      f"(${model.input_price}/M input tokens)")

# Plan a complex task
planner = Planner()
plan = planner.plan(
    task="Build a REST API for a todo app",
    budget=0.50,
)
print(f"Plan: {len(plan.steps)} steps, "
      f"estimated ${plan.total_estimated_cost:.4f}")

# Execute the plan — tracks spend, escalates on failure
executor = Executor()
result = executor.execute(plan)
print(f"Done! Cost: ${result.total_cost:.4f}, "
      f"success: {result.success}")
```

**OpenAI-compatible proxy:**

```bash
tokenwise serve --port 8000
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="unused",
)
response = client.chat.completions.create(
    model="auto",  # TokenWise picks the best model
    messages=[{"role": "user", "content": "Hello!"}],
)
```

> **Background reading:**
> [LLM Routers Are Not Enough](https://itsarbit.substack.com/p/llm-routers-are-not-enough)
> — the blog post that motivated TokenWise's design.

## Example

Plan a task, execute it, and inspect the cost ledger — all in
three commands:

```bash
# 1. Plan and execute a task ($0.05 budget)
tokenwise plan "Write a Python function to validate \
  email addresses, then write unit tests for it" \
  --budget 0.05 --execute

# 2. View your spend history
tokenwise ledger --summary
```

Example output:

```
Plan for: Write a Python function to validate email addresses...
Budget: $0.05
Estimated cost: $0.0023

┌─────────────────────────────────────────────────────────────┐
│ #  Description              Model               Est. Cost   │
│ 1  Write validation func    openai/gpt-4.1-mini  $0.0009    │
│ 2  Write unit tests         openai/gpt-4.1-mini  $0.0014    │
└─────────────────────────────────────────────────────────────┘

Status: Success
Total cost: $0.0019
Budget remaining: $0.0481
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
            ┌───────────────────┐    ┌──────────────────┐
 query ───▶ │  1. Detect        │───▶│  2. Route        │───▶ model
            │     Scenario      │    │     w/ Strategy   │
            │                   │    │                   │
            │  · capabilities   │    │  · filter budget  │
            │    (code, reason, │    │  · cheapest /     │
            │     math)         │    │    balanced /     │
            │  · complexity     │    │    best_quality   │
            │    (simple→hard)  │    │                   │
            └───────────────────┘    └──────────────────┘
```

**Router** separates *understanding what the query needs* from
*choosing how to spend*. Budget is a universal parameter — not
a strategy. By default, the router enforces the budget as a
hard ceiling: if no model fits, it raises an error instead of
silently exceeding the limit.

**Planner** decomposes a complex task into subtasks using a
cheap LLM, then assigns the optimal model to each step within
your budget. If the plan exceeds budget, it automatically
downgrades expensive steps.

**Executor** runs a plan step by step, tracks actual token
usage and cost via a `CostLedger`, and escalates to a stronger
model if a step fails. Escalation tries stronger tiers first
(flagship before mid) and filters by the step's required
capabilities.

### Observability

Every execution produces a structured trace. Inspect which
model was used, whether escalation occurred, and where each
dollar went:

```python
result = executor.execute(plan)

# Per-step: which model ran, whether it was escalated
for sr in result.step_results:
    print(f"Step {sr.step_id}: model={sr.model_id}, "
          f"cost=${sr.actual_cost:.4f}, "
          f"escalated={sr.escalated}")

# Cost ledger: every LLM call including failed attempts
for entry in result.ledger.entries:
    print(f"  {entry.reason}: {entry.model_id} "
          f"({entry.input_tokens}in/"
          f"{entry.output_tokens}out) "
          f"${entry.cost:.6f} "
          f"{'ok' if entry.success else 'FAIL'}")

# Aggregate
print(f"Total: ${result.total_cost:.4f}, "
      f"wasted: ${result.ledger.wasted_cost:.4f}, "
      f"remaining: ${result.budget_remaining:.4f}")
```

Example output when step 1 fails and escalates:

```
Step 1: model=openai/gpt-4.1, cost=$0.0052, escalated=True
  step 1 attempt 1: openai/gpt-4.1-mini (82in/0out) $0.000000 FAIL
  step 1 escalation attempt 1: openai/gpt-4.1 (82in/204out) $0.001800 ok
Total: $0.0052, wasted: $0.0000, remaining: $0.9948
```

## Routing Strategies

| Strategy | When to Use | How It Works |
|---|---|---|
| `cheapest` | Minimize cost | Picks the lowest-price capable model |
| `best_quality` | Maximize quality | Picks the best flagship-tier capable model |
| `balanced` | Default | Matches model tier to query complexity |

All strategies enforce the budget as a hard ceiling. Pass
`budget_strict=False` in the Python API to fall back to
best-effort behavior.

## Configuration

TokenWise reads configuration from environment variables and
an optional config file (`~/.config/tokenwise/config.yaml`).

| Variable | Required | Description | Default |
|---|---|---|---|
| `OPENROUTER_API_KEY` | **Yes** | OpenRouter API key | — |
| `OPENAI_API_KEY` | Optional | Direct OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Optional | Direct Anthropic API key | — |
| `GOOGLE_API_KEY` | Optional | Direct Google AI API key | — |
| `OPENROUTER_BASE_URL` | Optional | OpenRouter base URL | `https://openrouter.ai/api/v1` |
| `TOKENWISE_DEFAULT_STRATEGY` | Optional | Routing strategy | `balanced` |
| `TOKENWISE_DEFAULT_BUDGET` | Optional | Budget in USD | `1.00` |
| `TOKENWISE_PLANNER_MODEL` | Optional | Decomposition model | `openai/gpt-4.1-mini` |
| `TOKENWISE_PROXY_HOST` | Optional | Proxy bind host | `127.0.0.1` |
| `TOKENWISE_PROXY_PORT` | Optional | Proxy bind port | `8000` |
| `TOKENWISE_CACHE_TTL` | Optional | Registry cache TTL (s) | `3600` |
| `TOKENWISE_LEDGER_PATH` | Optional | Ledger JSONL path | `~/.config/tokenwise/ledger.jsonl` |
| `TOKENWISE_LOCAL_MODELS` | Optional | Local models YAML | — |

```yaml
# ~/.config/tokenwise/config.yaml
default_strategy: balanced
default_budget: 0.50
planner_model: openai/gpt-4.1-mini
```

## Architecture

```
src/tokenwise/
├── models.py        # Pydantic data models
├── config.py        # Settings from env vars and config file
├── registry.py      # ModelRegistry — fetches/caches models
├── router.py        # Two-stage pipeline: scenario → strategy
├── planner.py       # Decomposes tasks, assigns models
├── executor.py      # Runs plans, tracks spend, escalates
├── ledger_store.py  # Persistent JSONL spend history
├── cli.py           # Typer CLI
├── proxy.py         # FastAPI OpenAI-compatible proxy
├── providers/       # LLM provider adapters
│   ├── openrouter.py
│   ├── openai.py
│   ├── anthropic.py
│   ├── google.py
│   └── resolver.py  # Maps model IDs → provider instances
└── data/
    └── model_capabilities.json
```

## Philosophy

LLM systems should be treated like distributed systems.

That means clear failure semantics, explicit cost ceilings,
predictable escalation, and observability. TokenWise is
designed with that philosophy.

## Benchmarks

`benchmarks/pareto.py` runs 5 tasks across models at different
price tiers and reports cost vs success rate. Reproduce it:

```bash
uv sync --group benchmark  # installs matplotlib
uv run python benchmarks/pareto.py --models \
  openai/gpt-4.1-nano deepseek/deepseek-chat \
  openai/gpt-4.1-mini google/gemini-2.5-flash \
  openai/gpt-4.1 anthropic/claude-sonnet-4 \
  google/gemini-2.5-pro anthropic/claude-opus-4.6 \
  openai/o4-mini google/gemini-3.1-pro-preview
```

Sample results (February 2026, 5 simple tasks per model):

| Model | Tier | Success | Avg Cost / Task |
|---|---|---|---|
| openai/gpt-4.1-nano | budget | 100% | $0.000059 |
| deepseek/deepseek-chat | budget | 100% | $0.000174 |
| openai/gpt-4.1-mini | budget | 100% | $0.000238 |
| google/gemini-2.5-flash | budget | 100% | $0.000498 |
| openai/o4-mini | mid | 100% | $0.001137 |
| openai/gpt-4.1 | mid | 100% | $0.001201 |
| anthropic/claude-sonnet-4 | mid | 100% | $0.002681 |
| google/gemini-2.5-pro | mid | 100% | $0.002913 |
| google/gemini-3.1-pro-preview | flagship | 100% | $0.003490 |
| anthropic/claude-opus-4.6 | flagship | 100% | $0.005029 |

All models pass simple tasks — the value shows in cost: ~85x
spread between cheapest and most expensive. Harder tasks
(multi-step reasoning, long-context coding) will show quality
differentiation. Use `--csv` to save raw results.

## Known Limitations (v0.4)

All three v0.3 limitations have been resolved:

- ~~Planner cost not budgeted~~ — planner LLM cost is now
  tracked and deducted from budget (v0.4)
- ~~Linear execution~~ — independent steps now run in parallel
  via async DAG scheduling (v0.4)
- ~~No persistent spend tracking~~ — execution history is
  persisted to JSONL; see `tokenwise ledger` (v0.4)

**Note on `execute()` inside async contexts:** If you call
`executor.execute(plan)` from inside an existing event loop
(Jupyter, FastAPI, etc.), it automatically falls back to
sequential step execution. For concurrent DAG scheduling in
async code, use `await executor.aexecute(plan)` directly.

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
