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

## 30-Second Demo

```bash
pip install tokenwise-llm

# Route a query to the best model within budget
tokenwise route "Debug this segfault" --strategy best_quality --budget 0.05

# Decompose a task, execute it, track spend
tokenwise plan "Write a Python function to validate email addresses, \
  then write unit tests for it" --budget 0.05 --execute
```

Example output:

```
Plan: 4 steps | Budget: $0.05 | Estimated: $0.0002

Status: Success | Total cost: $0.0007 | Budget remaining: $0.0493
```

If a step fails, TokenWise automatically escalates to a
stronger model and retries within budget.

## Why

Most LLM frameworks optimize for capability. TokenWise
optimizes for cost governance. You declare a budget per
request — TokenWise enforces it, selects the best model
within that constraint, and escalates only when needed.
No hidden traffic allocation, no implicit ceilings. All
decisions are explicit and per-request.

## Quick Start

### Set your API key

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

> TokenWise uses [OpenRouter](https://openrouter.ai) as the
> default gateway. Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or
> `GOOGLE_API_KEY` to bypass OpenRouter for those providers.

### Route a query

```bash
tokenwise route "Write a haiku about Python"
```

### Route with budget ceiling

```bash
tokenwise route "Debug this segfault" --strategy best_quality --budget 0.05
```

### Plan and execute

```bash
tokenwise plan "Build a REST API" --budget 0.50 --execute
```

### Inspect spend

```bash
tokenwise ledger --summary
```

### Routing strategies

| Strategy | When to Use | How It Works |
|---|---|---|
| `cheapest` | Minimize cost | Lowest-price capable model |
| `best_quality` | Maximize quality | Best flagship-tier capable model |
| `balanced` | Default | Matches model tier to query complexity |

Budget is a universal parameter on all strategies. Pass
`budget_strict=False` to fall back to best-effort.

### Python API

```python
from tokenwise import Router, Planner, Executor

# Route a single query
router = Router()
model = router.route("Explain quantum computing", strategy="balanced", budget=0.10)
print(f"{model.id} (${model.input_price}/M input)")

# Plan and execute a complex task
planner = Planner()
plan = planner.plan(task="Build a REST API for a todo app", budget=0.50)

executor = Executor()
result = executor.execute(plan)
print(f"Cost: ${result.total_cost:.4f}, success: {result.success}")
```

### OpenAI-compatible proxy

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

## Core Features

- **Budget-aware routing** — cost ceilings enforced via `max_tokens` caps with conservative estimation ([details](#budget-semantics)).
- **Tiered escalation** — budget, mid, flagship; escalates upward on failure, never downward.
- **Capability-aware fallback** — routes and fallbacks filtered by `code`, `reasoning`, `math`, or `general`.
- **Task decomposition** — LLM-powered planning with per-step model assignment and async DAG scheduling.
- **Cost ledger** — structured per-call accounting including failures and retries, persisted to JSONL.
- **Multi-provider failover** — OpenRouter, OpenAI, Anthropic, and Google with connection pooling.

## Benchmark: Cost–Quality Frontier

![Routing Strategies](assets/pareto.png)

*X-axis: average cost per task (USD, log scale). Y-axis: success rate (%).
The star marks the TokenWise escalation strategy, which uses `Router.route()`
with escalating strategies (cheapest → balanced → best_quality) and retries
on validation failure. Baselines use a single fixed model for all tasks.*

On this 20-task benchmark set, TokenWise Escalation is the only
strategy that reaches 100% success, at about 5x lower average cost
per task than Flagship Only.

| Strategy | Success | Avg Cost / Task | Models |
|---|---|---|---|
| Budget Only | 85% | $0.000177 | gpt-4.1-nano |
| Mid Only | 90% | $0.003842 | gpt-4.1 |
| Flagship Only | 95% | $0.009492 | claude-sonnet-4 |
| **TokenWise Escalation** | **100%** | **$0.001985** | Router-selected |

In multi-step workflows (10–50 steps), the per-step savings compound:
a 5x reduction per step means 5x for the entire workflow.  With more
steps there are more opportunities for escalation to save on easy
sub-tasks while still escalating when needed.

**Success metric:** 15 of 20 tasks have dedicated validators (reasoning
correctness, code structure, substantiveness checks). The remaining 5
simple tasks use a length-check fallback (>20 chars). Validators are
defined in `benchmarks/strategy_pareto.py`.

**Budget:** $0.03/task soft target — used for Router model filtering
but not enforced as a hard ceiling.

### How to reproduce

```bash
# Requires an OpenRouter API key (or direct provider keys)
export OPENROUTER_API_KEY="sk-or-..."

uv sync --group benchmark
uv run python benchmarks/strategy_pareto.py
```

Generated artifacts:

| File | Contents |
|---|---|
| `assets/pareto.png` | Cost–quality scatter plot |
| `benchmarks/strategy_results.csv` | Per-task results: strategy, task, category, success, cost, model, escalated, latency, budget, budget_violation |

Use `--dry-run` to preview the plan without making API calls.

### Limitations

- Small task set (20 tasks across 4 categories); not a comprehensive LLM benchmark.
- Validators are heuristic — they check for known correct patterns, not full semantic correctness.
- Model pricing and availability change over time; results are provider-dependent.
- Single-run results; no confidence intervals. The run date is printed in the script output.
- Escalation in the benchmark uses a validation-retry loop; in production, the Executor escalates on execution failure.

## Comparison

Most routing tools optimize per-request model choice. TokenWise
treats routing as a workflow-level control system.

High-level comparison (as of February 2026). Corrections welcome
via [issues](https://github.com/itsarbit/tokenwise/issues).

| Feature | TokenWise | [RouteLLM](https://github.com/lm-sys/RouteLLM) | [LiteLLM](https://github.com/BerriAI/litellm) | [Not Diamond](https://notdiamond.ai) | [Martian](https://withmartian.com) | [Portkey](https://portkey.ai) | [OpenRouter](https://openrouter.ai) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Task decomposition | **Yes** | - | - | - | - | - | - |
| Strict budget ceiling | **Yes** | - | Yes | - | Per-request | Yes | Yes |
| Tier-based escalation | **Yes** | - | Yes | - | - | Yes | - |
| Capability-aware fallback | **Yes** | - | - | Partial | Yes | Partial | Partial |
| Cost ledger | **Yes** | - | Yes | - | - | Yes | Dashboard |
| OpenAI-compatible proxy | **Yes** | Yes | Yes | Yes | Yes | Yes | Yes |
| CLI | **Yes** | - | Yes | - | - | - | - |
| Self-hosted / open source | **Yes** | Yes | Yes | - | - | Gateway only | - |

## How It Works

### Architecture

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

### Router pipeline

The router uses a two-stage pipeline:
**detect** (capabilities + complexity) then
**route** (filter by budget, apply strategy: `cheapest` /
`balanced` / `best_quality`).

### Planner and Executor

**Planner** decomposes a task into subtasks using a cheap LLM,
assigns the optimal model to each step within budget, and
auto-downgrades expensive steps if over budget.

**Executor** runs the plan via async DAG scheduling, tracks
actual cost via `CostLedger`, and escalates to stronger models
on failure (flagship before mid, filtered by capability).

If `executor.execute(plan)` is called inside an existing event
loop (Jupyter, FastAPI), it falls back to sequential execution.
Use `await executor.aexecute(plan)` directly for concurrent DAG
scheduling in async code.

### Observability

Every execution produces a structured trace:

```python
result = executor.execute(plan)

for sr in result.step_results:
    print(f"Step {sr.step_id}: model={sr.model_id}, "
          f"cost=${sr.actual_cost:.4f}, escalated={sr.escalated}")

for entry in result.ledger.entries:
    print(f"  {entry.reason}: {entry.model_id} "
          f"({entry.input_tokens}in/{entry.output_tokens}out) "
          f"${entry.cost:.6f} {'ok' if entry.success else 'FAIL'}")

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

## Budget Semantics

TokenWise enforces budget ceilings by capping `max_tokens`
before each LLM call. Input token counts are estimated using a
`chars / 4` heuristic with a 1.2x safety margin — not a
tokenizer. The budget ceiling is real and enforced, but small
overruns are possible when the heuristic underestimates input
tokens. A future release will support pluggable tokenizer-based
estimation for stricter guarantees.

## Known Limitations (v0.4)

All three v0.3 limitations have been resolved:

- ~~Planner cost not budgeted~~ — tracked and deducted (v0.4)
- ~~Linear execution~~ — parallel DAG scheduling (v0.4)
- ~~No persistent spend tracking~~ — JSONL ledger (v0.4)

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
| `TOKENWISE_MIN_OUTPUT_TOKENS` | Optional | Min output tokens per step | `100` |
| `TOKENWISE_LOCAL_MODELS` | Optional | Local models YAML | — |

```yaml
# ~/.config/tokenwise/config.yaml
default_strategy: balanced
default_budget: 0.50
planner_model: openai/gpt-4.1-mini
```

## Development

```bash
git clone https://github.com/itsarbit/tokenwise.git
cd tokenwise
uv sync
uv run pytest
uv run ruff check src/ tests/
uv run mypy src/
```

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
predictable escalation, and observability. TokenWise is designed
with that philosophy.

> **Background reading:**
> [LLM Routers Are Not Enough](https://itsarbit.substack.com/p/llm-routers-are-not-enough)
> — the blog post that motivated TokenWise's design.

## License

MIT
