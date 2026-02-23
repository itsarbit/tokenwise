# Changelog

All notable changes to TokenWise will be documented in this file.

## [0.5.0] - 2026-02-23

### Added
- **Routing traces** — `Router.route_with_trace()` returns a structured `RoutingTrace` with request_id, initial/final model and tier, escalation records, termination state, and budget usage; `PlanResult.routing_trace` carries the trace through execution
- **Termination states** — every routing/execution session ends with an explicit `TerminationState`: `completed`, `exhausted`, `failed`, `aborted`, or `no_go`
- **Risk gate** — lightweight, rule-based check that blocks dangerous queries (destructive operations like `DELETE ... database`, `rm -rf`, `DROP TABLE`) and ambiguous requests; opt-in via `TOKENWISE_RISK_GATE_ENABLED=true`; raises `RiskGateBlockedError` with attached trace
- **Monotonic escalation** — new `TOKENWISE_ESCALATION_POLICY=monotonic` mode restricts fallback to strictly stronger tiers (no same-tier retries); default remains `flexible`
- **Escalation records** — each fallback attempt is recorded as an `EscalationRecord` with from/to model, tier, reason code, and step ID; reason codes: `model_error`, `budget_exhausted`, `capability_mismatch`, `complexity_threshold`, `no_go`, `manual_override`
- **Trace level config** — `TOKENWISE_TRACE_LEVEL` (`basic` / `verbose`) controls trace verbosity
- **New exports** — `EscalationPolicy`, `EscalationReasonCode`, `RoutingTrace`, `TerminationState`, `TraceLevel` available from `tokenwise` package
- **Proxy trace** — auto-routed responses include `tokenwise_trace` dict with request_id, initial/final model, termination_state, and models_tried
- **Proxy risk gate** — returns HTTP 422 when risk gate blocks a request
- **CLI trace display** — `tokenwise route` shows request_id and termination state; `tokenwise plan --execute` shows routing trace summary with escalation table
- **Ledger trace** — `LedgerStore.save()` accepts optional `RoutingTrace` for persistence

### Changed
- **Executor** — escalation reason codes are now classified from failure details (HTTP status, error message) instead of hardcoded `MODEL_ERROR`
- **CLI route** — now uses `route_with_trace()` and displays trace metadata

## [0.4.5] - 2026-02-22

### Changed
- **Router benchmark** — aligned benchmark with Router strategies; tightened claims and added cost standard deviation
- **Deduplicated fallback loops** — refactored executor and proxy fallback logic to reduce code duplication
- **Anthropic streaming** — fixed double `[DONE]` yield in Anthropic provider adapter

## [0.4.4] - 2026-02-22

### Changed
- **README polish** — removed duplicate install section, modular CLI subsections, one-line feature bullets, compact Pareto plot with model labels, escalation shown in demo output, sharper positioning before comparison table
- **Strategy benchmark plot** — smaller figure (7x4), tighter Y-axis, model name labels, directional x-axis hints

## [0.4.3] - 2026-02-22

### Fixed
- **Total-cost-aware budget enforcement** — `_compute_max_tokens` now subtracts estimated input token cost before computing the output cap, ensuring total cost (input + output) stays within the budget ceiling; previously only output price was considered, allowing input cost to push total spend over budget

### Added
- **Configurable minimum output tokens** — `min_output_tokens` is now a setting (env: `TOKENWISE_MIN_OUTPUT_TOKENS`, config file, or `Executor(min_output_tokens=N)`); defaults to 100; set lower for workflows that need tiny outputs under small budgets
- **Pareto benchmark script** — `benchmarks/pareto.py` runs tasks across model tiers, saves results to CSV, and generates a cost-vs-quality scatter plot; supports `--dry-run`, `--csv`, and `--models` flags
- **Example section** — new README section with copy-paste CLI commands and example output
- **`benchmark` dependency group** — `matplotlib` available via `uv sync --group benchmark`

### Changed
- **README budget accuracy footnote** — "never silently overspend" claim now footnoted to clarify that input token estimation is heuristic (`chars/4` + 1.2x safety margin), not tokenizer-based; budget ceiling is enforced but small overruns are possible

### Fixed
- **mypy strict compliance** — resolved all 20 mypy errors: added `types-PyYAML` stubs, annotated bare `dict` generics across proxy/registry/config, fixed `no-any-return` in providers, corrected `LLMProvider` protocol's `astream_completion` signature to match async generator implementations

## [0.4.2] - 2026-02-21

### Fixed
- **Remove "creative" capability** — removed unused `"creative"` capability from decomposition prompt, curated model mapping, and test fixtures; the router had no creative patterns so it was silently restricting fallback to Anthropic-only models
- **Budget consistency in async executor** — `aexecute()` now passes `step.estimated_cost` (not `estimated_cost * 2`) as the budget for each step, matching the reservation amount
- **Multi-capability downgrade** — `_optimize_for_budget` now checks *all* required capabilities when downgrading models, not just the first; prevents a step needing `["code", "reasoning"]` from being downgraded to a code-only model
- **max_tokens budget guardrail** — `_execute_step` and `_aexecute_step` now compute and pass `max_tokens` based on remaining budget, preventing a single completion from overshooting; steps with budget below 100 output tokens are skipped gracefully

### Changed
- **README restructured** — moved blog post link to after Quick Start; softened absolute claims about cost ledger coverage
- **Registry** — `find_models()` and `cheapest()` accept a `capabilities` list parameter for multi-capability filtering

## [0.4.1] - 2026-02-21

### Fixed
- **Strict budget enforcement at call time** — sequential executor now checks step's estimated cost against remaining budget before making the LLM call; steps that would exceed budget are skipped
- **Ledger store summary** — removed redundant `load(limit=0)` call; `summary()` now uses `_load_all()` directly

## [0.4.0] - 2026-02-21

### Added
- **Planner cost budgeted** — the LLM call used for task decomposition now has its cost tracked and deducted from the user's budget; `Plan.planner_cost` exposes the cost; CLI displays it when > 0
- **Parallel step execution** — executor now runs independent steps concurrently via `asyncio.gather()`; steps declare `depends_on` indices in the decomposition prompt; the executor builds a DAG and launches ready steps in parallel; `execute()` delegates to `asyncio.run(aexecute())` transparently
- **Persistent spend tracking** — new `LedgerStore` class persists execution history to a JSONL file (`~/.config/tokenwise/ledger.jsonl` by default); `tokenwise plan --execute` auto-saves; new `tokenwise ledger` CLI command shows history and `--summary` aggregates
- **`TOKENWISE_LEDGER_PATH`** — new env var / config field to customize the ledger file path

### Changed
- **Decomposition prompt** — now asks the LLM to produce `depends_on` (0-indexed step indices) for each step; planner parses these and falls back to sequential chain if missing
- **Executor** — `execute()` now dispatches to async DAG-based scheduling; falls back to sequential when already inside an async event loop
- **Parallel budget enforcement** — reservation-based allocation prevents parallel steps from collectively overshooting the budget; each step reserves its estimated cost before launch
- **Async escalation cost tracking** — failed attempt costs are now correctly included in `total_cost` when escalation succeeds (previously lost in async path)
- **Deadlock detection** — async executor detects dependency cycles and reports stuck steps as failures instead of silently exiting
- **`_optimize_for_budget`** — now uses `step.required_capabilities` instead of the current model's first capability when downgrading models
- **JSON parsing** — more tolerant code fence stripping (no newline required after opening fence); bracket-extraction fallback when fences don't match

## [0.3.0] - 2026-02-20

### Added
- **CostLedger** — structured cost tracking across attempts and escalations; `PlanResult.ledger` records every LLM call with reason, model, tokens, cost, and success/failure
- **Strict budget ceiling** — `router.route()` now raises `ValueError` when no model fits the budget (controlled via `budget_strict` parameter; default `True`)
- **Decomposition visibility** — `Plan` now exposes `decomposition_source` ("llm" or "fallback") and `decomposition_error` so callers know when task decomposition fell back
- **TTL on failed models** — proxy's failed-model set now expires entries after 5 minutes (configurable) and caps at 50 entries, preventing unbounded growth
- **Shared HTTP client** — providers reuse the proxy's `httpx.AsyncClient` instead of creating a new client per request, reducing connection overhead
- **Ledger table in CLI** — `tokenwise plan --execute` now prints a Rich cost breakdown table with wasted-cost summary

### Changed
- **Escalation ordering** — executor and proxy now escalate to stronger tiers first (FLAGSHIP → MID) instead of trying budget tier first; fallback candidates are filtered by capability
- **Retryable error codes** — removed HTTP 400 from retryable/fallback codes (400 is a request schema error, not a model outage)
- **Router** — budget is strict by default; planner uses `budget_strict=False` for its own internal routing

### Fixed
- Proxy `failed_models` set no longer grows without bound across the server lifetime

## [0.2.0] - 2026-02-20

### Added
- **SSE streaming** — proxy now supports `stream: true` with Server-Sent Events pass-through from upstream providers
- **Direct provider support** — use OpenAI, Anthropic, and Google APIs directly by setting `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`; requests bypass OpenRouter when a direct key is available
- **Provider abstraction** — `LLMProvider` protocol with adapters for OpenRouter, OpenAI, Anthropic Messages API, and Google Gemini API
- **Curated capability detection** — prefix-matched model family mapping replaces substring heuristics; vision detected from API metadata; user overrides supported via config

### Changed
- **Router** — refactored to a two-stage pipeline: scenario detection (capabilities + complexity) → strategy routing; all strategies are now scenario-aware by default
- **Router** — budget is now a universal parameter on all strategies instead of a separate `budget_constrained` strategy
- **Proxy** — `_forward_to_upstream` and `_handle_streaming` now use provider abstraction instead of raw httpx calls
- **Planner & Executor** — use `ProviderResolver` instead of direct httpx calls with manual header construction

### Removed
- `RoutingStrategy.BUDGET_CONSTRAINED` — use any strategy with `budget=` parameter instead
- **Registry** — capability and tier inference rewritten to use curated JSON mapping with keyword fallback

### Notes
- OpenRouter API key is still required for model discovery (registry); direct provider keys are optional for LLM calls
- Steps execute sequentially (parallel execution planned)

## [0.1.0] - 2026-02-20

### Added
- **Router** — select the best model for a query using 4 strategies: `cheapest`, `balanced`, `best_quality`, `budget_constrained`
- **Planner** — LLM-powered task decomposition into steps with automatic model assignment
- **Executor** — sequential step execution with cost tracking, budget enforcement, and automatic escalation on failure
- **Proxy** — OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/models`) with `model: "auto"` routing
- **Model Registry** — loads model metadata and pricing from OpenRouter API with local YAML fallback and TTL caching
- **Capability Detection** — heuristic keyword matching for code, reasoning, math, vision, and creative tasks
- **CLI** — `tokenwise serve`, `tokenwise route`, `tokenwise plan` commands via Typer
- **Resilient fallback** — executor and proxy retry with alternative models on 402/403/404 errors, tracking failed models across steps
- **Example scripts** — 4 runnable demos covering routing, plan & execute, budget comparison, and the proxy
