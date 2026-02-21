# Changelog

All notable changes to TokenWise will be documented in this file.

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
