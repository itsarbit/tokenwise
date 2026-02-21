# Changelog

All notable changes to TokenWise will be documented in this file.

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
- **Resilient fallback** — executor and proxy retry with alternative models on 400/402/403/404 errors, tracking failed models across steps
- **Example scripts** — 4 runnable demos covering routing, plan & execute, budget comparison, and the proxy
