# Roadmap

## v0.5.0 — Routing Transparency & Control (shipped)

- Structured `RoutingTrace` on every request (request_id, tiers tried, escalation records, termination state, budget used/remaining)
- Explicit termination states: `completed`, `exhausted`, `failed`, `aborted`, `no_go`
- Trace verbosity config (`basic` / `verbose`)
- Monotonic escalation as a named, tested policy
- Risk gate — blocks destructive and ambiguous queries (opt-in)

## Future

- Replay-based determinism diagnostics
- Cost variance analytics
- Load simulation
- Adaptive boundary tuning
- Pluggable tokenizer-based input estimation
