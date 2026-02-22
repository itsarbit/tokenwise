# Roadmap

## v0.5.0 — Routing Transparency (planned)

- Structured `routing_trace` on every request (request_id, tiers tried, escalations, termination state, budget used/remaining)
- Explicit termination states: `completed`, `exhausted`, `failed`, `blocked`
- Trace verbosity config (`basic` / `verbose`)
- Monotonic escalation as a named, tested policy

## Future

- Risk gates (block unsafe escalation before it happens)
- Replay-based determinism diagnostics
- Cost variance analytics
- Load simulation
- Adaptive boundary tuning
