# OpenClaw + TokenWise Integration Guide

Use TokenWise as a drop-in proxy between [OpenClaw](https://github.com/open-claw/open-claw)
and your LLM providers to get budget enforcement, intelligent model routing, and
multi-provider failover — with zero code changes to OpenClaw.

## What TokenWise Adds

- **Cost control** — set per-request or global budget ceilings so you never overspend.
- **Intelligent routing** — automatically pick the cheapest, best, or balanced model
  for each query based on task complexity.
- **Multi-provider failover** — if a model returns 402/403/404 or a transient 5xx,
  TokenWise retries on a fallback model automatically.
- **Streaming support** — full SSE streaming, forwarded transparently.

## Quick Start

### 1. Install TokenWise and set your API key

```bash
pip install tokenwise-llm          # or: uv pip install tokenwise-llm
export OPENROUTER_API_KEY="sk-or-..."
```

### 2. Start the proxy

```bash
tokenwise serve --port 8000
```

The proxy is now listening at `http://127.0.0.1:8000` with these endpoints:

| Endpoint                      | Method | Description                     |
|-------------------------------|--------|---------------------------------|
| `/v1/chat/completions`        | POST   | Chat completions (OpenAI-compatible) |
| `/v1/models`                  | GET    | List available models           |
| `/health`                     | GET    | Health check (returns version)  |

### 3. Configure OpenClaw

In your `openclaw.json`, point the API provider at the TokenWise proxy:

```json
{
  "apiProvider": "openai",
  "openAiBaseUrl": "http://localhost:8000/v1",
  "openAiApiKey": "unused",
  "openAiModelId": "auto"
}
```

> **Note:** The `openAiApiKey` value can be any non-empty string — TokenWise
> authenticates via the `OPENROUTER_API_KEY` environment variable, not the
> per-request API key header.

That's it. OpenClaw will now route all LLM traffic through TokenWise.

## Model Options

Set `openAiModelId` in your OpenClaw config to control routing behavior:

### `auto` (recommended)

Balanced routing — TokenWise analyses the query and picks an appropriate model
weighing quality against cost. This is the default strategy.

```json
"openAiModelId": "auto"
```

### `tokenwise/<anything>`

Any model ID prefixed with `tokenwise/` also triggers intelligent routing.
The actual strategy and budget are controlled via the `tokenwise` request body
field (see [Budget Control](#budget-control) below) or environment variables:

```bash
export TOKENWISE_DEFAULT_STRATEGY="cheapest"   # cheapest | balanced | best_quality
export TOKENWISE_DEFAULT_BUDGET="0.50"          # USD per request
```

### Direct model IDs

Bypass routing entirely by specifying a model ID from your provider:

```json
"openAiModelId": "openai/gpt-4.1"
```

Run `tokenwise models` to see all available model IDs and their pricing.

## Budget Control

TokenWise accepts an optional `tokenwise` field in the chat completion request
body for per-request routing overrides:

```json
{
  "model": "auto",
  "messages": [{"role": "user", "content": "..."}],
  "tokenwise": {
    "strategy": "cheapest",
    "budget": 0.01
  }
}
```

Available strategies: `cheapest`, `balanced`, `best_quality`.

Whether this works with OpenClaw depends on whether it passes extra fields through
to the upstream API. If OpenClaw strips unknown fields, you can still control
defaults globally via environment variables:

```bash
export TOKENWISE_DEFAULT_STRATEGY="balanced"
export TOKENWISE_DEFAULT_BUDGET="1.00"
```

## Streaming

TokenWise supports full SSE streaming. When OpenClaw sends `"stream": true`,
the proxy forwards Server-Sent Events from the upstream provider, so token-by-token
output works exactly as expected.

## Production Tips

**Bind to all interfaces** for remote or container access:

```bash
tokenwise serve --host 0.0.0.0 --port 8000
```

**Health check** — use `/health` for readiness probes:

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "0.5.0"}
```

**Auto-reload** during development:

```bash
tokenwise serve --port 8000 --reload
```

**View available models and pricing:**

```bash
tokenwise models
tokenwise models --tier budget
tokenwise models --capability code
```
