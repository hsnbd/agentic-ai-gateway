# Agentic AI Gateway

A unified middleware between agentic AI applications and large language model
providers — one API, many providers, with routing, caching, guardrails, and
full observability.

## Why

Agentic applications depend on multiple LLM providers, each with different
APIs, authentication, capabilities, pricing, and failure behaviour. Agent
loops (RAG, MCP, tool calls) multiply request volume, which magnifies both
cost and reliability risk.

This gateway puts one consistent, provider-independent layer in front of all
of them.

## Features

- **Unified API** — OpenAI-compatible and Anthropic-compatible endpoints, so
  existing SDKs and coding agents work unchanged
- **Multi-provider** — OpenAI, Anthropic, Google Gemini, and Ollama, written
  natively against each vendor's API
- **Reliability** — automatic retries with backoff, cross-provider fallback
  chains, and circuit breakers
- **Smart routing** — least-cost, lowest-latency, weighted, priority, and
  conditional strategies with health-aware load balancing
- **Semantic caching** — embedding-similarity cache on Redis vector search
- **RAG** — document ingestion, chunking, embedding, storage, and retrieval
- **MCP** — Model Context Protocol servers proxied as gateway-callable tools
- **Tool calling** — normalized function calling across every provider
- **Guardrails** — regex, denylist, and PII policies on input and output
- **Cost control** — virtual API keys with budgets, rate limits, and model
  allowlists
- **Observability** — Prometheus metrics, OpenTelemetry traces, structured logs
- **Console UI** — React + MUI admin console served at `/ui`

## Quick start

```bash
# Install
uv venv --python 3.12
uv pip install -e ".[dev]"

# Configure
cp .env.example .env   # then add your provider keys

# Run the dependencies and the gateway
docker compose -f deploy/docker/compose.yaml up -d
uv run uvicorn app.main:app --reload --port 4000
```

Then point any OpenAI client at it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-your-virtual-key")
client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

The console is at <http://localhost:4000/ui>.

## Command line

The `aigateway` entry point covers the operations that must work before the
console is reachable:

```bash
uv run aigateway init-db                                  # create the schema
uv run aigateway create-admin --email you@example.com --password '...'
uv run aigateway create-key --name smoke-test --budget 5  # prints the key once
uv run aigateway routes                                   # list mounted routes
uv run aigateway serve --port 4000
```

## Evaluating a deployment

`scripts/evaluate.py` drives a running gateway over HTTP and reports the
metrics this project is graded on — success rate, latency percentiles, time to
first token, cache hit ratio, failover, cost per successful request, routing
behaviour, and guardrail effectiveness:

```bash
uv run python scripts/evaluate.py \
    --base-url http://localhost:4000 \
    --api-key sk-your-virtual-key \
    --model gpt-4o-mini --requests 50 --concurrency 5
```

It exits non-zero below `--min-success-rate`, so it doubles as a CI gate.

## Testing

Three suites, from fastest to most complete. [features.md](./features.md) maps
every feature to the tests that cover it.

| Suite | Command | Needs |
|---|---|---|
| Unit (`tests/unit`) | `make test-unit` | nothing |
| Integration (`tests/integration`): the real app on real Postgres and Redis Stack, fake LLM providers | `make test-integration` | Docker |
| End-to-end (`e2e/`): Cucumber scenarios through the official OpenAI/Anthropic SDKs and a real browser driving the console, against the dockerised stack | `make e2e-install` once, then `make e2e` | Docker, Node 22 |

`make test-integration` starts throwaway datastores from
`tests/integration/docker-compose.test.yaml` (ports 55432 and 56379); point the
suite elsewhere with `AIGW_TEST_DATABASE_URL` and `AIGW_TEST_REDIS_URL`.
`make e2e` builds the gateway image and starts it with the fake upstream
(`scripts/fake_upstream.py`) and fake MCP server (`scripts/fake_mcp_server.py`)
on port 18000; reports land in `e2e/reports/`. Stop the stacks with
`make test-down` and `make e2e-down`.

## Documentation

| Document | What it covers |
|---|---|
| [Architecture](./docs/architecture.md) | Request lifecycle, pipeline stage order and why, routing and circuit-breaker semantics, streaming rules, degraded modes |
| [Configuration](./docs/configuration.md) | Every environment variable, the three YAML files, and per-request overrides |
| [Adding a provider](./docs/providers.md) | The `Provider` contract, capability declaration, pricing units, and the conformance suite |
| [Agent & SDK setup](./docs/agents.md) | Pointing the OpenAI and Anthropic SDKs, opencode, Claude Code, and Cursor at the gateway |
| [Console guide](./docs/console.md) | Operating the gateway from the browser |

## License

Apache-2.0
