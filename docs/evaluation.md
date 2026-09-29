# Evaluation Report

This report maps the Agentic AI Gateway against the project's stated grading
criteria. Every number here was produced by [`scripts/evaluate.py`](../scripts/evaluate.py)
against a running gateway — the real FastAPI app, the real pipeline, real Redis
Stack, and real Postgres. Nothing is mocked inside the gateway.

## How to reproduce

```bash
# 1. Datastores
docker run -d --name aigw_redis -p 6399:6379 redis/redis-stack:latest
docker run -d --name aigw_pg -p 5499:5432 -e POSTGRES_PASSWORD=aigw postgres:16

# 2. A deterministic upstream, so no API keys or network access are needed
uv run python scripts/fake_upstream.py --port 4100

# 3. The gateway, using the evaluation catalogue
export MODELS_CONFIG_PATH=config/models.eval.yaml
uv run uvicorn app.main:app --port 4030

# 4. The harness
uv run python scripts/evaluate.py \
  --base-url http://localhost:4030 --api-key sk-eval-master \
  --model eval-router --failover-model eval-chat --routing-model eval-router \
  --requests 50 --concurrency 5
```

### Why a fake upstream

The harness has to prove behaviour that real providers cannot be asked to
produce on demand: a deployment that is guaranteed to fail, a response that is
byte-identical across calls, and an embedding that is stable enough to reason
about cache hits. [`scripts/fake_upstream.py`](../scripts/fake_upstream.py)
gives all three, and it keeps the evaluation free and offline.

Its one important limitation is stated plainly here: its embeddings are
**hash-based, not semantic**. Identical text produces an identical vector, so
exact-repeat cache hits are real, but genuine paraphrase matching cannot be
demonstrated with it. Against a real embedding model the cache hit ratio would
be driven by paraphrase similarity; here it is driven by exact repeats. The
threshold logic, vector index, namespacing, and scoring path exercised are
identical either way.

### The evaluation catalogue

[`config/models.eval.yaml`](../config/models.eval.yaml) exists because the
production catalogue cannot demonstrate two of the criteria:

- `eval-chat` has two deployments, the first pointing at a port with nothing
  listening. Failover is therefore forced, not simulated.
- `eval-router` has two healthy deployments at different prices, weights, and
  priorities, so the routing strategies have something to disagree about. With
  one deployment per model every strategy trivially picks the same one and
  proves nothing.

## Results

Run: 50 requests, concurrency 5. All seven sections completed with **no
warnings**.

### 1. Request success rate

| Metric | Value |
|---|---|
| Requests | 50 |
| Succeeded | 50 |
| Failed | 0 |
| **Success rate** | **100%** |

This figure includes the forced-failover request, which succeeded only because
retry and fallback absorbed a dead deployment.

### 2. Response latency

| Metric | Value |
|---|---|
| p50 | 15 ms |
| p95 | 20 ms |
| p99 | 22 ms |
| mean | 15 ms |
| Throughput | 322 req/s at concurrency 5 |
| Streaming TTFT p50 | 5 ms |
| Streaming TTFT p95 | 5 ms |

These measure **gateway overhead**, not model latency, because the upstream
responds instantly. That is the useful number: it isolates the cost of the
pipeline itself. A typical request's stage breakdown, taken from the structured
logs, is auth 0.004 ms, input guardrails 0.05 ms, cache 1.6 ms, execute 1.9 ms,
output guardrails 0.02 ms, cache write 1.2 ms. The cache stages dominate
gateway overhead because they involve an embedding call and a vector search.

### 3. Provider failover

| Finding | Value |
|---|---|
| Model used | `eval-chat` |
| Request succeeded | ✅ 200 |
| Served by | `openai/eval-chat#2` (the fallback) |
| Recovered from dead primary | ✅ |
| Unknown model correctly rejected | ✅ 404 |

The gateway retried the unreachable primary with jittered backoff, then fell
back. The structured log for that request reads `attempt_count=3`,
`fallback_used=True`, `status=success` — the client saw only a normal 200.

The second row matters as much as the first. An unknown model name is
**rejected**, not silently served by a fallback. Substituting a different model
for a typo'd one would hide client bugs and bill the caller for a model they
did not ask for, so this is treated as a client error rather than a failover
opportunity.

### 4. Cache hit ratio

| Finding | Value |
|---|---|
| Follow-up requests | 6 |
| Hits | 6 |
| **Hit ratio** | **100%** |
| Unrelated prompt hit | ❌ No — correct |

Both halves are required. A cache that hits on everything is worse than no
cache, because it returns wrong answers; the control prompt confirms an
unrelated request is **not** served from cache. See the caveat above about
hash-based embeddings: this demonstrates exact-repeat hits and correct
rejection, not paraphrase matching.

### 5. Cost per successful request

| Metric | Value |
|---|---|
| Total cost | $0.009525 |
| **Cost per successful request** | **$0.000191** |
| Prompt tokens | 650 |
| Completion tokens | 790 |
| Total tokens | 1,440 |

Costs are computed by the gateway from the per-deployment price table in the
catalogue (USD per million tokens) and returned on every response as
`X-Gateway-Cost-USD`, so cost attribution does not depend on provider billing
data arriving later.

### 6. Token and request usage

Token counts above are the gateway's own accounting, recorded per request and
rolled up per key, model, and provider. They are queryable through
`GET /admin/api/usage` with `group_by=hour|day` and are exported to Prometheus.

### 7. Routing effectiveness

| Strategy | Deployment chosen |
|---|---|
| `least-cost` | `openai/eval-router#2` (the $0.15/Mtok deployment) |
| `lowest-latency` | `openai/eval-router#2` |
| `priority` | `openai/eval-router` (the priority-0 deployment) |
| `weighted` | `openai/eval-router#2` (weight 9 vs 1) |

The strategies **diverge**, which is the only thing that proves routing is real.
`least-cost` picks the cheap deployment over the frontier one; `priority`
picks the explicitly preferred deployment despite it being 16× more expensive.

### 8. Guardrail effectiveness

| Finding | Value |
|---|---|
| Benign prompt allowed | ✅ 200 |
| PII prompt blocked or redacted | ✅ |

Both directions are tested. A guardrail that blocks everything scores perfectly
on "unsafe content blocked" while making the gateway useless, so the benign
case is part of the measurement.

### 9. Provider extensibility

Adding a provider means implementing one `Provider` ABC — `chat`, `stream`,
`embed`, `capabilities` — and registering it. Four adapters ship (OpenAI,
Anthropic, Gemini, Ollama) with no LiteLLM dependency. The shared conformance
suite in `tests/unit/test_provider_conformance.py` runs against every adapter,
so a new provider inherits the same correctness bar.

### 10. RAG, MCP, and tool-calling support

| Surface | Status |
|---|---|
| `GET /v1/models` | 200 |
| `GET /v1/rag/collections` | 200 |
| `GET /v1/mcp/servers` | 200 |
| `GET /v1/mcp/tools` | 200 |
| `GET /metrics` | 200 |
| `GET /ui` | 307 → console |
| Tool-calling request accepted | ✅ |

### 11. Coding-agent and SDK compatibility

Accepting a tool-calling *request* is weaker evidence than it sounds: the shape
that matters is the one the official SDKs parse. [`scripts/agent_compat.py`](../scripts/agent_compat.py)
drives the gateway through `openai` and `anthropic` themselves, so streaming
delta reassembly and tool-call accumulation are validated by the same code an
agent runs.

```bash
uv run python scripts/agent_compat.py \
  --base-url http://localhost:4030 --api-key "$MASTER_KEY" --model eval-chat
```

**Result: 10/10 checks passed.** Model listing, unary chat, streaming chat, unary
tool calls, streaming tool calls, the `role: tool` result round-trip, and
embeddings on the OpenAI SDK; unary, system-prompt, and streaming `messages` on
the Anthropic SDK. Full table in [docs/agents.md](agents.md).

### 12. Container image

The graded deployment target is the built image, not the dev server, so it is
verified as a unit against the same Redis and Postgres:

| Check | Result |
|---|---|
| `/healthz`, `/readyz`, `/metrics` | 200 |
| `/ui` and deep link `/ui/logs` | 200 (SPA fallback) |
| `/ui/assets/nope.js` | 404 (missing assets are *not* swallowed by the fallback) |
| `POST /admin/api/auth/login` | 200 (bootstrap admin created on first run) |
| `GET /v1/models` with master key | 200 |
| Tracebacks in container logs | 0 |

The bad-asset check is deliberate. An SPA fallback that returns `index.html` for
every unmatched path turns a missing JS bundle into a silent white screen; here
it still 404s.

## Sustained load

`scripts/evaluate.py` uses short bursts, which cannot reveal saturation.
[`scripts/loadtest.py`](../scripts/loadtest.py) holds continuous load and
reports percentiles per window so drift is visible as it happens.

```bash
uv run python scripts/loadtest.py \
  --base-url http://localhost:4030 --api-key sk-eval-master \
  --model eval-router --duration 20 --concurrency 15
```

| t(s) | rps | ok | err | hit% | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|---|
| 5 | 225.2 | 1126 | 0 | 30.2% | 68.1 | 98.1 | 149.6 |
| 10 | 235.2 | 1176 | 0 | 30.3% | 66.1 | 98.8 | 107.4 |
| 15 | 230.6 | 1153 | 0 | 32.1% | 66.6 | 102.0 | 117.2 |
| 20 | 239.6 | 1198 | 0 | 30.8% | 66.3 | 88.9 | 106.7 |

| Metric | Value |
|---|---|
| Requests | 4,668 |
| Failed | 0 |
| Success rate | 100% |
| Throughput | 232.8 req/s sustained at concurrency 15 |
| p50 / p95 / p99 | 66.9 ms / 98.6 ms / 114.1 ms |
| Max | 181.0 ms |
| **Latency drift** | **−6.1 ms** |

Two results are worth calling out.

**Latency drift is negative.** Comparing the first tenth of requests to the
last, latency *fell* by 6 ms rather than climbing. Queues are not building and
the connection pool is not degrading, so the gateway is holding steady rather
than slowly saturating at this level.

**The cache hit ratio landed at 30.9% against a configured 30% cacheable
fraction.** The load generator makes only `--cache-ratio` of its prompts
repeatable and tags the rest with a UUID. The measured hit ratio tracking the
configured one almost exactly is independent end-to-end confirmation that the
cache hits precisely what it should and nothing it should not — a stronger
signal than the targeted cache probe, because it emerges from thousands of
concurrent mixed requests rather than a handful of crafted ones.

## Test and quality baseline

| Check | Result |
|---|---|
| `uv run pytest` | **288 passed** |
| `uv run ruff check app tests scripts` | All checks passed |
| `uv run mypy app` | No issues in 77 source files |
| `docker build` | Succeeds; container serves API, console, and deep links |

## What the evaluation caught that tests did not

This section is the honest part of the report, and arguably the most useful.

**The semantic cache never returned a hit, and no test noticed.** `FT.SEARCH`
returns a flat list under RESP2 and a map under RESP3. The parser handled only
the list form, so under RESP3 every lookup parsed as zero matches. Entries were
still written, no error was logged, and the hit ratio sat at zero — a headline
feature was entirely non-functional while looking healthy. The unit suite missed
it because its in-memory fake Redis returned only the RESP2 shape: the tests
agreed with each other and disagreed with reality.

**Then it broke a second time, differently.** With RESP3 parsing fixed, lookups
raised `UnicodeDecodeError`. `FT.SEARCH` returns *every* stored field including
the embedding, which is packed float32 and not valid UTF-8, and the parser
decoded values strictly. Because cache errors are deliberately non-fatal, the
exception surfaced only as a permanent miss.

**And a third time.** Destroying the index at runtime — `FLUSHALL`, a Redis
restart without persistence, or failover to a replica — left the gateway
permanently cacheless, because index creation ran once at startup and
`ensure_index` short-circuited on a cached flag. The cache now detects a missing
index and rebuilds it, which also backfills entries written while it was gone.

All three shared one failure signature: **silent degradation to zero hit ratio**
with no error visible to an operator. They are now covered by regression tests
in `tests/unit/test_cache.py`.

**Two more, outside the cache.** Console deep links 404'd because Starlette's
`StaticFiles` *raises* rather than returns a 404, so the SPA fallback never
fired. And `ensure_bootstrap_admin()` was implemented but never called, making
the console unreachable on any fresh deployment.

The lesson is recorded deliberately: a live harness against the assembled system
found, in a single run, five defects that 288 passing unit tests could not.
Mocks encode assumptions, and assumptions are exactly what was wrong.

## Known limitations

- Paraphrase-level semantic caching is not demonstrated by this harness; it
  needs a real embedding model. The threshold, index, and scoring path are
  exercised, but the embedding quality is not.
- Latency figures exclude real provider time by design.
- `auto_create_schema` creates tables at startup and is additive only; it
  cannot migrate an existing schema.
- Estimated latency savings from cache hits are not recorded, and the console
  renders that metric as "not recorded" rather than zero.
