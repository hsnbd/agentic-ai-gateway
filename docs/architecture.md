# Architecture

## What this is

The Agentic AI Gateway sits between AI applications and LLM providers. Applications
speak one API; the gateway handles provider differences, failures, cost, caching,
safety, and observability.

The design goal is that an application should never need to know which provider
served a request, and should never break because one provider had a bad day.

## Request lifecycle

Every request — whatever dialect it arrives in — follows the same path:

```
HTTP request (OpenAI or Anthropic dialect)
      │
      ▼
 dialect adapter ──────────► canonical ChatRequest
      │
      ▼
┌─────────────────────────────────────────────┐
│ PRE-STAGES (any may short-circuit)          │
│   1. auth        key, limits, budget        │
│   2. rag         retrieve context (aigw.rag)│
│   3. guardrails  inspect prompt + context   │
│   4. cache       semantic lookup            │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│ AGENTIC EXECUTOR (aigw.mcp only)            │
│   offer MCP tools, run the ones called,     │
│   repeat until the model answers            │
│ ┌─────────────────────────────────────────┐ │
│ │ RESILIENT EXECUTOR (every model call)   │ │
│ │   router picks a deployment             │ │
│ │   skip deployments over rpm/tpm limits  │ │
│ │   retry on the same deployment          │ │
│ │   fall back to the next on exhaustion   │ │
│ │   circuit breaker tracks health         │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│ POST-STAGES (each sees the final response)  │
│   1. guardrails_out  inspect the output     │
│   2. cache_write     store for next time    │
│   3. observability   metrics, cost, logs    │
└─────────────────────────────────────────────┘
      │
      ▼
 dialect adapter ──────────► HTTP response
```

The order is declared exactly once, in [`app/core/builder.py`](../app/core/builder.py).
That file is the composition root: if you want to know what the gateway actually
does to a request, read it top to bottom.

### Why this order

**Auth first.** Unauthenticated work should cost nothing. Nothing that touches a
provider, an embedding model, or the cache runs before the caller is known.

**RAG after auth, before guardrails and cache.** Retrieval costs an embedding
call, so it waits for a known caller. Running it before input guardrails means
retrieved text is screened like the prompt (a poisoned document cannot smuggle
instructions past policy), and running it before the cache means the cache keys
on the grounded request, scoped to the collection.

**Guardrails before cache.** A prompt that policy forbids should never even be
looked up. Putting the cache first would mean a blocked prompt still performs a
vector search.

**The agent loop lives in the executor.** Tool hops re-enter routing, retries,
and fallback, but not the pre- and post-stages: a request is authenticated,
budget-checked, guarded, and logged once, with usage summed across every hop.

**Observability last.** The request log must record the response the client
actually received — after output guardrails have redacted it — along with the
timings of every stage that ran before it.

## The canonical IR

Everything in the middle of the pipeline speaks one language:
[`app/core/schemas.py`](../app/core/schemas.py). Dialects translate inbound,
adapters translate outbound, and no provider-specific shape is allowed to leak
between them.

This is what makes adding a provider cheap: a new adapter only has to map its own
API to and from the IR. It never has to know that OpenAI and Anthropic disagree
about where the system prompt goes.

Some fields on `ChatRequest` are **gateway-only** (`no_cache`, `cache_ttl`,
`fallbacks`, `routing_strategy`, `guardrail_policy`, `tags`). They control gateway
behaviour and are never forwarded upstream.

## Routing and resilience

### Strategies

Five strategies live in [`app/routing/strategies.py`](../app/routing/strategies.py):

| Strategy | Picks | Use when |
|---|---|---|
| `priority` (default) | Lowest `priority` number | You have a clear first choice |
| `least-cost` | Cheapest per estimated token | Bulk or background work |
| `lowest-latency` | Best observed EWMA latency | Interactive traffic |
| `weighted` | Weighted random sample | A/B tests, gradual rollout |
| `conditional` | Based on request shape | Long prompts, tool use, tags |

Every strategy implements both `select()` (which one now) and `order()` (the whole
fallback chain), so the fallback order respects the same policy as the first pick.

`priority: 1` means **first choice** — the lowest number wins, as in DNS SRV records
and nginx upstreams.

### Circuit breaker

The breaker in [`app/routing/breaker.py`](../app/routing/breaker.py) is deliberately
**process-local**. It reacts in microseconds with no network round trip, and
because every replica sees the same upstream failures, replicas converge on the
same view of provider health without coordinating.

```
CLOSED ──N consecutive failures──► OPEN
   ▲                                 │
   │                            cooldown elapses
   │                                 ▼
   └──probe succeeds──────────  HALF_OPEN ──probe fails──► OPEN
```

`HALF_OPEN` admits exactly one probe. A failed probe re-opens immediately rather
than waiting to accumulate the threshold again.

**The router never hard-fails on a total outage.** If every deployment is
circuit-broken, it retries them anyway and records why. A healed outage is better
discovered by a real request than by a permanent self-inflicted refusal.

### Retries and fallbacks

Two different failure responses, chosen by the error taxonomy in
[`app/core/errors.py`](../app/core/errors.py):

- `RETRYABLE_CODES` — try the **same** deployment again (rate limits, timeouts,
  5xx, overload).
- `FALLBACKABLE_CODES` — try a **different** deployment (the retryable set, plus
  context-length exceeded, content filtered, and no healthy deployment).

Backoff uses **full jitter** (`random.uniform(0, delay)`), which avoids replicas
synchronising into a retry storm after a shared outage. A server-supplied
`Retry-After` overrides the computed delay but is still clamped to `max_backoff`.

### Streaming fails over only before the first chunk

Once bytes have reached the client the response is committed. Switching providers
mid-stream would splice two different generations together, producing text no model
ever wrote. After the first token, errors surface to the client instead. There is a
test asserting the backup provider is never called in that case.

### Output guardrails on streams

Streamed text passes through a redactor that holds back the most recent
`GUARDRAILS_STREAM_HOLDBACK_CHARS` characters and never releases part of a
match, so redaction works across chunk boundaries. A `block` rule ends the
stream with an in-band error. The output stage still sees the raw text once the
stream completes, so violations are recorded exactly as for unary requests.

## Caching

Semantic only — there is no exact-match tier. A request is embedded, searched
against a Redis Stack vector index, and served if the nearest neighbour is above
the similarity threshold.

That power comes with a correctness risk: two prompts can be similar but not
equivalent. The mitigations are structural:

- a high default threshold (0.95 cosine);
- **namespacing** by model, key/team, system-prompt hash, temperature bucket,
  response format, and tools hash — so entries can only match within a context
  where they are actually interchangeable;
- never caching tool calls, or responses above `cache_max_temperature`;
- `X-Gateway-Cache` response headers exposing hit/miss and the similarity score.

The cache embedder defaults to a local Ollama model, so the per-request embedding
needed to *use* the cache is close to free.

## Degraded modes

The gateway distinguishes subsystems it can live without from ones it cannot.

| Subsystem | Missing or broken | Behaviour |
|---|---|---|
| Auth | fatal | Refuses to start |
| Guardrails | degraded | Logs, serves without policy checks |
| Cache | degraded | Logs, serves every request upstream |
| Cost accounting | degraded | Metrics still recorded, cost rows lost |

Serving unauthenticated traffic silently would be worse than not booting, so auth
is the one hard requirement. Everything else prefers availability.

## Data model

Postgres holds durable state ([`app/db/models.py`](../app/db/models.py)):

- `Team`, `VirtualKey`, `AdminUser` — who may call what, and their budgets
- `RequestLog` — one row per request; the backbone of the console log explorer,
  carrying stage timings, routing reason, cache similarity, and guardrail results
- `UsageRollup` — pre-aggregated hourly, so dashboards never scan raw logs
- `RagCollection` / `RagDocument` / `RagChunk` — the RAG corpus
- `McpServer` — registered MCP servers
- `GuardrailViolation` — the violation feed

Redis holds ephemeral state: the semantic cache vectors, sliding-window rate limit
counters, and RAG embeddings.

## Extending the gateway

**A new provider** means one file in `app/providers/`, a `Provider` subclass, and an
entry in `config/models.yaml`. The conformance suite in
`tests/unit/test_provider_conformance.py` runs against every registered adapter and
will tell you if yours diverges from the contract.

**A new guardrail** means a rule class in `app/guardrails/` and a policy entry in
`config/guardrails.yaml`.

**A new routing strategy** means a `RoutingStrategy` subclass implementing `select()`
and `order()`, registered in `get_strategy()`.
