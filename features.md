# Features

The feature inventory of the Agentic AI Gateway: what exists, what is broken or
incomplete, and what is proposed. It is kept up to date as work lands.

**Status legend**

| Mark | Meaning |
|---|---|
| ✅ | Complete, and covered by an automated test (unit, integration, or E2E) |
| 🟡 | Implemented, but not yet covered end-to-end |
| 🐞 | Implemented but broken (bug found during verification) |
| ⛔ | Incomplete: declared, documented, or half-wired but not functional |
| 💡 | Proposed: not started |

**Bugs found and fixed during verification** (each has a regression test; details in the tables below):
budgets never charged · streamed and failed requests never logged · cache hits billed with $0 recorded savings ·
double-counted metrics · `/v1/rag/query` always 500 · any virtual key or viewer could register a `stdio` MCP server
(host command execution) · HTTP MCP servers at a path (`/mcp`) unreachable · deployment health check always 404 ·
failed-over requests attributed to the dead deployment · Anthropic streaming broken for the TypeScript SDK ·
`MAX_RETRIES` off by one · logging/tracing settings ignored · system info reporting RAG/MCP as off ·
console: RAG page crash, every key shown "Disabled", Models columns blank, playground bypassing fallback and
breaking on stream errors, never sending the message just typed, no user management · MCP server header values (often credentials) returned in clear by the API · token counting stalled requests while downloading its tokenizer · streamed upstream errors crashed while being mapped (raw 500,
no context-length fallback).

Test suites referenced below:

- **unit**: `tests/unit/` (`uv run pytest tests/unit`)
- **int**: `tests/integration/`, which boots the real app against real Postgres and Redis Stack (`make test-integration`)
- **ui-unit**: `ui/src/**/*.test.ts(x)`, Vitest + Testing Library (`make ui-test`)
- **e2e**: `e2e/` Cucumber + Playwright scenarios against the dockerised stack (`make e2e`)

**Coverage gates.** unit + int together must cover 100% of `app/` (lines and branches);
`make coverage` and the CI `coverage` job fail otherwise. ui-unit must cover 100% of the console's
logic layer (API client, SSE parsing, playground inspector, formatters, auth, shared components).

---

## 1. Unified API (dialects)

| Feature | Status | Tests |
|---|---|---|
| `POST /v1/chat/completions` (OpenAI), unary | ✅ | unit, int, e2e (official `openai` SDK) |
| `POST /v1/chat/completions` SSE streaming, `[DONE]`, `stream_options.include_usage` | ✅ | unit, int, e2e |
| `POST /v1/completions` legacy wrapper (unary and stream) | ✅ | unit, int, e2e (`legacy_completions.feature`) |
| `POST /v1/embeddings` | ✅ | int, e2e |
| `GET /v1/models`, `GET /v1/models/{model}` (disabled-only models are 404) | ✅ | unit, int |
| `POST /v1/messages` (Anthropic), unary + named SSE events | ✅ | unit, int, e2e (official `@anthropic-ai/sdk`) |
| Anthropic SSE payloads carry `"type"` (required by the TypeScript SDK) | ✅ | **fixed**: streaming broke Node clients; int, e2e |
| `POST /v1/messages/count_tokens` | ✅ | unit, int, e2e. Note: does not authenticate the caller |
| Tool / function calling normalised across dialects | ✅ | unit, e2e (SDK tool call) |
| Model aliases (`gpt-4o` → `test-model`), allowlist applies through aliases | ✅ | int |
| Structured OpenAI / Anthropic error envelopes | ✅ | unit, int |
| Per-request overrides (`no_cache`, `cache_ttl`, `routing_strategy`, `tags`, via body / `metadata` / `aigw`) | ✅ | unit, e2e (`no_cache`, `routing_strategy`) |

## 2. Providers

| Feature | Status | Tests |
|---|---|---|
| OpenAI adapter (chat, stream, embeddings, health) | ✅ | unit (respx), e2e against the fake upstream |
| Streamed upstream error statuses are mapped with their detail (all adapters) | ✅ | **fixed**: body was unread, so mapping raised `ResponseNotRead`; unit |
| Anthropic adapter (chat, stream, tools, tool results, images) | ✅ | unit (respx); e2e against the fake upstream's native Messages API, incl. the Anthropic SDK on a native deployment (`native_providers`) |
| Google Gemini adapter (chat, stream, tools, tool results, embeddings, health) | ✅ | unit (respx); e2e against the fake upstream's native `generateContent` / SSE / `embedContent` (`native_providers`) |
| Ollama adapter (chat, NDJSON stream, tools, tool results, embed, tags) | ✅ | unit (respx); e2e against the fake upstream's native `/api/chat` and `/api/embed` (`native_providers`) |
| YAML model catalogue, `${VAR:-default}` expansion, multiple deployments per model | ✅ | unit, int, e2e |
| Runtime catalogue reload (`POST /admin/api/config/reload`) | ✅ | unit, int, e2e |

## 3. Routing and resilience

| Feature | Status | Tests |
|---|---|---|
| Strategies: priority, least-cost, lowest-latency (EWMA), weighted, conditional | ✅ | unit, e2e (`routing.feature`) |
| Per-request strategy override | ✅ | unit, e2e |
| Retries with full-jitter backoff honouring `Retry-After` | ✅ | unit, int |
| Cross-deployment fallback chain (`MAX_FALLBACKS`) | ✅ | unit, int, e2e (dead primary in `models.eval.yaml`; `eval-multi` fails over from Anthropic to Gemini) |
| Request log, cost, and console name the deployment that *served* a failed-over request | ✅ | **fixed**: logs blamed the dead primary; int, e2e |
| Streaming fallback before first chunk | ✅ | unit, int |
| Circuit breaker (closed / open / half-open) | ✅ | unit |
| Model allow / block lists on virtual keys | ✅ | unit, int (`test_policies_http.py`) |
| `MAX_RETRIES` means *retries*, not total attempts | ✅ | **fixed** (was off by one); int |
| Conditional routing on request `tags`: matching deployments preferred (cheapest first), other matches fall back first | ✅ | **new**; unit, int (`test_routing_cache_http.py`), e2e |
| Deployment `rpm_limit` / `tpm_limit`: saturated deployments are skipped (fallback); all saturated → 429 | ✅ | **new**; int (`test_limits_http.py`) |

## 4. Semantic cache

| Feature | Status | Tests |
|---|---|---|
| Redis Stack HNSW vector cache, lazy index creation / rebuild | ✅ | unit, int, e2e |
| Namespace isolation (model, tenant, system prompt, params, tools) | ✅ | unit |
| Eligibility rules (no tools, temperature cap, finish = stop), `no_cache` opt-out | ✅ | unit, e2e |
| Admin: stats, entries, invalidate | ✅ | unit, int, e2e (invalidation forces a fresh answer) |
| Cache hit served over HTTP against real Redis Stack | ✅ | int (`test_accounting_http.py`) |
| Cache hits are not billed; `cost_saved_usd` recorded | ✅ | **fixed**; int |
| `X-Gateway-Cache-Similarity` response header on hits (unary and streamed) | ✅ | **new**; int, e2e |
| `estimated_latency_saved_ms` in cache stats (original latency minus lookup time) | ✅ | **new**; int, e2e |

## 5. RAG

| Feature | Status | Tests |
|---|---|---|
| Collections CRUD (`/v1/rag/collections`) | ✅ | int (`test_rag_api.py`) |
| Document ingest (JSON / multipart), list, delete; multipart validation (file field, file type, metadata JSON) | ✅ | unit, int |
| Markdown-aware chunking, idempotent ingest, rollback on vector failure | ✅ | unit, int |
| Chunk inspector (list / get with embedding preview) | ✅ | unit, int |
| `POST /v1/rag/search` with MMR diversity and filters | ✅ | unit, int |
| **Hybrid retrieval** (`search_mode: hybrid`): BM25 keyword search fused with vector search by reciprocal rank; `BM25STD` scorer with legacy fallback | ✅ | **new**: unit (`test_rag_hybrid.py`), int, e2e; recall@5 0.85 → 0.98 on `bench/datasets/rag_eval.json` |
| Optional **LLM reranking** (`rerank_model` per request or collection), falls back to retrieval order on any failure | ✅ | **new**: unit, int (`test_rag_rerank.py`) |
| **Metadata filters**: collections declare `filterable_fields`, indexed as tags; undeclared fields refused | ✅ | **new**: int (`test_rag_filters.py`) |
| `POST /v1/rag/query` (retrieve + chat), runs as the calling key | ✅ | **fixed** (was always 500); int |
| RAG in ordinary chat requests (`aigw.rag`, both dialects, streaming; sources in `aigw.sources` + `X-Gateway-RAG-Sources`) | ✅ | **new**: `RagStage` after auth, before guardrails/cache; int (`test_agentic_http.py`), e2e (`agentic.feature`) |
| Retrieved text screened by input guardrails; cache scoped per collection | ✅ | int |
| One RediSearch index per collection (`aigw:rag:<collection>:idx`) | ✅ | dead `RAG_INDEX_NAME` setting removed; int |
| **Tenancy**: collections owned by the creating key's team (or the key); others get 404; shared collections read-only to applications; names unique per owner | ✅ | **new**: migration 0002; int (`test_rag_tenancy.py`), e2e (`rag_enterprise.feature`) |
| **Recoverability**: a vanished index is recreated and backfilled; `GET …/index` status; `POST …/reindex` rebuilds vectors from Postgres | ✅ | **new**: int (`test_rag_recovery.py`, incl. FLUSHALL), e2e |
| **File ingestion**: .txt, .md, .html, .pdf, .docx (`rag-docs` extra, in the Docker image) | ✅ | **new**: unit (`test_rag_extract.py`), int |
| **Ingestion at scale**: size limit (413), background ingestion for large documents (202, poll), interrupted ingestions failed on restart, re-upload replaces by source | ✅ | **new**: int (`test_rag_ingestion.py`) |
| RAG metrics: retrievals, latency, empty results, ingestions, reranks | ✅ | **new**: int (`test_rag_mcp_metrics.py`) |

## 6. MCP and tools

| Feature | Status | Tests |
|---|---|---|
| MCP client: streamable HTTP (JSON + SSE) and stdio transports | ✅ | unit, int, e2e against `scripts/fake_mcp_server.py` |
| MCP server CRUD, refresh, tool discovery, namespaced tool names | ✅ | unit, int (`test_mcp_api.py`) |
| `POST /v1/mcp/tools/call` with JSON-schema argument validation | ✅ | unit, int |
| `GET /v1/mcp/health` | ✅ | int |
| Only admins may register MCP servers (stdio runs host commands) | ✅ | **fixed** (security); int |
| HTTP MCP servers mounted at a path (e.g. `/mcp`) are reachable | ✅ | **fixed**: client posted to `/mcp/`, so such servers were always unhealthy; int |
| `MCP_TIMEOUT_SECONDS` / `MCP_TOOL_CACHE_TTL_SECONDS` settings | ✅ | **fixed**: declared and documented |
| MCP clients / stdio processes closed on shutdown | ✅ | **fixed**: `McpRegistry.close()` in `GatewayState.shutdown` |
| Server-side agent loop (`aigw.mcp`): MCP tools offered and executed until the model answers; client tools returned; `max_iterations` | ✅ | **new**: `AgenticExecutor`; int, e2e |
| Agent tool calls logged (`stage_timings.tool_calls`, `tool_calls_count`), usage summed across hops, auth/budget once per request | ✅ | int |
| **Secrets at rest**: server `env` and `headers` values Fernet-encrypted (`SECRETS_ENCRYPTION_KEY`, rotation); header values redacted in API responses | ✅ | **new** (headers were returned in clear); unit, int (`test_mcp_secrets.py`) |
| **Access control**: per-key `allowed_mcp_servers` / `allowed_tools` (wildcards) on listing, agent loop, and direct calls | ✅ | **new**: migration 0004; int (`test_mcp_governance.py`), e2e |
| **Guardrails on tool traffic** (`apply_to_tools`): input rules on arguments, output rules on results; result size cap | ✅ | **new**: int, e2e |
| **Tool-call audit log** (`/admin/api/tool-calls`, console MCP page): who, what, outcome, duration, guardrail; arguments hashed | ✅ | **new**: int, e2e |
| **Resilience**: per-server circuit breaker; expired sessions re-initialised and replayed; crashed stdio servers restarted with a fresh handshake; stderr kept for diagnostics; per-server timeouts; discovery retry | ✅ | **new**: unit (`test_mcp_resilience.py`); e2e fake server expires sessions every 5 s |
| Background health checks (`MCP_HEALTH_INTERVAL_SECONDS`) | ✅ | **new**: unit |
| MCP metrics: tool calls by server, tool, and status; latency; breaker state | ✅ | **new**: int |

## 7. Guardrails

| Feature | Status | Tests |
|---|---|---|
| Rule types: regex, denylist, length, topic, PII (Luhn, SSN, email…) | ✅ | unit, int |
| Actions: block, redact, flag; input and output stages | ✅ | unit, int, e2e |
| Violations persisted and listed (`/admin/api/guardrails/violations`); policies listed | ✅ | unit, int, e2e |
| Block / redact observed over HTTP (input PII, prompt injection, output secrets, per-key policy) | ✅ | int (`test_policies_http.py`) |
| Output guardrails on streamed responses: redaction across chunk boundaries (holdback window), block ends the stream | ✅ | **new** `StreamRedactor`; unit, int, e2e |
| LLM-judge guardrail (`type: llm_judge`, threshold, `on_error` allow/block); judge calls bypass the pipeline and are not billed | ✅ | **new**; unit, int (`test_llm_judge_http.py`), e2e (`judged` policy in `config/guardrails.eval.yaml`) |

## 8. Auth, virtual keys, quotas

| Feature | Status | Tests |
|---|---|---|
| Master key; virtual keys (`sk-aigw-…`, SHA-256 hashed, Redis snapshot cache) | ✅ | unit, int |
| Key create (shown once), list, get, update, regenerate, delete, disable | ✅ | unit, int (`test_admin_flows.py`) |
| Expired / inactive key rejection | ✅ | unit, int |
| RPM rate limit (Redis sliding window, fail-open) → 429 + `Retry-After` | ✅ | unit, int |
| Budgets on keys and teams (charged for unary and streamed requests) | ✅ | **fixed** (spend was never recorded); int, e2e (key and team budgets) |
| Teams CRUD and team usage | ✅ | int |
| Key `tpm_limit`, `max_parallel_requests`, `allowed_routes` | ✅ | **new**; int, e2e |
| `Retry-After` reflects real wait time (rounded up, never 0) | ✅ | **fixed**; int |

## 9. Accounting and observability

| Feature | Status | Tests |
|---|---|---|
| Price table (YAML + per-deployment pricing), token counting (tiktoken) | ✅ | unit, int |
| Request log + hourly usage rollups | ✅ | unit, int, e2e |
| Body logging with redaction (`LOG_REQUEST_BODIES`) | ✅ | unit |
| Prometheus `/metrics` (`aigw_*`) | ✅ | int |
| Streamed requests logged, costed, and metered | ✅ | **fixed**; int |
| Failed requests logged and metered (authenticated callers only) | ✅ | **fixed**; int |
| Metrics recorded exactly once (cache lookups, retries, fallbacks, TTFT) | ✅ | **fixed**; int |
| Structured logging honours `LOG_LEVEL` / `LOG_FORMAT` | ✅ | **fixed**: configured in `create_app` |
| OpenTelemetry tracing (`TRACING_ENABLED`, `OTLP_ENDPOINT`) | ✅ | **fixed**; int |
| `aigw_active_requests` gauge, rate-limit-hit counter (`scope` = key_rpm / key_tpm / key_parallel / deployment) | ✅ | **fixed**; int |

## 10. Admin API (`/admin/api`)

| Feature | Status | Tests |
|---|---|---|
| Console login (argon2 + JWT), `me`, logout | ✅ | unit, int |
| Change password (admin + self-service); wrong current password refused | ✅ | unit, int, e2e |
| Users CRUD, last-admin protection, no self-demotion or self-deletion | ✅ | unit, int, e2e |
| Role enforcement (admin vs viewer) on admin, RAG, and MCP routes | ✅ | unit, int |
| Dashboard summary and timeseries; malformed windows and `order_by` rejected (422) | ✅ | unit, int, e2e |
| Logs list / filter / detail / reveal / CSV export | ✅ | unit, int |
| Usage by dimension, usage costs | ✅ | unit, int |
| Deployments, models, provider status, health check | ✅ | **fixed**: health check 404'd for every generated id (`provider/model`); int |
| System info reports RAG / MCP enabled | ✅ | **fixed**; int |
| Refresh tokens with rotation (`/auth/refresh`, single use); console renews expired sessions silently | ✅ | **new**; int (`test_sessions_http.py`), e2e |
| Server-side logout and revocation (Redis denylist); password, role, or status change signs the user out everywhere | ✅ | **new**; int, e2e |

## 11. Console UI (`/ui`)

All console scenarios run in Chromium via Playwright (`e2e/features/ui`).

| Feature | Status | Tests |
|---|---|---|
| Login, route guards, redirect back after login, sign out | ✅ | ui-unit (`AuthProvider`, `RouteGuards`), e2e (`auth.feature`) |
| Role-based navigation (viewers see read-only sections only) | ✅ | **fixed**: nav hid Cache/RAG/MCP from viewers although routes allowed them; e2e |
| Every page renders without an error | ✅ | e2e (`navigation.feature`) |
| Dashboard | ✅ | e2e |
| Models & routing: priority, weight, tags, true capabilities, chains in priority order | ✅ | **fixed**: columns were hard-coded "—" and chains sorted by id; e2e |
| Deployment health check button | ✅ | **fixed**: always 404 (unencoded `provider/model#n` ids); e2e |
| Logs (filters, search, detail drawer, routing decision) | ✅ | e2e (`logs.feature`) |
| Usage | ✅ | e2e (lists a model that served traffic) |
| Guardrails (policies, violations) | ✅ | e2e (viewer sees the `default` policy) |
| Cache (stats, entries, invalidate) | ✅ | e2e (reports the cache available) |
| RAG collections (create, paste text, retrieval test) | ✅ | **fixed**: page crashed as soon as a collection existed (TDZ); e2e |
| MCP servers (add, discover tools, health) | ✅ | e2e |
| Virtual keys (create shows secret once, status, disable, delete) | ✅ | **fixed**: every key showed "Disabled" (`is_active` vs `enabled`); e2e |
| Playground (streaming chat through normal routing); SSE parsing and inspector | ✅ | **fixed**: pinned a deployment (bypassing fallback), stream errors broke the connection, model picker had no label, the message just typed was never sent (the request used the conversation from before it); ui-unit (`sse`, `inspector`), int, e2e |
| Settings: user management (add, role, activate/deactivate, delete) | ✅ | **new**: page claimed the API was missing; e2e |
| Settings: provider credential status, system info, subsystems | ✅ | **new** (was "not exposed"); e2e |
| Logout calls the server | ✅ | **fixed** |
| Dead code (`PlaceholderPage`, unused hooks) removed | ✅ | |
| Lint rule against render-time use-before-define | ✅ | `@typescript-eslint/no-use-before-define` |
| Playground RAG mode, compare mode, replay from log | 🟡 | not covered end-to-end |
| Keys: regenerate and edit limits from the UI | 🟡 | API covered (int, e2e); UI not |
| Teams page (create, edit, delete, usage; read-only for viewers) | ✅ | **new**; e2e (`teams.feature`) |
| Logs CSV export button (uses the current filters) | ✅ | **new**; e2e |

## 12. CLI, packaging, deployment

| Feature | Status | Tests |
|---|---|---|
| `aigateway serve / init-db / create-admin / create-key / routes` | ✅ | int (`TestCli`) |
| Docker image (UI build stage + runtime) | ✅ | CI build; the E2E suite runs against it |
| docker compose stack (Postgres, Redis Stack, Prometheus, Grafana, Ollama) | 🟡 | manual |
| Helm chart | ✅ | `helm lint` + `helm template` (CI `helm` job) |
| Grafana dashboard + Prometheus scrape config | 🟡 | manual |
| Consistent default port: 4000 locally (`make dev`, `aigateway serve`, Vite proxy, smoke.sh), 8000 in the container | ✅ | **fixed** |
| `ENVIRONMENT` documented correctly (`dev`/`staging`/`prod`) | ✅ | **fixed** docs |
| Alembic migrations: baseline `0001`, `aigateway migrate` / `db-stamp`, Helm init container, E2E stack boots from migrations | ✅ | **new**; int (`test_migrations.py`: no model/migration drift, round trip, stamp) |
| Python version pinned to 3.12 for local dev | ✅ | `.python-version` |

## 13. Quality and CI

| Feature | Status | Tests |
|---|---|---|
| Unit suite | ✅ | `make test-unit` |
| Integration suite on real Postgres + Redis Stack | ✅ | `make test-integration` |
| Cucumber + Playwright E2E suite (API via official SDKs + console UI) | ✅ | 129 scenarios / 663 steps, `make e2e` |
| SDK compatibility script (`scripts/agent_compat.py`) | ✅ | 10/10 against fake upstream |
| Evaluation harness (`scripts/evaluate.py`) | ✅ | 100% success, failover and routing verified |
| Deterministic fakes: multi-provider upstream (OpenAI, Anthropic, Gemini, Ollama wire formats; bag-of-words embeddings), MCP server (HTTP + stdio) | ✅ | `scripts/fake_upstream.py`, `scripts/fake_mcp_server.py` |
| CI: ruff on `tests/`, UI lint, integration job with Postgres + Redis Stack services, E2E job with report artifact | ✅ | `.github/workflows/ci.yml` |
| Coverage gate: 100% line and branch on `app/`, 100% on the console's logic layer | ✅ | `make coverage`, `make ui-test`; CI `coverage` job |
| **Benchmarks** in one command, saved as JSON, compared against a committed baseline with tolerances | ✅ | **new**: `make bench` / `bench-check` / `bench-baseline`, `scripts/bench.py`, `scripts/bench_compare.py`, `bench/thresholds.yaml`; CI `bench` job |
| **Retrieval-quality benchmark** (recall@k, MRR, nDCG, by question kind; vector vs hybrid) | ✅ | **new**: `scripts/rag_eval.py`, `bench/datasets/rag_eval.json` |
| Agent-loop benchmark (tool-loop success, tool calls per request, latency per hop) | ✅ | **new**: `scripts/evaluate.py` `agentic_tool_calls` section |
| Load test in CI with success and p95 gates | ✅ | **new**: CI `bench` job |
| Repo-wide `ruff format`, enforced in CI | ✅ | **new**: `make fmt-check` |
| Token counting never blocks on tokenizer downloads (bounded background load; files baked into the image) | ✅ | **fixed**: requests stalled on first use with a slow network; unit |

## Proposed next steps

The ten items from the previous list are all done: RAG in chat, the agent
loop, key and deployment limits, streaming output guardrails, Alembic
migrations, refresh tokens and revocation, the Teams page and CSV export, tag
routing, cache similarity and latency saved, and the LLM judge.

Done since the last list: guardrails on tool traffic, non-text and background
RAG ingestion, the load test in CI, and repo-wide formatting.

What is still open, ordered by value; none are started:

1. **Streaming agent hops**: stream intermediate model turns instead of only the final answer.
2. **Console: tool-call and RAG-index views** beyond the audit table, e.g. reindex and filter fields from the browser.
3. **Load test against real providers** on a schedule, tracked over time.
4. **OCR** for scanned PDFs (today they are refused with a clear message).
5. **Provider adapters against live APIs**: an opt-in `live`-marked suite for Anthropic, Gemini, and Ollama. Today they run end to end against the fake upstream, which follows each vendor's documented wire format but cannot catch the vendor changing it.
