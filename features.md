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
breaking on stream errors, no user management.

Test suites referenced below:

- **unit**: `tests/unit/` (`uv run pytest tests/unit`)
- **int**: `tests/integration/`, which boots the real app against real Postgres and Redis Stack (`make test-integration`)
- **e2e**: `e2e/` Cucumber + Playwright scenarios against the dockerised stack (`make e2e`)

---

## 1. Unified API (dialects)

| Feature | Status | Tests |
|---|---|---|
| `POST /v1/chat/completions` (OpenAI), unary | ✅ | unit, int, e2e (official `openai` SDK) |
| `POST /v1/chat/completions` SSE streaming, `[DONE]`, `stream_options.include_usage` | ✅ | unit, int, e2e |
| `POST /v1/completions` legacy wrapper (unary and stream) | ✅ | unit, int |
| `POST /v1/embeddings` | ✅ | int, e2e |
| `GET /v1/models`, `GET /v1/models/{model}` | ✅ | unit, int |
| `POST /v1/messages` (Anthropic), unary + named SSE events | ✅ | unit, int, e2e (official `@anthropic-ai/sdk`) |
| Anthropic SSE payloads carry `"type"` (required by the TypeScript SDK) | ✅ | **fixed**: streaming broke Node clients; int, e2e |
| `POST /v1/messages/count_tokens` | ✅ | unit, int |
| Tool / function calling normalised across dialects | ✅ | unit, e2e (SDK tool call) |
| Model aliases (`gpt-4o` → `test-model`), allowlist applies through aliases | ✅ | int |
| Structured OpenAI / Anthropic error envelopes | ✅ | unit, int |
| Per-request overrides (`no_cache`, `cache_ttl`, `routing_strategy`, `tags`, via body / `metadata` / `aigw`) | ✅ | unit, e2e (`no_cache`, `routing_strategy`) |

## 2. Providers

| Feature | Status | Tests |
|---|---|---|
| OpenAI adapter (chat, stream, embeddings, health) | ✅ | unit (respx), e2e against the fake upstream |
| Anthropic adapter (chat, stream, tools, images) | 🟡 | unit (respx) |
| Google Gemini adapter (chat, stream, embeddings, health) | 🟡 | unit (respx) |
| Ollama adapter (chat, embed, tags) | 🟡 | unit (respx) |
| YAML model catalogue, `${VAR:-default}` expansion, multiple deployments per model | ✅ | unit, int, e2e |
| Runtime catalogue reload (`POST /admin/api/config/reload`) | ✅ | unit, int |

## 3. Routing and resilience

| Feature | Status | Tests |
|---|---|---|
| Strategies: priority, least-cost, lowest-latency (EWMA), weighted, conditional | ✅ | unit, e2e (`routing.feature`) |
| Per-request strategy override | ✅ | unit, e2e |
| Retries with full-jitter backoff honouring `Retry-After` | ✅ | unit, int |
| Cross-deployment fallback chain (`MAX_FALLBACKS`) | ✅ | unit, int, e2e (dead primary in `models.eval.yaml`) |
| Request log, cost, and console name the deployment that *served* a failed-over request | ✅ | **fixed**: logs blamed the dead primary; int, e2e |
| Streaming fallback before first chunk | ✅ | unit, int |
| Circuit breaker (closed / open / half-open) | ✅ | unit |
| Model allow / block lists on virtual keys | ✅ | unit, int (`test_policies_http.py`) |
| `MAX_RETRIES` means *retries*, not total attempts | ✅ | **fixed** (was off by one); int |
| Conditional routing on request `tags` (as documented) | ⛔ | strategy reads deployment tags only |
| Deployment `rpm_limit` / `tpm_limit` enforcement | ⛔ | stored, never enforced |

## 4. Semantic cache

| Feature | Status | Tests |
|---|---|---|
| Redis Stack HNSW vector cache, lazy index creation / rebuild | ✅ | unit, int, e2e |
| Namespace isolation (model, tenant, system prompt, params, tools) | ✅ | unit |
| Eligibility rules (no tools, temperature cap, finish = stop), `no_cache` opt-out | ✅ | unit, e2e |
| Admin: stats, entries, invalidate | ✅ | unit, int |
| Cache hit served over HTTP against real Redis Stack | ✅ | int (`test_accounting_http.py`) |
| Cache hits are not billed; `cost_saved_usd` recorded | ✅ | **fixed**; int |
| `X-Gateway-Cache-Similarity` response header (documented) | ⛔ | not emitted |
| `estimated_latency_saved_ms` in cache stats | ⛔ | always `null` |

## 5. RAG

| Feature | Status | Tests |
|---|---|---|
| Collections CRUD (`/v1/rag/collections`) | ✅ | int (`test_rag_api.py`) |
| Document ingest (JSON / multipart text), list, delete | ✅ | unit, int |
| Markdown-aware chunking, idempotent ingest, rollback on vector failure | ✅ | unit, int |
| Chunk inspector (list / get with embedding preview) | ✅ | unit, int |
| `POST /v1/rag/search` with MMR diversity and filters | ✅ | unit, int |
| `POST /v1/rag/query` (retrieve + chat), runs as the calling key | ✅ | **fixed** (was always 500); int |
| RAG augmentation inside the chat pipeline (`RagService.augment`) | ⛔ | never called |
| `RAG_INDEX_NAME` setting | ⛔ | never read |
| Non-text ingestion (PDF, DOCX) | 💡 | |

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
| MCP clients / stdio processes closed on shutdown | ⛔ | leaked |
| Agentic tool loop (`run_agentic_loop`) wired into chat | ⛔ | implemented, never called |
| Tool calls routed through the pipeline (logging, cost, guardrails) | ⛔ | documented, not implemented |

## 7. Guardrails

| Feature | Status | Tests |
|---|---|---|
| Rule types: regex, denylist, length, topic, PII (Luhn, SSN, email…) | ✅ | unit, int |
| Actions: block, redact, flag; input and output stages | ✅ | unit, int, e2e |
| Violations persisted and listed (`/admin/api/guardrails/violations`) | ✅ | unit, int |
| Block / redact observed over HTTP (input PII, prompt injection, output secrets, per-key policy) | ✅ | int (`test_policies_http.py`) |
| Output guardrails on streamed responses | ⛔ | now run after the stream (flag + record), but cannot redact text already sent |
| LLM-judge guardrail | ⛔ | raises `NotImplementedError` |

## 8. Auth, virtual keys, quotas

| Feature | Status | Tests |
|---|---|---|
| Master key; virtual keys (`sk-aigw-…`, SHA-256 hashed, Redis snapshot cache) | ✅ | unit, int |
| Key create (shown once), list, get, update, regenerate, delete, disable | ✅ | unit, int (`test_admin_flows.py`) |
| Expired / inactive key rejection | ✅ | unit, int |
| RPM rate limit (Redis sliding window, fail-open) → 429 + `Retry-After` | ✅ | unit, int |
| Budgets on keys and teams (charged for unary and streamed requests) | ✅ | **fixed** (spend was never recorded); int |
| Teams CRUD and team usage | ✅ | int |
| Key `tpm_limit`, `max_parallel_requests`, `allowed_routes` | ⛔ | stored, never enforced |
| `Retry-After` reflects real wait time | ⛔ | always the full 60 s window |

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
| `aigw_active_requests` gauge, rate-limit-hit counter | ⛔ | never updated |

## 10. Admin API (`/admin/api`)

| Feature | Status | Tests |
|---|---|---|
| Console login (argon2 + JWT), `me`, logout | ✅ | unit, int |
| Change password (admin + self-service) | ✅ | unit, int |
| Users CRUD, last-admin protection | ✅ | unit, int |
| Role enforcement (admin vs viewer) on admin, RAG, and MCP routes | ✅ | unit, int |
| Dashboard summary and timeseries | ✅ | unit, int |
| Logs list / filter / detail / reveal / CSV export | ✅ | unit, int |
| Usage by dimension, usage costs | ✅ | unit, int |
| Deployments, models, provider status, health check | ✅ | **fixed**: health check 404'd for every generated id (`provider/model`); int |
| System info reports RAG / MCP enabled | ✅ | **fixed**; int |
| Refresh tokens | ⛔ | `create_refresh_token` has no endpoint |
| Server-side logout / token revocation | ⛔ | logout is a no-op |

## 11. Console UI (`/ui`)

All console scenarios run in Chromium via Playwright (`e2e/features/ui`).

| Feature | Status | Tests |
|---|---|---|
| Login, route guards, redirect back after login, sign out | ✅ | e2e (`auth.feature`) |
| Role-based navigation (viewers see read-only sections only) | ✅ | **fixed**: nav hid Cache/RAG/MCP from viewers although routes allowed them; e2e |
| Every page renders without an error | ✅ | e2e (`navigation.feature`) |
| Dashboard | ✅ | e2e |
| Models & routing: priority, weight, tags, true capabilities, chains in priority order | ✅ | **fixed**: columns were hard-coded "—" and chains sorted by id; e2e |
| Deployment health check button | ✅ | **fixed**: always 404 (unencoded `provider/model#n` ids); e2e |
| Logs (filters, search, detail drawer, routing decision) | ✅ | e2e (`logs.feature`) |
| Usage | ✅ | e2e (renders) |
| Guardrails (policies, violations) | ✅ | e2e (renders) |
| Cache (stats, entries, invalidate) | ✅ | e2e (renders) |
| RAG collections (create, paste text, retrieval test) | ✅ | **fixed**: page crashed as soon as a collection existed (TDZ); e2e |
| MCP servers (add, discover tools, health) | ✅ | e2e |
| Virtual keys (create shows secret once, status, disable, delete) | ✅ | **fixed**: every key showed "Disabled" (`is_active` vs `enabled`); e2e |
| Playground (streaming chat through normal routing) | ✅ | **fixed**: pinned a deployment (bypassing fallback), stream errors broke the connection, model picker had no label; int, e2e |
| Settings: user management (add, role, activate/deactivate, delete) | ✅ | **new**: page claimed the API was missing; e2e |
| Settings: provider credential status, system info, subsystems | ✅ | **new** (was "not exposed"); e2e |
| Logout calls the server | ✅ | **fixed** |
| Dead code (`PlaceholderPage`, unused hooks) removed | ✅ | |
| Lint rule against render-time use-before-define | ✅ | `@typescript-eslint/no-use-before-define` |
| Playground RAG mode, compare mode, replay from log | 🟡 | not covered end-to-end |
| Keys: regenerate and edit limits from the UI | 🟡 | API covered (int, e2e); UI not |
| Teams management UI | 💡 | API exists (`/admin/api/teams`), no page |
| Logs CSV export button | 💡 | API exists (`/admin/api/logs/export`), no button |

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
| Alembic migrations (`make migrate`) | ⛔ | no migrations directory |
| Python version pinned to 3.12 for local dev | ✅ | `.python-version` |

## 13. Quality and CI

| Feature | Status | Tests |
|---|---|---|
| Unit suite | ✅ | `make test-unit` |
| Integration suite on real Postgres + Redis Stack | ✅ | `make test-integration` |
| Cucumber + Playwright E2E suite (API via official SDKs + console UI) | ✅ | 70 scenarios / 360 steps, `make e2e` |
| SDK compatibility script (`scripts/agent_compat.py`) | ✅ | 10/10 against fake upstream |
| Evaluation harness (`scripts/evaluate.py`) | ✅ | 100% success, failover and routing verified |
| Deterministic fakes: OpenAI upstream (bag-of-words embeddings), MCP server (HTTP + stdio) | ✅ | `scripts/fake_upstream.py`, `scripts/fake_mcp_server.py` |
| CI: ruff on `tests/`, UI lint, integration job with Postgres + Redis Stack services, E2E job with report artifact | ✅ | `.github/workflows/ci.yml` |
| Coverage report / gate | 💡 | |
| Load test in CI (`scripts/loadtest.py`) | 💡 | |
| Repo-wide `ruff format` (the codebase is not currently format-clean) | 💡 | |

## Proposed next steps

Ordered by value; none are started.

1. **RAG in the chat pipeline**: call `RagService.augment` for requests that name a collection, so any client (not only `/v1/rag/query`) gets grounded answers.
2. **Agentic tool loop**: wire `run_agentic_loop` so MCP tools can be auto-injected and executed server-side, with each tool call logged and costed.
3. **Enforce the stored-but-ignored limits**: key `tpm_limit` / `max_parallel_requests` / `allowed_routes`, deployment `rpm_limit` / `tpm_limit`.
4. **Streaming output guardrails**: buffer-and-release or chunk-level redaction so secrets cannot leak through streams.
5. **Alembic migrations** so schema changes are safe after the first deploy (`make migrate` currently has nothing to run).
6. **Refresh tokens and server-side logout** (token revocation list in Redis).
7. **Teams page and logs CSV export button** in the console (APIs already exist).
8. **Conditional routing on request `tags`**, as documented.
9. **`X-Gateway-Cache-Similarity` header** and recorded latency savings for cache hits.
10. **LLM-judge guardrail**.
