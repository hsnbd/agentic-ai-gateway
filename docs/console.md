# Console user guide

The console is a React single-page application served by the gateway itself at
`/ui`. It is not a separate service: the same process that answers
`/v1/chat/completions` also serves the console bundle, so there is no CORS
surface and no second deployment to operate.

Everything the console does is backed by a documented endpoint under
`/admin/api/*`. The UI is a client of that API, never a privileged side door —
anything you can do in the browser you can also do with `curl`.

## Contents

- [Signing in](#signing-in)
- [Roles](#roles)
- [Dashboard](#dashboard)
- [Keys](#keys)
- [Teams](#teams)
- [Models and routing](#models-and-routing)
- [Logs](#logs)
- [Usage](#usage)
- [Guardrails](#guardrails)
- [Cache](#cache)
- [RAG](#rag)
- [MCP](#mcp)
- [Playground](#playground)
- [Settings](#settings)
- [Running the console in development](#running-the-console-in-development)
- [Troubleshooting](#troubleshooting)

## Signing in

Open `http://<gateway-host>:8000/ui`. Unauthenticated visitors are redirected to
`/ui/login`.

Console accounts are **not** virtual API keys. They are rows in the
`admin_users` table, authenticated with email and password and exchanged for a
short-lived JWT. The two credential systems are deliberately separate:

| Credential | Authenticates | Used for |
|---|---|---|
| Virtual API key (`sk-...`) | `/v1/*`, `/gateway/*` | Applications and agents sending traffic |
| Console account (email + password → JWT) | `/admin/api/*` | Humans operating the gateway |

A console JWT sent to `/v1/chat/completions` is rejected with `401`, and a
virtual key sent to `/admin/api/keys` is rejected too. This is intentional: an
operator session should never be able to silently spend a customer's budget.

### The first administrator

There is no sign-up page. On startup, if the `admin_users` table is empty, the
gateway seeds one administrator from configuration:

```bash
BOOTSTRAP_ADMIN_EMAIL=admin@example.com
BOOTSTRAP_ADMIN_PASSWORD=<a real password>
```

The seed runs **only when no users exist**, so it will not resurrect an account
you deliberately deleted, and it will not overwrite a changed password. If you
leave `BOOTSTRAP_ADMIN_PASSWORD` at its default of `admin`, the gateway logs a
loud security warning at startup. Change it before exposing the console.

You can also create the first administrator from the command line, which is the
better option in production because the password never has to sit in the
environment of a long-running process:

```bash
aigateway create-admin --email admin@example.com --password "$ADMIN_PASSWORD" --role admin
```

### Sessions

Sign-in returns a short-lived access token (`JWT_ACCESS_TTL_SECONDS`, default
one hour) and a refresh token (`JWT_REFRESH_TTL_SECONDS`, default seven days).
When the access token expires, the console trades the refresh token for a new
pair without interrupting you; each refresh token works once. You are returned
to the login page only when the refresh token is also expired or revoked.

**Signing out is enforced by the server.** Sign out revokes both tokens, so a
copied token stops working immediately. Changing a password, or an admin
changing someone's role or deactivating them, signs that user out of every
session. Revocations live in Redis; if Redis is unreachable, tokens are checked
only for signature and expiry until it returns. Set `JWT_SECRET` to a real random value of
at least 32 bytes in production — the default is a placeholder, and anyone who
knows it can mint an admin session.

## Roles

| Role | Can read | Can change |
|---|---|---|
| `viewer` | Dashboard, Teams, Models, Logs, Usage, Guardrails, Cache, RAG, MCP | nothing |
| `admin` | everything, plus Keys, Settings, Playground | everything |

Viewers get read access to the operational pages because the common case for a
second pair of eyes — "is the gateway healthy, what is it costing us, why did
that request fail?" — should not require handing out the ability to mint keys.

Three areas stay admin-only. **Keys** displays and rotates credentials.
**Settings** manages accounts and provider credential status. **Playground**
sends real requests to real providers and therefore spends real money.

The role guard in the browser is a convenience, not the security boundary. Every
admin endpoint re-checks the role server-side, so a viewer who edits their own
JavaScript still cannot create a key.

## Dashboard

The landing page. KPI cards across the top cover request success rate, p50 and
p95 latency, cache hit ratio, spend, tokens, failover count, and in-flight
requests for the selected time window. Below them are time series for requests
by status, spend, and tokens, plus breakdowns by model, provider, and key.

The **provider health** panel at the bottom lists every configured deployment
with its circuit-breaker state:

- **closed** — healthy, taking traffic.
- **open** — failing; the breaker has tripped and the router is skipping this
  deployment until the cool-down expires.
- **half-open** — cool-down elapsed, probing with a trial request.

A deployment sitting `open` is the first thing to check when latency rises: the
gateway is quietly failing over, which works but costs more if the fallback is a
more expensive model.

Where a metric is not available the card says so explicitly rather than showing
a zero. A zero and "no data" mean very different things when you are deciding
whether to page someone.

## Keys

Create, inspect, rotate, and revoke virtual API keys.

**The full key is shown exactly once, at creation.** Only a hash is stored, so
it cannot be recovered — copy it then, or rotate to get a new one. Each key
carries:

- **Budget** — a spend cap. Requests are refused once it is exhausted.
- **Rate limits** — requests and tokens per minute, enforced in Redis.
- **Model allowlist** — which models the key may reach. Empty means all.
- **Tags** — arbitrary labels used by conditional routing and for grouping in
  usage reports.

**Rotate** issues a new secret for the same key record, preserving its budget,
limits, and usage history. **Revoke** disables it immediately. Prefer rotation
for routine credential hygiene and revocation for a suspected leak.

## Teams

Teams group virtual keys under a shared budget. The page lists each team's
budget, spend, and budget period; the usage button shows the last 30 days of
requests, tokens, and cost across all of the team's keys. Admins create, edit,
and delete teams, and assign keys to a team from the Keys page. Deleting a team
keeps its keys working; they simply stop sharing its budget.

## Models and routing

Read-only inspection of what the gateway will actually do with a request.

The **active routing strategy** is shown at the top, along with the fallback
order it produces. The **fallback chains** section lists, per model alias, the
ordered deployments the router will try.

`priority` is **lowest-number-wins**, following the DNS SRV and nginx
convention. A deployment with `priority: 0` is tried before `priority: 10`. This
is worth stating because it is the opposite of what "higher priority" suggests
in English, and getting it backwards silently sends all traffic to the wrong
provider.

Deployments come from `config/models.yaml`. **Reload config** re-reads that file
without a restart. The health check button next to each deployment issues a real
probe, which is the quickest way to confirm a credential works after changing
it.

## Logs

The request log explorer, with server-side pagination and filtering by time,
key, model, provider, and status.

Selecting a row opens a detail drawer showing the per-stage timing breakdown,
the routing decision and why it was made, the cache verdict and similarity
score, guardrail results, retry and fallback attempts, and the token and cost
breakdown.

The attempt list is the most useful part of that drawer. A request that
succeeded after two failed providers looks identical to a clean one in the list
view — the attempt history is what tells you the gateway absorbed an outage on
your behalf.

**Export CSV** downloads every log row matching the current filters (not just
the visible page) with the same columns as the table.

Prompt and response bodies sit behind an explicit **reveal** action. They are
subject to the redaction policy and the viewer role, so turning on logging does
not quietly turn the console into a PII browser.

## Usage

Cost and token reporting, grouped by key, model, provider, team, or day, over a
selected window. Use **day** grouping for spend trends and **model** or
**provider** grouping to find where the money is going.

Results export to CSV for finance reporting and for reconciling the gateway's
estimates against a provider invoice.

Costs are **estimates**, computed as tokens × the rates in
`config/pricing.yaml`. They will not match a provider bill to the cent —
discounts, minimums, and rounding all differ. They are accurate enough to catch
a runaway agent loop, which is what they are for.

## Guardrails

Lists the configured policies with their phase (input or output), action
(block, redact, or flag), and scope, alongside a feed of recent violations.

Policies are defined in `config/guardrails.yaml` and bound per key. The
violation feed is the fastest way to spot an over-broad rule: a policy that
fires constantly on legitimate traffic is worse than no policy, because people
start routing around it.

## Cache

Semantic cache statistics: hit ratio, entry count, index size, configured
similarity threshold, and estimated cost and latency saved.

The gateway uses **semantic** caching — a request hits when its embedding is
within the similarity threshold of a stored one, not only when the text matches
exactly. That is what makes it effective for agent traffic, where the same
question arrives phrased slightly differently. It is also why the threshold
matters: set it too low and the cache returns answers to questions nobody asked.
The default is deliberately conservative.

The entry inspector lists cached entries with their model, namespace, hit count,
age, and remaining TTL.

Invalidation comes in three forms, in increasing order of bluntness:

- **By key** — drop one entry.
- **By namespace** — drop everything for a model or key. This is the one you
  want after changing a system prompt, because the old cached answers are now
  wrong in a way no TTL will notice.
- **Flush all** — clear the cache entirely.

Every response carries `X-Gateway-Cache: hit` or `miss`, so callers can tell
what happened without opening the console.

## RAG

Manage retrieval collections. Create a collection, upload documents, and watch
ingestion progress as each document is chunked, embedded, and stored.

The chunk inspector shows exactly what was indexed. When retrieval returns
something irrelevant, the cause is usually visible here — a chunk boundary that
split a table, or a document that ingested as one unreadable blob.

A document that fails ingestion shows an **error message** explaining why rather
than sitting silently in a failed state.

The retrieval tester runs a query against the collection and shows scored
results, which lets you tune the collection before any application depends on
it.

## MCP

The registry of MCP servers, over both HTTP and stdio transports, with the tools
discovered from each and their health status.

Registering a server triggers tool discovery. If discovery fails, the console
shows the underlying reason — an unreachable URL, a command that is not on
`PATH`, a handshake rejection — rather than a generic failure.

Server configuration is echoed back with **environment values redacted**, so you
can confirm which variables were set without exposing the secrets in them.

Discovered tools are callable through `POST /v1/mcp/tools/call`, with arguments
validated against each tool's JSON schema. Tool calls do not yet go through the
chat pipeline, so they are not request-logged, costed, or guardrailed (see
[features.md](../features.md)).

Registering, editing, or deleting a server requires the master key or a console
admin, because a `stdio` server runs a command on the gateway host. Virtual
keys and admins may call tools; viewers may only browse.

## Playground

A streaming chat console for trying models directly.

Pick a model, set parameters and a system prompt, and send. Responses stream
token by token over SSE — the same transport real clients use, which makes the
playground a continuous manual test of the streaming path.

Tool calls are rendered as structured blocks showing the arguments and results,
rather than as raw JSON buried in the text.

The playground routes through `/admin/api/playground/chat`, **not** through
`/v1/chat/completions`. It has to: the browser holds a console JWT, and the
data-plane routes authenticate virtual keys. The admin endpoint authenticates
the operator, then runs the same pipeline internally, so playground requests
still pass through routing, guardrails, caching, and accounting and still appear
in the logs.

Playground requests **cost real money**. The page is admin-only for that reason.

## Settings

Console account management — create, edit, and deactivate users, and assign the
`admin` or `viewer` role — plus provider credential status.

The provider panel reports whether each provider has a usable credential. It
reports presence and validity, never the secret itself. Provider keys come from
the environment or Kubernetes secrets and are not editable from the browser by
design: a credential that can be changed from a web session is a credential that
can be changed by anyone who steals that session.

## Running the console in development

In production the console is prebuilt and served from `/ui`. For UI work, run
Vite separately for hot reload:

```bash
# Terminal 1 — the gateway
uv run aigateway serve --reload

# Terminal 2 — the console dev server
cd ui
npm install
npm run dev
```

Vite serves on port 5173 and proxies `/admin/api` and `/v1` to the gateway, so
the dev server and production builds see the same URLs and there is no CORS
configuration to get wrong.

To produce the assets the gateway serves:

```bash
cd ui && npm run build
```

This writes to `app/ui_static/`, which the gateway mounts at startup. The Docker
image does the same thing in a Node build stage and copies the result into the
Python image, so the shipped container needs no Node runtime.

Set `UI_ENABLED=false` to disable the console entirely — appropriate for a
gateway deployed purely as a data plane. `UI_PATH` changes the mount point if
`/ui` collides with something in your ingress.

## Troubleshooting

**`/ui` returns 404.** The console assets were never built. Run
`cd ui && npm run build`, or use the Docker image, which builds them for you.
The gateway logs `console assets not built` at startup when this is the case.

**A deep link such as `/ui/logs` 404s but `/ui` works.** The SPA fallback is not
serving `index.html` for client-side routes. Those paths exist only in the
browser's router. This is fixed in current builds; if you see it, you are
running an older image.

**Cannot sign in on a fresh deployment.** The bootstrap administrator is only
created when the user table is empty. If you have already created users, the
bootstrap credentials will not work — use `aigateway create-admin` instead. If
the table really is empty, check the startup logs for a bootstrap failure.

**Signed out unexpectedly.** The access token expired. Raise
`JWT_ACCESS_TTL_SECONDS` if the default hour is too short for your operators.

**Every admin request returns 401 after a restart.** `JWT_SECRET` changed, which
invalidates every issued token. Set it explicitly rather than letting it vary
between restarts or between replicas — with more than one replica, an unset
secret means sessions break whenever the load balancer moves you.

**KPI cards show "Not available".** That metric is genuinely not exposed for the
selected window, as opposed to being zero. Check that request logging is enabled
and that the window contains traffic.
