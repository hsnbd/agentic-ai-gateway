# Configuration reference

Configuration comes from two places: **environment variables** for secrets and
deployment-specific values, and **YAML files** for the model catalogue, pricing,
and guardrail policies.

Every environment variable maps to a field on `Settings` in
[`app/config/settings.py`](../app/config/settings.py). Names are case-insensitive.

## Environment variables

### Core

| Variable | Default | Purpose |
|---|---|---|
| `ENVIRONMENT` | `development` | Environment label used in logs and traces |
| `DEBUG` | `false` | Verbose errors; never enable in production |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `4000` | Bind port |
| `ROOT_PATH` | `""` | Set when served behind a path-prefixing proxy |

### Datastores

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | local Postgres | Async SQLAlchemy URL (`postgresql+asyncpg://…`) |
| `REDIS_URL` | `redis://localhost:6379/0` | Must be **Redis Stack** — vector search is required |
| `DB_POOL_SIZE` | `10` | Connection pool size |
| `DB_MAX_OVERFLOW` | `20` | Overflow connections |
| `DB_ECHO` | `false` | Log every SQL statement |

### Security

| Variable | Default | Purpose |
|---|---|---|
| `MASTER_KEY` | — | Root key; bypasses virtual-key checks. Treat as a secret |
| `JWT_SECRET` | — | Signs console sessions. Use at least 32 bytes |
| `BOOTSTRAP_ADMIN_EMAIL` | — | First-run console admin |
| `BOOTSTRAP_ADMIN_PASSWORD` | — | First-run console password; change after login |

### Providers

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | OpenAI credentials and endpoint |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` | Anthropic credentials and endpoint |
| `GEMINI_API_KEY` / `GEMINI_BASE_URL` | Google Gemini credentials and endpoint |
| `OLLAMA_BASE_URL` | Local Ollama endpoint; needs no key |

Provider keys are optional. The gateway starts with whatever is configured, so a
laptop with only Ollama running is a valid deployment.

### Request handling, routing, resilience

| Variable | Default | Purpose |
|---|---|---|
| `REQUEST_TIMEOUT_SECONDS` | `120.0` | Whole-request timeout |
| `CONNECT_TIMEOUT_SECONDS` | `10.0` | Connection timeout |
| `MAX_RETRIES` | `2` | Retries **per deployment** before falling back |
| `RETRY_BASE_DELAY_SECONDS` | `0.5` | Initial backoff |
| `RETRY_MAX_DELAY_SECONDS` | `8.0` | Backoff ceiling |
| `MAX_FALLBACKS` | `3` | Deployments tried before giving up |
| `ROUTING_STRATEGY` | `priority` | Default strategy; overridable per request |
| `CIRCUIT_BREAKER_THRESHOLD` | `5` | Consecutive failures before opening |
| `CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `30.0` | Wait before a probe is admitted |

### Cache

| Variable | Default | Purpose |
|---|---|---|
| `CACHE_ENABLED` | `true` | Master switch |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Cosine similarity required for a hit |
| `CACHE_TTL_SECONDS` | `3600` | Entry lifetime |
| `CACHE_EMBEDDING_MODEL` | `nomic-embed-text` | Model used to embed prompts |
| `CACHE_EMBEDDING_DIMENSIONS` | `768` | Must match the embedding model |
| `CACHE_MAX_TEMPERATURE` | `0.3` | Above this, responses are not cached |

Lowering `CACHE_SIMILARITY_THRESHOLD` raises the hit rate and the risk of serving a
subtly wrong answer. Treat it as a correctness setting, not a performance knob.

### Guardrails and observability

| Variable | Default | Purpose |
|---|---|---|
| `GUARDRAILS_ENABLED` | `true` | Master switch |
| `LOG_LEVEL` | `INFO` | Logging threshold |
| `LOG_FORMAT` | `json` | `json` or `console` |
| `LOG_REQUEST_BODIES` | `false` | Log prompts and responses. Privacy-sensitive |
| `METRICS_ENABLED` | `true` | Expose Prometheus metrics |
| `TRACING_ENABLED` | `false` | Emit OpenTelemetry traces |
| `OTLP_ENDPOINT` | — | OTLP collector endpoint |

## `config/models.yaml`

The model catalogue. Each entry binds a client-visible model name to a provider
deployment.

```yaml
model_list:
  - model_name: gpt-4o          # what clients ask for
    params:
      provider: openai          # which adapter serves it
      model: gpt-4o             # the upstream model id
      api_key: ${OPENAI_API_KEY}
    priority: 1                 # lower is preferred
    weight: 1                   # tie-break, and weighted routing
    tags: [premium]             # used by conditional routing

  - model_name: gpt-4o          # same name, second deployment
    params:
      provider: anthropic
      model: claude-sonnet-4
      api_key: ${ANTHROPIC_API_KEY}
    priority: 2                 # the fallback

aliases:
  gpt-4-turbo: gpt-4o           # keep old client names working
```

Several deployments may share a `model_name`. That is how load balancing and
cross-provider fallback work: the router sees them all as candidates for the same
request.

`${VAR}` and `${VAR:-default}` are expanded from the environment, so secrets stay
out of the file.

## `config/pricing.yaml`

Per-million-token prices, used for cost estimation and budget enforcement.

```yaml
models:
  gpt-4o:
    input_per_mtok: 2.50
    output_per_mtok: 10.00
    cached_input_per_mtok: 1.25
    providers:                  # optional per-provider overrides
      azure:
        input_per_mtok: 2.40
```

## `config/guardrails.yaml`

Named policies, each with input and output rules. A `default` policy is required.
Virtual keys may bind to a specific policy.

```yaml
policies:
  default:
    description: Baseline protections
    input:
      - type: pii
        action: redact          # block | redact | flag
        entities: [email, phone, credit_card, ssn]
      - type: denylist
        action: block
        terms: [...]
    output:
      - type: pii
        action: redact
```

Actions:

- **block** — reject the request with a guardrail error
- **redact** — replace the matched span and continue
- **flag** — allow, but record the violation

Redaction resolves all matches across all rules over the *original* text in a single
pass, so overlapping entities cannot corrupt each other.

## Per-request overrides

Clients may override gateway behaviour per request. These fields are consumed by
the gateway and never forwarded upstream:

| Field | Effect |
|---|---|
| `no_cache` | Skip cache lookup and write |
| `cache_ttl` | Override TTL for this entry |
| `fallbacks` | Explicit ordered fallback models |
| `routing_strategy` | Override the strategy for this request |
| `guardrail_policy` | Use a named policy |
| `tags` | Labels for conditional routing and log filtering |
