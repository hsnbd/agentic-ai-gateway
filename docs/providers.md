# Adding a provider

The gateway has no vendor SDKs. Every provider is a hand-written adapter that
translates the canonical intermediate representation into one vendor's HTTP API
and back. Adding a provider is therefore a single file plus one line in the
registry, and nothing else in the gateway needs to know it exists.

This guide walks through the contract, the three things that are easy to get
wrong, and how to prove the adapter is correct before wiring it in.

## The contract

A provider subclasses [`Provider`](../app/providers/base.py). Two methods are
required; the rest have working defaults.

```python
class Provider(abc.ABC):
    name: str = "base"

    def __init__(self, client: httpx.AsyncClient) -> None: ...

    @abc.abstractmethod
    async def chat(self, request: ChatRequest, deployment: Deployment) -> ChatResponse: ...

    @abc.abstractmethod
    def stream(self, request: ChatRequest, deployment: Deployment) -> AsyncIterator[StreamChunk]: ...

    async def embed(self, request, deployment) -> EmbeddingResponse: ...   # optional
    async def health_check(self, deployment) -> bool: ...                  # optional
```

Three invariants:

- **Adapters are stateless and shared.** One instance serves every concurrent
  request. Per-request state belongs in local variables, never on `self`.
- **The `httpx.AsyncClient` is injected**, not created. The registry owns
  connection pooling and lifetime.
- **`stream()` is not `async def`.** It returns an async iterator. Declaring it
  `async def` with `yield` also produces an async generator, but the registry
  calls it without awaiting, so define it as a normal method returning a
  generator, or make the body an inner `async def _gen()` you return.

## Step 1 — write the adapter

Create `app/providers/yourvendor.py`.

```python
class YourVendorProvider(Provider):
    name = "yourvendor"

    def _headers(self, deployment: Deployment) -> dict[str, str]:
        return {
            **super()._headers(deployment),
            "Authorization": f"Bearer {deployment.api_key}",
        }

    async def chat(self, request: ChatRequest, deployment: Deployment) -> ChatResponse:
        try:
            response = await self._client.post(
                f"{self._base_url(deployment)}/chat",
                headers=self._headers(deployment),
                json=self._payload(request, deployment),
            )
            response.raise_for_status()
        except Exception as exc:
            raise self.map_error(exc, deployment) from exc
        return self._to_canonical(response.json(), request)
```

Inherit `map_error()` wherever possible — it already maps timeouts,
connection failures and HTTP status codes onto the shared taxonomy, and the
retry policy and circuit breaker make their decisions from those codes. An
adapter that raises raw `httpx` exceptions will not be retried correctly and
will not open the breaker.

Override `_map_status_error()` only when the vendor hides the real reason in
the body — for example returning `400` for a context-length overflow that
should surface as a distinct error rather than a generic bad request.

### Building the request payload

Use `self._merge_params(request, deployment)` so deployment-level defaults sit
*beneath* explicit request values. Then apply the vendor's own naming.

**Never forward gateway-only fields to a vendor.** `ChatRequest` carries
`no_cache`, `cache_ttl`, `fallbacks`, `routing_strategy`, `guardrail_policy`
and `tags`. These are instructions to the gateway, and a vendor that rejects
unknown fields will fail the request. Build the payload from an explicit
allowlist of keys rather than dumping the model.

### Translating the response

Populate `ChatResponse` fully. `choices` is required. Map the vendor's stop
reason onto the canonical `finish_reason` — routing and the tool loop both
read it, and a wrong value silently breaks multi-turn tool calling.

Extract `Usage` honestly. If the vendor does not report token counts, leave
them at zero rather than estimating; the accounting layer treats zero as
"unknown" and an invented number becomes a wrong cost on someone's invoice.

### Streaming

Yield `StreamChunk` objects. **The field is `content`, not `delta`.** Pydantic
ignores an unknown keyword silently, so a typo here produces a stream that
parses cleanly and delivers nothing — this has already bitten this codebase
once.

Emit tool-call fragments as they arrive and set `finish_reason` on the final
chunk. If the vendor reports usage in a trailing event, attach it to the last
chunk so accounting and the console inspector can see it.

## Step 2 — declare capabilities and pricing

Routing filters candidates through `Capabilities.supports(request)` before it
ranks them. If you leave `tools=False` on a provider that supports tools, the
router will silently never choose it for a tool-calling request, and the
symptom looks like a routing bug rather than a declaration mistake.

```python
def capabilities(self, deployment: Deployment) -> Capabilities:
    return Capabilities(
        chat=True, streaming=True, tools=True,
        vision=False, json_mode=True, embeddings=True,
        max_context_tokens=128_000, max_output_tokens=8_192,
    )
```

Pricing is USD per **one million** tokens. Getting the unit wrong is a
thousandfold cost error in the dashboards, so check it against the vendor's
published page rather than copying another adapter.

## Step 3 — register it

Add the class to `_register_providers` in
[`app/providers/registry.py`](../app/providers/registry.py):

```python
from app.providers.yourvendor import YourVendorProvider

for cls in (OpenAIProvider, AnthropicProvider, GeminiProvider,
            OllamaProvider, YourVendorProvider):
```

This import is eager and unconditional: **a syntax error in any adapter stops
the whole gateway from booting.** That is deliberate — a provider that cannot
load is a configuration failure, not something to discover at request time.

## Step 4 — configure a deployment

In `config/models.yaml`. Secrets come from the environment; `${VAR}` is
expanded at load time so keys never live in the file.

```yaml
model_list:
  - model_name: my-model               # what clients ask for
    provider: yourvendor               # must equal Provider.name
    provider_model: vendor-model-v2    # what the vendor calls it
    api_key: ${YOURVENDOR_API_KEY}
    base_url: https://api.yourvendor.com/v1
    priority: 1                        # lower is preferred
    weight: 1
    tags: [cheap, fast]
```

Several deployments may share one `model_name`. That is how fallback chains and
load balancing are expressed: the router sees them as interchangeable
candidates for the same public model and orders them by the active strategy.

## Step 5 — prove it

The cross-provider conformance suite in `tests/unit/test_provider_conformance.py`
runs the same expectations against every adapter. Add yours to its parameter
list and it will check the behaviour that actually matters:

- canonical request in, vendor payload out, with no gateway-only fields leaking
- vendor response in, fully populated `ChatResponse` out
- streaming yields non-empty `content` and terminates with a `finish_reason`
- each vendor error shape maps to the right `ErrorCode`
- usage extraction, including the absent-usage case

Write vendor-specific tests with `respx` to stub HTTP, so no test touches the
network. Then run the real thing once by hand:

```bash
uv run python scripts/evaluate.py --api-key sk-... --model my-model --requests 5
```

The evaluation harness drives the deployed gateway over HTTP and will show
whether the adapter's latency, token accounting, cost and streaming all behave
under real conditions.

## Checklist

- [ ] `name` is unique, stable, and matches `provider:` in `models.yaml`
- [ ] Adapter holds no per-request state
- [ ] All errors pass through `map_error()`
- [ ] No gateway-only fields reach the vendor
- [ ] `StreamChunk` uses `content`
- [ ] `finish_reason` is mapped, not passed through
- [ ] Usage is extracted or honestly zero
- [ ] Capabilities are accurate — especially `tools`
- [ ] Pricing is per million tokens
- [ ] Registered in `_register_providers`
- [ ] Added to the conformance suite
