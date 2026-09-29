# Using the gateway with coding agents and SDKs

The gateway exposes both an **OpenAI-compatible** and an **Anthropic-compatible**
API, so most tools work by changing a base URL and an API key. Nothing else needs
to change.

## Why both dialects

Coding agents are the most demanding clients a gateway sees. They stream, they call
tools in loops, they send long contexts, and they list models at startup. Supporting
them is a correctness test, not just a feature — which is why agent compatibility is
treated as an acceptance criterion here.

Some agents speak OpenAI's API, others speak Anthropic's. Supporting both means a
single gateway deployment serves either without a translation shim in front of it.

## OpenAI-compatible clients

**Endpoints:** `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
`/v1/models`, `/v1/models/{model}`

**Auth:** `Authorization: Bearer <virtual-key>`

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="sk-your-virtual-key",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Streaming, tool calls, and JSON mode all work through the same client.

## Anthropic-compatible clients

**Endpoints:** `/v1/messages`, `/v1/messages/count_tokens`

**Auth:** `x-api-key: <virtual-key>`

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:4000",
    api_key="sk-your-virtual-key",
)

message = client.messages.create(
    model="claude-sonnet-4",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

## opencode

```json
{
  "provider": {
    "aigateway": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://localhost:4000/v1",
        "apiKey": "sk-your-virtual-key"
      },
      "models": {
        "gpt-4o": { "name": "gpt-4o via gateway" },
        "claude-sonnet-4": { "name": "claude-sonnet-4 via gateway" }
      }
    }
  }
}
```

The model names are whatever `config/models.yaml` exposes — they need not match any
real upstream model name. That indirection is the point: you can repoint `gpt-4o` at
a different provider without touching the agent's configuration.

## Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=sk-your-virtual-key
```

## Cursor / Continue

Set the OpenAI base URL to `http://localhost:4000/v1` and the API key to a virtual
key.

## curl

```bash
# Unary
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-virtual-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}'

# Streaming
curl -N http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-virtual-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

## Gateway-specific request fields

Any request, in either dialect, may carry extra fields the gateway consumes and
strips before calling a provider, at the top level or nested under `aigw`:

```json
{
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "Hello"}],

  "no_cache": true,
  "fallbacks": ["claude-sonnet-4", "llama3.1"],
  "routing_strategy": "least-cost",
  "guardrail_policy": "strict",
  "tags": ["batch-job"]
}
```

Most SDKs allow extra body fields (`extra_body` in the OpenAI Python client). SDKs
that reject unknown top-level fields can nest them: `{"aigw": {"no_cache": true}}`.

## Server-side RAG and tools

Two fields turn an ordinary chat request into an agentic one, with no change to
the client beyond an extra body field.

**`rag`** retrieves from a gateway RAG collection and adds the context to the
prompt before the model sees it. Retrieved text passes through input
guardrails, and the answer's sources come back with it:

```python
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is our refund policy?"}],
    extra_body={"aigw": {"rag": {"collection_id": "<id>", "top_k": 4}}},
)
completion.aigw["sources"]   # [{"id", "text", "score", "source", ...}]
```

**`mcp`** offers the tools of registered MCP servers to the model and runs any
it calls, feeding results back until the model answers:

```python
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is 2 + 3?"}],
    extra_body={"aigw": {"mcp": {"servers": ["<server id>"], "max_iterations": 8}}},
)
completion.aigw["stop_reason"]          # "completed" | "max_iterations" | "client_tool_call"
completion.aigw["tool_calls_executed"]  # MCP calls the gateway ran
```

- Tools the client sends itself are offered too; if the model calls one, the
  gateway stops and returns that call to the client unexecuted
  (`stop_reason: "client_tool_call"`).
- Auth, budgets, guardrails, and the request log apply once per request; usage
  and cost are summed across every model hop.
- A streamed request is answered by running the loop, then streaming the final
  answer.
- Omit `servers` to offer every healthy server's tools.

## Response headers

| Header | Meaning |
|---|---|
| `X-Gateway-Cache` | `hit` or `miss` |
| `X-Gateway-Cache-Similarity` | Cosine similarity of the matched entry (hits only) |
| `X-Gateway-RAG-Sources` | Chunks retrieved for `rag` requests |
| `X-Gateway-Cost-USD` | Cost charged for this request (0 on a cache hit) |
| `X-Gateway-Provider` | Provider that actually served the request |
| `X-Gateway-Deployment` | Deployment id |
| `X-Gateway-Request-Id` | Correlates with the request log and traces |

These make failover visible: if `X-Gateway-Provider` differs from what you expected,
a fallback happened, and the request log will say why.

## Verified SDK compatibility

`scripts/agent_compat.py` drives the gateway through the **official** SDKs rather
than raw HTTP. That distinction matters: the SDKs validate response types,
reassemble streaming deltas, and accumulate tool-call arguments across chunks, so
they reject shapes that a hand-written `curl` check would happily accept.

```bash
uv run python scripts/agent_compat.py \
  --base-url http://localhost:4030 \
  --api-key "$MASTER_KEY" \
  --model eval-chat
```

Latest run — **10/10 checks passed** against `openai==3.20.0` and `anthropic`:

| SDK | Check | Result |
|---|---|---|
| OpenAI | `models.list` | 3 models, requested model present |
| OpenAI | `chat.completions` unary | content + usage reported |
| OpenAI | `chat.completions` streaming | 7 chunks reassembled |
| OpenAI | Tool call (unary) | `finish_reason=tool_calls` |
| OpenAI | Tool call (streaming) | name reassembled from deltas |
| OpenAI | Tool result round-trip | `role: tool` turn accepted |
| OpenAI | `embeddings` | 256-dim vector |
| Anthropic | `messages` unary | content blocks + usage |
| Anthropic | `messages` with system prompt | accepted |
| Anthropic | `messages` streaming | deltas + final usage |

The tool-result check asserts that the gateway *accepts* an assistant message
carrying `tool_calls` followed by a `role: tool` message and still returns a valid
assistant turn. It deliberately does not require text content, because a second
tool call is an equally valid continuation — and is exactly what a real agent loop
produces.

Run it against the offline stub (`scripts/fake_upstream.py`, prompt trigger
`__tool__` returns a tool call) so the suite needs no provider credentials.

## Things to know

**Model names are gateway names.** `gpt-4o` means whatever `config/models.yaml` says
it means. Aliases let you keep legacy client names working after you repoint them.

**Streaming does not fail over after the first token.** Once bytes have reached the
client, switching providers would splice two different generations together. Errors
after the first token surface to the client instead.

**Tool calls are never cached.** Tool use is stateful and side-effecting; replaying a
cached tool call would be wrong.

**Rate limits and budgets are per virtual key.** A key that exceeds its budget gets a
structured error, not a provider error.
