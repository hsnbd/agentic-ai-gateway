# Building agentic applications on the gateway

An agent application (a chatbot, a support assistant, a LangGraph workflow)
can treat the gateway as its whole model-and-data plane. The application keeps
what is specific to it: conversation memory, its users and their roles, and
the UX for approvals. The gateway supplies the rest and enforces it, so an
application bug or a prompt injection cannot widen what a user may do.

| The application needs | Gateway feature |
|---|---|
| A chat model with tools and streaming | `/v1/chat/completions` (any OpenAI SDK, LangChain `ChatOpenAI`) |
| Fallback, retries, caching, cost limits | Routing, semantic cache, key/team budgets and rate limits |
| Retrieval over its documents | RAG collections, `/v1/rag/search` (several collections at once) |
| Actions in other systems | MCP servers, `/v1/mcp/tools` and `/v1/mcp/tools/call` |
| Per-role access | One virtual key per role: model, collection and tool allowlists |
| Acting for one user only | `tool_arg_bindings` + the request's end user |
| Safety | Guardrail policies on prompts, answers, tool arguments and tool results |
| Accountability | Request log, tool-call audit (with bound arguments), metrics |
| Answer quality | DeepEval harness (`python -m evals`) with the gateway as judge |

`apps/iit-hub/` is a complete example: a LangGraph agent for a university
(guests, students, teachers) built this way.

## 1. One key per role

Create a team for the application (one budget) and a key per user role. Each
key carries that role's scope:

```json
POST /admin/api/keys
{
  "name": "support-bot-customer",
  "team_id": "<team>",
  "allowed_models": ["gpt-4o-mini"],
  "allowed_collections": ["help-center-*"],
  "allowed_mcp_servers": ["crm"],
  "allowed_tools": ["crm__get_my_*", "crm__open_ticket"],
  "tool_arg_bindings": {"crm__*": {"customer_id": "end_user"}},
  "guardrail_policy": "user-data",
  "rpm_limit": 60,
  "max_budget_usd": 25
}
```

- `allowed_collections`: ids or names, `*` wildcards. It narrows ownership and
  never widens it: the key still sees only collections its team owns, or global
  ones. Anything else is reported as not found, in listings, search, and
  chat grounding (`aigw.rag`).
- `allowed_mcp_servers` and `allowed_tools` hide other tools from
  `/v1/mcp/tools`. They also refuse those tools in the agent loop and in direct
  calls.

The application chooses the key from the user's role, so a role is enforced
by the gateway even if the application asks for more.

## 2. Bind identity arguments

Tools that act for a user usually take that user's id (`customer_id`,
`student_id`). If the model fills it in, a prompt such as "show the orders of
customer 42" can make it act for someone else. A key's `tool_arg_bindings`
makes the gateway fill such arguments itself:

| Source | Value |
|---|---|
| `end_user` | The end user: the chat request's standard OpenAI `user` field (agent loop), or `end_user` in `POST /v1/mcp/tools/call` |
| `key_id`, `team_id` | The calling key or its team |
| `none` | Always removed, so the model can never supply it, e.g. for anonymous users |

Patterns match namespaced tool names (`server__tool`, `*` wildcards). Later
patterns override earlier ones. A bound argument the model sends is always
discarded. A value is filled in for arguments the tool's schema declares, so one
pattern can cover a whole server. Bound arguments are **removed from the tool schemas** the model is shown, in
`/v1/mcp/tools` and in the agent loop, so the model does not even see them. A
call that needs an end user, made without one, is refused with status
`unbound`. Every bound argument is recorded in the tool-call audit
(`bound_args`, `/admin/api/tool-calls`).

```python
# Direct call (the application runs the tool loop itself, e.g. LangGraph):
httpx.post(f"{GW}/v1/mcp/tools/call", headers=auth, json={
    "name": "crm__get_my_orders", "arguments": {}, "end_user": "cust-7"})

# Agent loop on the gateway (`aigw.mcp`): the end user is the request's `user`.
client.chat.completions.create(model="gpt-4o-mini", user="cust-7", messages=[...],
                               extra_body={"aigw": {"mcp": {}}})
```

## 3. LangChain and LangGraph

The gateway is an OpenAI endpoint, so LangChain needs no adapter. This is
checked by `scripts/agent_compat.py` (LangChain section) and
`tests/integration/test_langchain_compat.py`, which cover invoke, streaming,
`bind_tools`, `with_structured_output`, embeddings and a LangGraph ReAct agent.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(
    model="gpt-4o-mini",                      # or "auto"
    base_url="http://gateway:4000/v1",
    api_key=role_key,                         # the user's role
    model_kwargs={"user": end_user_id},       # request log + end_user bindings
    extra_body={"aigw": {"tags": ["support-bot"]}},  # any gateway control
)
embeddings = OpenAIEmbeddings(model="nomic-embed-text", base_url=..., api_key=role_key,
                              check_embedding_ctx_length=False)
```

Retrieval: call `/v1/rag/search` with `collection_ids` (every collection the
role key can list) and wrap the results as a retriever or a tool. Tools: turn
each entry of `GET /v1/mcp/tools` into a tool for the model, since its JSON
schema can be bound as is. Then execute calls with `POST /v1/mcp/tools/call`, passing
`end_user`. Approvals (human in the loop) belong in the application, e.g. a
LangGraph `interrupt()` before write tools, because only it can ask the user.

## 4. Guardrails for applications that collect user data

The `default` and `strict` policies mask email addresses, so a sign-up or
enrollment flow would receive `***`. The `user-data` policy keeps email
addresses. It still blocks prompt injection and oversized input, redacts
government ids and card numbers, and redacts leaked secrets from answers.
Policies also screen tool arguments and results (`apply_to_tools`).

## 5. Evaluate answer quality

`python -m evals <dataset.yaml>` scores answers with DeepEval, judging
through the gateway. A dataset names a target:

- `gateway`: chat with optional `aigw.rag` / `aigw.mcp`. Sources become the retrieval context and executed tools become `tools_called`.
- `http`: any application API, with JSONPath-like selectors for the answer, sources and tools.

It also lists cases with metrics such as `answer_relevancy`, `faithfulness`,
`contextual_recall`, `hallucination`, `tool_correctness`, `toxicity`, `bias`,
`pii_leakage` and `geval:<criteria>`. It writes `eval.json`, which
`scripts/bench_compare.py` gates as `eval.*` metrics. See
`bench/datasets/eval_gateway.yaml` and `apps/iit-hub/evals/dataset.yaml`.
