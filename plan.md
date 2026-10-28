# IIT Information Hub as a LangChain app on the gateway, plus generic gateway upgrades

## Context
Your supervisor's feedback (`todo.md`) asks for an IIT information hub: role-based information, helping students enroll in courses, helping teachers follow program progress, and a chatbot that acts on a student's behalf. It also asks for DeepEval-style evaluation.

**Your direction:**
- The hub is a **separate LangChain/LangGraph application** that uses the gateway only through its API.
- The gateway stays **general-purpose**. Anything the hub needs is added as a generic capability that any agentic app can use, with nothing IIT-specific inside `app/`.
- The hub keeps its own conversation memory and identity (a LangGraph checkpointer) and does its own approvals (a LangGraph interrupt).
- The hub's UI is FastAPI plus a simple web chat, in `apps/iit-hub/`.

**How the hub uses the gateway:**

| Gateway feature | Hub use |
|---|---|
| OpenAI-compatible chat (`ChatOpenAI`) | the hub's LLM |
| Routing, fallback, cache, guardrails, budgets, observability | apply to every hub call |
| `/v1/rag/search` | the hub's retriever |
| `/v1/mcp/tools` + `/v1/mcp/tools/call` | the hub's tools: governed, guardrailed, audited |
| Virtual keys, one per role | role-based access |

**Gaps found in the gateway, all closed with generic features:**
1. A key can't be limited to *some* RAG collections. Every key in a team sees all of the team's collections.
2. Search covers one collection per call.
3. There's no way to bind tool arguments to the end user. A model, or a prompt injection, can pass another person's ID.
4. LangChain compatibility isn't verified.
5. Answers aren't evaluated for quality.

## Part A: generic gateway upgrades (in `app/`, each with 100% coverage)

### A1. Per-key RAG collection allow-list
- **Data:**
  - add `allowed_collections` (JSON list of ids or names, with fnmatch patterns; empty means no extra limit) to `VirtualKey` in [app/db/models.py](app/db/models.py);
  - add the same field to `ResolvedKey` and `_resolved_from_row` in [app/auth/keys.py](app/auth/keys.py), and to the key admin schemas in [app/api/schemas_admin.py](app/api/schemas_admin.py) and [app/api/admin.py](app/api/admin.py).
- **Enforcement:** `RagAccess` in [app/rag/access.py](app/rag/access.py) gains `collections: tuple[str, ...]`, filled by `for_principal` and `for_context`.
  - `can_read` and `visible_filter` additionally require the collection's id or name to match.
  - A collection outside the list returns "not found", following the existing rule.
- **Migration:** `app/db/migrations/versions/0005_key_scopes.py`, modeled on `0004_mcp_governance.py`.
- **Console:** a "Collections" field in the Keys page form (`ui/src/pages/Keys.tsx`).

### A2. Search several collections at once
- **API:** `/v1/rag/search` and `/v1/rag/query` in [app/api/rag.py](app/api/rag.py) accept `collection_ids: list[str]` as an alternative to `collection_id`.
- **Chat:** `RagOptions` in [app/core/schemas.py](app/core/schemas.py) gets `extra_collection_ids`.
- **Retrieval:** `RagService.search_many()` in [app/rag/service.py](app/rag/service.py) searches each collection the caller may read and merges the results by score, keeping the top k.
  - Results from different embedding models can't be compared on score; mixing them is refused with 400.
  - `retrieve_and_augment` in [app/rag/stage.py](app/rag/stage.py) uses it.

### A3. Tool arguments bound to the end user
- **Key setting:** keys get `tool_arg_bindings`, e.g. `{"iit__*": {"student_id": "end_user"}}`, which maps tool patterns to argument names. Its source is `end_user`, which is either:
  - the request's standard OpenAI `user` field, for the `aigw.mcp` agent loop; or
  - a new `end_user` field in `McpToolCallRequest`, for `/v1/mcp/tools/call`.
- **Executor:** `ToolExecutor._run` in [app/tools/executor.py](app/tools/executor.py) overwrites the bound arguments before schema validation.
  - If a binding applies but there's no end user, the call is refused with the status `unbound`.
  - The audit records which arguments were bound, in a new `ToolCallLog.bound_args` column added by migration 0005.
  - `ToolCallContext` in [app/tools/governance.py](app/tools/governance.py) carries `end_user` and the bindings.
- **Tool listing:** `/v1/mcp/tools` and the agent loop's offered tools **remove bound arguments from the schema**, so the model never sees `student_id` and can't fake it.

This works for any "act on behalf of a user" agent, not only IIT.

### A4. LangChain compatibility
- **New checks** in [scripts/agent_compat.py](scripts/agent_compat.py):
  - `ChatOpenAI` invoke, streaming and `bind_tools` tool calls;
  - `with_structured_output`;
  - `OpenAIEmbeddings`;
  - a minimal LangGraph `create_react_agent` run against the gateway.
- **Dev dependency:** add `langchain-openai` and `langgraph` to the dev group.
- **Integration test:** `tests/integration/test_langchain_compat.py`, run against the fake upstream.
- Any wire-format gaps these checks find get fixed in the dialects, in [app/dialects/openai_dialect.py](app/dialects/openai_dialect.py).

### A5. Generic DeepEval evaluation of the gateway (top-level `evals/`, outside `app/`)
- **Dependency:** `deepeval` in the dev group of [pyproject.toml](pyproject.toml).
- **`evals/judge.py`:** `GatewayJudge(DeepEvalBaseLLM)`, which judges through the gateway itself using a configurable `EVAL_JUDGE_MODEL`.
- **`evals/targets.py`:** pluggable targets.
  - `GatewayChatTarget` is plain chat, chat with `aigw.rag` grounding, or chat with the `aigw.mcp` agent.
  - `HttpTarget` covers any OpenAI-compatible endpoint, which the hub reuses.
- **`evals/runner.py`:** loads YAML datasets with `input`, `expected_output`, `retrieval_context`, `expected_tools` and `metrics`, then runs these DeepEval metrics:
  - `AnswerRelevancy`, `Faithfulness`, `ContextualPrecision` and `ContextualRecall`;
  - `Hallucination`, `ToolCorrectness`, `Toxicity` and `Bias`;
  - `GEval`.

  It writes `bench/results/eval.json`.
- **Dataset:** `bench/datasets/eval_gateway.yaml` covers RAG grounding, tool use and guardrail refusals.
- **Gating:** `make eval` checks the scores against new `eval.*` entries in [bench/thresholds.yaml](bench/thresholds.yaml) through [scripts/bench_compare.py](scripts/bench_compare.py).
- **Tests:** unit tests for the runner and targets, with DeepEval metrics replaced by test doubles.

## Part B: the IIT Hub app (`apps/iit-hub/`, its own uv project)
```
apps/iit-hub/
  pyproject.toml            # langchain-openai, langgraph, langgraph-checkpoint-sqlite, fastapi, httpx, mcp; dev: deepeval, pytest
  iit_hub/
    config.py               # GATEWAY_URL, per-role gateway keys, model, thread DB path
    gateway.py              # thin httpx client: rag search, mcp tools list/call
    llm.py                  # ChatOpenAI(base_url=GATEWAY/v1, api_key=<role key>, model_kwargs={"user": subject_id}, extra_body={"aigw": {...}})
    retriever.py            # GatewayRetriever(BaseRetriever) → /v1/rag/search (collection_ids from role key's scope)
    tools.py                # StructuredTools built from /v1/mcp/tools (role key) → /v1/mcp/tools/call with end_user
    roles.py                # guest / student / teacher: system prompt, key, which tools need approval
    graph.py                # LangGraph agent (below)
    api.py                  # FastAPI routes + serves static/index.html
    static/index.html       # chat UI: role badge, sources, Approve/Reject cards
  mock_iit/
    mcp_server.py           # MCP server over SQLite: students, courses, enrollments, marks, teachers
    seed.py                 # seeds SQLite + provisions the gateway (see B3)
    docs/{public,student,teacher}/*.md
  evals/
    dataset.yaml, run.py    # reuses the gateway's evals harness through HttpTarget against the hub API
  tests/                    # pytest for graph, tools, api (gateway mocked with httpx MockTransport)
```

### B1. Agent graph (LangGraph)
- **State:** `messages`, `role` (`guest`, `student` or `teacher`), `subject_id`, `facts` (name, email) and `sources`.
- **Persistence:** a SQLite checkpointer keyed by `thread_id`, so the hub remembers who the user is across turns.
- **Nodes:**
  - `agent`: role system prompt plus `ChatOpenAI` through the gateway, using **that role's key** with tools bound. There's a `search_iit_docs` retriever tool, plus MCP tools.
  - `approval`: a LangGraph `interrupt()` before any role write tool (`enroll_course`, `drop_course`). The API returns the pending action, and `POST /approve` or `/reject` resumes the graph with `Command(resume=...)`.
  - `tools`: runs the tool through the gateway's `/v1/mcp/tools/call` with `end_user=subject_id`.
  - `identity`: after a successful `enroll_course` by a guest, it reads the new `student_id` from the result and updates the state to `role=student`, `subject_id` and facts. Later turns switch to the student key and prompt.
- **How a session gets its role:**
  - Everyone starts as a **guest**.
  - A guest becomes a **student only by enrolling**.
  - A **teacher** role is set by the hub when the thread starts, through a teacher sign-in that's checked by the mock IIT system's `verify_teacher` tool. That role is injected into the system prompt and the key choice.

### B2. Roles and how they map onto the gateway

| Role | Gateway key's `allowed_collections` | `allowed_tools` | `tool_arg_bindings` |
|---|---|---|---|
| guest | `iit-public` | `iit__list_offered_courses`, `iit__enroll_course` | none (enrolling creates a new student) |
| student | `iit-public`, `iit-student` | `iit__list_offered_courses`, `iit__get_my_*`, `iit__enroll_course`, `iit__drop_course` | `student_id ← end_user` |
| teacher | `iit-public`, `iit-teacher` | `iit__list_my_courses`, `iit__get_course_progress`, `iit__get_program_progress`, `iit__list_at_risk_students` | `teacher_id ← end_user` |

All three keys belong to an `iit-hub` team, with a budget, rate limits and a guardrail policy (`strict` for guests). Even if the hub has a bug, the gateway still enforces scope and identity.

### B3. Mock IIT system and provisioning
- **`mock_iit/mcp_server.py`:**
  - The tools listed in B2, plus `verify_teacher`.
  - `enroll_course` without a `student_id` creates a student from name, email and program and returns `{student_id, ...}`.
  - It enforces prerequisites, seats and the credit cap, and teacher-owns-course checks.
- **`mock_iit/seed.py`** is idempotent and uses the gateway admin API with the master key. It:
  - builds the SQLite data: the BSSE and MIT programs, about 12 courses, teachers, students and marks;
  - registers the `iit` MCP server;
  - creates the three collections and ingests the docs;
  - creates the team and the three role keys;
  - writes `apps/iit-hub/.env`.
- **Makefile targets:** `hub-seed`, `hub-run` and `hub-eval` in the root Makefile, delegating to `apps/iit-hub`.

### B4. Hub API
- `POST /threads` takes an optional teacher sign-in.
- `POST /threads/{id}/messages` returns the answer, sources, any `pending_action`, the role and subject.
- `POST /threads/{id}/approve` and `/reject`.
- `GET /threads/{id}`.
- `/` serves the chat page.

### B5. Hub evals
`apps/iit-hub/evals/dataset.yaml` covers:
- a guest asking about admission (faithfulness and relevancy against the public docs);
- a guest who enrolls, is asked to approve and becomes a student (tool correctness);
- a student trying to see someone else's grades (a GEval "no impersonation" check, plus a check of the gateway audit for `bound_args`);
- a teacher asking about a course they don't teach (refused).

It runs through the gateway's `evals` runner with `HttpTarget`.

## Implementation order
1. A1 + A2 + migration 0005.
2. A3.
3. A4, fixing any gaps found.
4. A5.
5. B3, the mock system and seed.
6. B1, B2 and B4, the hub.
7. B5 and the docs.

Run `make check` after each gateway step.

## Docs
- New `docs/agentic-apps.md`: how to build any agent app on the gateway, covering role keys, collection scopes, argument binding, the LangChain recipes and evals.
- New `apps/iit-hub/README.md`: setup and a demo script.
- Update `features.md` and the README.
- Cucumber: `e2e/features/api/key_scopes.feature` for collection scope, multi-collection search and argument binding.

## Verification
1. **Gateway:** `make check` passes (lint, types, unit and integration tests, 100% coverage). `uv run python scripts/agent_compat.py --langchain` passes.
2. **Hub:** `make hub-seed`, then `make hub-run`, then open `http://localhost:8100`.
   - **As a guest**, ask about admission. The answer cites public sources, and grades are refused.
   - **The guest says "Enroll me in SE 501"** and gives their details. An approval card appears; after you approve, the SQLite student exists, the badge shows student `S-xxxx`, and "my enrollments" lists SE 501.
   - **As that student, ask for "S-0001's grades".** You only get your own. The gateway's tool-call log shows `bound_args=["student_id"]`, and the console Logs page shows the hub's requests with cost and cache data.
   - **Teacher sign-in, then "SE 302 progress"** returns stats. A course you don't teach is refused.
3. **Evals:** `make eval` (the gateway) and `make hub-eval` both pass their thresholds.
4. **Hub tests:** `cd apps/iit-hub && uv run pytest`.
