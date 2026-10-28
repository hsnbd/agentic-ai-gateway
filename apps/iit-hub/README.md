# IIT Information Hub

A chatbot for the Institute of Information Technology, built with LangGraph on
top of the AI gateway. It answers questions from IIT's documents, enrolls
students on their behalf, and helps teachers follow course and program
progress. Each user sees only what their role allows.

```
browser ── hub (FastAPI + LangGraph) ──► AI gateway ──► LLM providers (OpenAI, Anthropic, Gemini, Ollama)
                                             ├──► RAG collections: iit-public / iit-student / iit-teacher
                                             └──► MCP server "iit" ──► mock student information system (SQLite)
```

## Roles

Everyone starts as a **guest**. A guest becomes a **student** by enrolling: the
confirmed enrollment admits them and the conversation switches to their new
student id. Teachers sign in with their id and PIN, which the IIT system checks.

| | Guest | Student | Teacher |
|---|---|---|---|
| Documents | public (programs, admission, fees, calendar) | public + student rules and services | public + faculty policies |
| Tools | offered courses, enroll (as a new student) | own profile, enrollments, grades; enroll, drop | own courses, course progress, at-risk students, program progress |
| Needs approval | enroll | enroll, drop | none (read-only) |

The hub picks the role's gateway key. **The gateway enforces the scope**:
collection allowlists, tool allowlists, and `tool_arg_bindings`, which fill
`student_id` / `teacher_id` from the signed-in user and hide them from the
model, so the model cannot act for someone else. Even a hub bug or a prompt
injection cannot widen a role. See `mock_iit/seed.py` (`ROLE_KEYS`) and
[docs/agentic-apps.md](../../docs/agentic-apps.md).

The hub itself keeps the conversation: LangGraph state (messages, role,
student or teacher id) in a SQLite checkpointer, and approvals as a LangGraph
`interrupt()` before any write tool.

## Run it

From the repository root, with the gateway running (`make dev`, or
`uv run aigateway serve`) and a chat model it serves:

```bash
make hub                # mock IIT system + gateway provisioning + hub on http://localhost:8100
make hub HUB_RESET=1    # same, with a freshly seeded student database
```

`make hub` checks that the gateway answers at `GATEWAY_URL` (default
`http://localhost:4000`). It then starts the mock IIT system in the background
on `HUB_MCP_PORT` (default 4300, log in `apps/iit-hub/mcp.log`) and provisions
the gateway: team, MCP server, collections and role keys, written to
`apps/iit-hub/.env`. Finally it serves the hub. Ctrl+C stops both. The steps
also exist separately as `make hub-mcp`, `make hub-seed` and `make hub-run`.

Settings come from `HUB_*` variables or `apps/iit-hub/.env`. `HUB_MODEL` picks
the chat model, any name in the gateway's `config/models.yaml` (default
`gpt-4o-mini`; `gemma4` or `llama3.2` run locally on Ollama). Seeding reads
`MASTER_KEY`, `BOOTSTRAP_ADMIN_EMAIL` and `BOOTSTRAP_ADMIN_PASSWORD` like the
gateway does.

## Try it

1. **As a guest**, ask "When do BSSE admissions open?". The answer comes with its
   sources. Then ask "Show me the grades of S-0001", which is declined.
2. **Enroll**: "Enroll me in SE 501 for MSSE. I'm Rafi Ahmed, rafi@example.com."
   An approval card appears. Approve it, and the badge turns into
   `student S-00xx`. Then ask "What am I enrolled in?".
3. **As that student**, ask "Show the grades of S-0001": you can only see your
   own. The gateway's tool-call log (`/ui`, MCP page) shows
   `bound_args: ["student_id"]`.
4. **Teacher sign-in** `T-03` / PIN `3333`: try "How is SE 302 progressing?",
   "Who is at risk in SE 302?", and "How is BSSE doing by batch?". "Progress
   of SE 501" is refused, because SE 501 isn't their course.

Demo teachers: T-01 … T-04 with PINs 1111 … 4444.

## Develop

```bash
cd apps/iit-hub
uv run pytest           # graph, service and API against a fake gateway and a scripted model
uv run ruff check . && uv run mypy iit_hub mock_iit
make hub-eval           # from the root: DeepEval answer quality of the running hub (evals/dataset.yaml)
```

| Path | |
|---|---|
| `iit_hub/graph.py` | LangGraph agent: agent → approval (interrupt) → tools → agent; guest-to-student promotion |
| `iit_hub/gateway.py` | The gateway's RAG search and MCP tools API |
| `iit_hub/roles.py` | Role prompts and which tools need approval |
| `iit_hub/service.py`, `api.py`, `static/` | Threads, HTTP API, chat page |
| `mock_iit/` | The mock student information system (SQLite), its MCP server, documents, and the provisioning script |
| `evals/dataset.yaml` | DeepEval cases for the hub |
