"""The hub's agent, a LangGraph state machine.

    START → agent ─(tool calls needing approval)→ approval ─(approved)→ tools → agent
                  ├(other tool calls)────────────────────────────────→ tools → agent
                  └(answer)→ END                  └(declined)→ agent

* ``agent`` asks the model (through the gateway, with the role's key) with the
  role's system prompt and tools.
* ``approval`` pauses the graph with ``interrupt()`` until the user approves or
  declines; the checkpointer keeps the paused thread.
* ``tools`` runs RAG searches and IIT tools through the gateway, and promotes a
  guest to student when their enrollment created a student record.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from iit_hub.gateway import GatewayClient, GatewayError
from iit_hub.roles import ROLES, system_prompt

SEARCH_TOOL = "search_iit_docs"
#: The IIT tool whose result can turn a guest into a student.
ENROLL_TOOL = "iit__enroll_course"

SEARCH_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL,
        "description": (
            "Search IIT's documents: programs, admission, fees, academic calendar, enrollment "
            "and grading rules, services, and (for teachers) faculty policies."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look up"}},
            "required": ["query"],
        },
    },
}


class HubState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    role: str
    subject_id: str | None
    facts: dict[str, str]
    #: RAG chunks used in the current turn.
    sources: list[dict[str, Any]]


#: (role, end user) -> chat model bound to that role's gateway key.
ModelFactory = Callable[[str, str], BaseChatModel]


def build_graph(
    clients: dict[str, GatewayClient],
    model_for: ModelFactory,
    *,
    checkpointer: Any = None,
    top_k: int = 4,
) -> Any:
    def identity(state: HubState) -> tuple[str, str | None, dict[str, str]]:
        return state.get("role", "guest"), state.get("subject_id"), state.get("facts", {})

    def tool_specs(role: str) -> list[dict[str, Any]]:
        return [SEARCH_SPEC, *clients[role].tools()]

    def agent(state: HubState, config: RunnableConfig) -> dict[str, Any]:
        role, subject_id, facts = identity(state)
        thread_id = config.get("configurable", {}).get("thread_id", "")
        # Guests have no id; the thread stands in for them in the gateway's request log.
        end_user = subject_id or f"guest:{thread_id}"
        model = model_for(role, end_user).bind_tools(tool_specs(role))
        prompt = SystemMessage(system_prompt(role, subject_id, facts))
        return {"messages": [model.invoke([prompt, *state["messages"]])]}

    def route(state: HubState) -> str:
        last = state["messages"][-1]
        calls = last.tool_calls if isinstance(last, AIMessage) else []
        if not calls:
            return END
        guarded = ROLES[identity(state)[0]].needs_approval
        return "approval" if any(call["name"] in guarded for call in calls) else "tools"

    def approval(state: HubState) -> Command[Literal["tools", "agent"]]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        guarded = ROLES[identity(state)[0]].needs_approval
        decision = interrupt(
            {
                "type": "approval",
                "actions": [
                    {"id": call["id"], "tool": call["name"], "arguments": call["args"]}
                    for call in last.tool_calls
                    if call["name"] in guarded
                ],
            }
        )
        if isinstance(decision, dict) and decision.get("approved"):
            return Command(goto="tools")
        declined = [
            ToolMessage(
                "The user declined this action. Do not retry it unless they ask again.",
                tool_call_id=call["id"],
            )
            for call in last.tool_calls
        ]
        return Command(goto="agent", update={"messages": declined})

    def tools(state: HubState) -> dict[str, Any]:
        role, subject_id, facts = identity(state)
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        client = clients[role]
        allowed = {spec["function"]["name"] for spec in tool_specs(role)}
        messages: list[ToolMessage] = []
        sources = list(state.get("sources", []))
        update: dict[str, Any] = {}
        for call in last.tool_calls:
            name, args = call["name"], call["args"]
            try:
                if name not in allowed:
                    content = f"Tool {name} is not available to you."
                elif name == SEARCH_TOOL:
                    found = client.search(str(args.get("query", "")), top_k=top_k)
                    sources.extend(found)
                    content = _render_chunks(found)
                else:
                    content = client.call_tool(name, args, end_user=subject_id)
                    if role == "guest" and name == ENROLL_TOOL:
                        update = _promotion(content, args) or update
            except GatewayError as exc:
                content = f"The request failed: {exc}"
            messages.append(ToolMessage(content, tool_call_id=call["id"], name=name))
        return {"messages": messages, "sources": sources, **update}

    graph = StateGraph(HubState)
    graph.add_node("agent", agent)
    graph.add_node("approval", approval)
    graph.add_node("tools", tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["approval", "tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


def _render_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No matching IIT documents."
    return "\n\n".join(
        f"[{index}] ({chunk.get('source') or 'IIT document'}) {chunk['text']}"
        for index, chunk in enumerate(chunks, start=1)
    )


def _promotion(content: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """The state update that makes a guest the student their enrollment created."""
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(result, dict)
        or not result.get("new_student")
        or not result.get("student_id")
    ):
        return None
    facts = {key: str(args[key]) for key in ("name", "email") if args.get(key)}
    return {"role": "student", "subject_id": str(result["student_id"]), "facts": facts}
