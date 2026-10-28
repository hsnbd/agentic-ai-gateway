"""Threads: starting a conversation, sending messages, approving actions, and
the view the API returns after each step."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from iit_hub.config import Settings
from iit_hub.gateway import GatewayClient, GatewayError
from iit_hub.graph import build_graph


class ThreadNotFound(LookupError):
    pass


class ActionPending(RuntimeError):
    """A message arrived while an action is waiting for approval."""


class NothingPending(RuntimeError):
    pass


class SignInFailed(PermissionError):
    pass


class HubService:
    def __init__(self, graph: Any, clients: dict[str, GatewayClient]) -> None:
        self.graph = graph
        self.clients = clients

    @classmethod
    def from_settings(cls, settings: Settings) -> HubService:
        clients = {
            role: GatewayClient(settings.gateway_url, settings.key_for(role))
            for role in ("guest", "student", "teacher")
        }

        def model_for(role: str, end_user: str) -> ChatOpenAI:
            return ChatOpenAI(
                model=settings.model,
                base_url=f"{settings.gateway_url.rstrip('/')}/v1",
                api_key=settings.key_for(role),  # type: ignore[arg-type]
                temperature=settings.temperature,
                # `user` names the end user to the gateway: request logs, and
                # the tool arguments its key binds to the end user.
                model_kwargs={"user": end_user},
                extra_body={"aigw": {"tags": ["iit-hub", f"role:{role}"]}},
            )

        connection = sqlite3.connect(settings.threads_db, check_same_thread=False)
        graph = build_graph(
            clients, model_for, checkpointer=SqliteSaver(connection), top_k=settings.top_k
        )
        return cls(graph, clients)

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def start(self, *, teacher_id: str | None = None, pin: str | None = None) -> dict[str, Any]:
        """Open a thread: as a guest, or as a teacher whose PIN the IIT system accepts.

        Students are never signed in here: a guest becomes a student by enrolling.
        """
        thread_id = uuid.uuid4().hex
        state: dict[str, Any] = {"role": "guest", "subject_id": None, "facts": {}, "sources": []}
        if teacher_id:
            verdict = self._verify_teacher(teacher_id.strip().upper(), pin or "")
            state.update(
                role="teacher", subject_id=verdict["teacher_id"], facts={"name": verdict["name"]}
            )
        self.graph.update_state(self._config(thread_id), state, as_node="__start__")
        return self.view(thread_id)

    def _verify_teacher(self, teacher_id: str, pin: str) -> dict[str, Any]:
        try:
            # The teacher key binds teacher_id to end_user, so only the PIN is sent.
            raw = self.clients["teacher"].call_tool(
                "iit__verify_teacher", {"pin": pin}, end_user=teacher_id
            )
            verdict = json.loads(raw)
        except (GatewayError, json.JSONDecodeError) as exc:
            raise SignInFailed("Teacher sign-in is unavailable right now") from exc
        if not isinstance(verdict, dict) or not verdict.get("verified"):
            raise SignInFailed("Unknown teacher id or wrong PIN")
        return verdict

    def _state(self, thread_id: str) -> Any:
        snapshot = self.graph.get_state(self._config(thread_id))
        if not snapshot.values:
            raise ThreadNotFound(thread_id)
        return snapshot

    def send(self, thread_id: str, content: str) -> dict[str, Any]:
        if self._state(thread_id).interrupts:
            raise ActionPending("Approve or decline the pending action first")
        self.graph.invoke(
            {"messages": [HumanMessage(content)], "sources": []}, self._config(thread_id)
        )
        return self.view(thread_id)

    def decide(self, thread_id: str, approved: bool) -> dict[str, Any]:
        if not self._state(thread_id).interrupts:
            raise NothingPending("There is no action waiting for approval")
        self.graph.invoke(Command(resume={"approved": approved}), self._config(thread_id))
        return self.view(thread_id)

    def view(self, thread_id: str) -> dict[str, Any]:
        snapshot = self._state(thread_id)
        values = snapshot.values
        messages = values.get("messages", [])
        turn_start = max(
            (i for i, message in enumerate(messages) if isinstance(message, HumanMessage)),
            default=0,
        )
        turn = messages[turn_start:]
        answer = next(
            (
                m.content
                for m in reversed(turn)
                if isinstance(m, AIMessage) and not m.tool_calls and m.content
            ),
            "",
        )
        pending = snapshot.interrupts[0].value if snapshot.interrupts else None
        return {
            "thread_id": thread_id,
            "role": values.get("role", "guest"),
            "subject_id": values.get("subject_id"),
            "name": values.get("facts", {}).get("name"),
            "answer": answer if pending is None else "",
            "pending_action": pending,
            "sources": [
                {"source": s.get("source"), "text": s.get("text"), "score": s.get("score")}
                for s in values.get("sources", [])
            ],
            "tools_called": [m.name for m in turn if isinstance(m, ToolMessage) and m.name],
            "messages": [
                {
                    "role": "user" if isinstance(m, HumanMessage) else "assistant",
                    "content": m.content,
                }
                for m in messages
                if isinstance(m, HumanMessage)
                or (isinstance(m, AIMessage) and m.content and not m.tool_calls)
            ],
        }
