"""The hub's graph, service and API against a fake gateway and a scripted model.

The fake gateway records which role key and end user each call carried, which
is what the real gateway enforces scopes and bindings on.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ConfigDict

from iit_hub.api import create_app
from iit_hub.gateway import GatewayClient
from iit_hub.graph import _promotion, _render_chunks, build_graph
from iit_hub.roles import system_prompt
from iit_hub.service import ActionPending, HubService, NothingPending, SignInFailed, ThreadNotFound

KEYS = {"guest": "k-guest", "student": "k-student", "teacher": "k-teacher"}
ROLE_OF = {key: role for role, key in KEYS.items()}
TOOLS = {
    "guest": ["iit__list_offered_courses", "iit__enroll_course"],
    "student": ["iit__get_my_grades", "iit__enroll_course", "iit__drop_course"],
    "teacher": ["iit__verify_teacher", "iit__get_course_progress"],
}


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_search = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        role = ROLE_OF[request.headers["authorization"].removeprefix("Bearer ")]
        path = request.url.path
        if path == "/v1/mcp/tools":
            return httpx.Response(
                200,
                json=[
                    {
                        "type": "function",
                        "function": {
                            "name": n,
                            "description": "",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                    for n in TOOLS[role]
                ],
            )
        if path == "/v1/rag/collections":
            return httpx.Response(200, json=[{"id": f"{role}-docs"}])
        if path == "/v1/rag/search":
            if self.fail_search:
                return httpx.Response(503, text="redis down")
            body = json.loads(request.content)
            assert body["collection_ids"] == [f"{role}-docs"]
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "text": "Admission opens in January.",
                            "source": "admission.md",
                            "score": 0.9,
                        }
                    ]
                },
            )
        body = json.loads(request.content)
        self.calls.append({"role": role, **body})
        name = body["name"]
        if name == "iit__enroll_course":
            content = json.dumps(
                {"student_id": "S-0041", "new_student": role == "guest", "course_code": "SE 501"}
            )
        elif name == "iit__verify_teacher":
            ok = body.get("end_user") == "T-03" and body["arguments"] == {"pin": "3333"}
            content = json.dumps(
                {"verified": True, "teacher_id": "T-03", "name": "Nusrat"}
                if ok
                else {"verified": False}
            )
        else:
            content = f"{name} ok"
        return httpx.Response(200, json={"message": {"role": "tool", "content": content}})


class Script:
    """Shared by every model the graph builds (pydantic would copy a plain list)."""

    def __init__(self) -> None:
        self.replies: list[AIMessage] = []
        self.seen: list[dict[str, Any]] = []


class ScriptedModel(BaseChatModel):
    """Returns queued replies; remembers who asked and with what tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: Script
    role: str = ""
    end_user: str = ""
    tools: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self.model_copy(update={"tools": [t["function"]["name"] for t in tools]})

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.script.seen.append(
            {
                "role": self.role,
                "end_user": self.end_user,
                "tools": self.tools,
                "system": messages[0].content,
            }
        )
        return ChatResult(generations=[ChatGeneration(message=self.script.replies.pop(0))])


def _call(tool: str, call_id: str = "c1", **args: Any) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": tool, "args": args, "id": call_id}])


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture
def script() -> Script:
    return Script()


@pytest.fixture
def hub(gateway: FakeGateway, script: Script) -> HubService:
    http = httpx.Client(transport=httpx.MockTransport(gateway.handler))
    clients = {role: GatewayClient("http://gw", key, client=http) for role, key in KEYS.items()}

    def model_for(role: str, end_user: str) -> ScriptedModel:
        return ScriptedModel(script=script, role=role, end_user=end_user)

    return HubService(build_graph(clients, model_for, checkpointer=InMemorySaver()), clients)


def test_guest_questions_are_grounded_in_role_documents(hub: HubService, script: Script) -> None:
    script.replies += [_call("search_iit_docs", query="admission"), AIMessage("In January.")]
    thread = hub.start()["thread_id"]
    view = hub.send(thread, "When does admission open?")
    assert view["answer"] == "In January." and view["role"] == "guest"
    assert view["sources"] == [
        {"source": "admission.md", "text": "Admission opens in January.", "score": 0.9}
    ]
    assert view["tools_called"] == ["search_iit_docs"]
    first = script.seen[0]
    assert first["role"] == "guest" and first["end_user"] == f"guest:{thread}"
    assert first["tools"] == ["search_iit_docs", *TOOLS["guest"]]
    assert "The user is a guest" in first["system"]
    assert [m["role"] for m in view["messages"]] == ["user", "assistant"]

    # The next turn starts with no sources of its own.
    script.replies += [AIMessage("Hello!")]
    assert hub.send(thread, "hi")["sources"] == []


def test_a_guest_becomes_a_student_by_an_approved_enrollment(
    hub: HubService, gateway: FakeGateway, script: Script
) -> None:
    args = {"course_code": "SE 501", "name": "Test User", "email": "t@x.com", "program": "MSSE"}
    script.replies += [_call("iit__enroll_course", **args)]
    thread = hub.start()["thread_id"]
    view = hub.send(thread, "Enroll me in SE 501")
    assert view["pending_action"] == {
        "type": "approval",
        "actions": [{"id": "c1", "tool": "iit__enroll_course", "arguments": args}],
    }
    assert view["answer"] == "" and gateway.calls == []  # nothing ran yet
    with pytest.raises(ActionPending):
        hub.send(thread, "hello?")

    script.replies += [AIMessage("You are enrolled; your id is S-0041.")]
    view = hub.decide(thread, True)
    assert gateway.calls == [{"role": "guest", "name": "iit__enroll_course", "arguments": args}]
    assert (view["role"], view["subject_id"], view["name"]) == ("student", "S-0041", "Test User")
    assert view["answer"].startswith("You are enrolled")
    assert script.seen[-1]["role"] == "student"  # the reply already uses the student key

    # From now on the student key is used, with the student as the end user.
    script.replies += [_call("iit__get_my_grades", call_id="c2"), AIMessage("Your grades...")]
    hub.send(thread, "Show my grades")
    assert gateway.calls[-1] == {
        "role": "student",
        "name": "iit__get_my_grades",
        "arguments": {},
        "end_user": "S-0041",
    }
    assert script.seen[-1]["end_user"] == "S-0041"
    assert "student S-0041 (Test User)" in script.seen[-1]["system"]
    with pytest.raises(NothingPending):
        hub.decide(thread, True)


def test_declined_actions_are_not_run(
    hub: HubService, gateway: FakeGateway, script: Script
) -> None:
    script.replies += [
        _call("iit__enroll_course", course_code="SE 101"),
        AIMessage("Okay, not enrolled."),
    ]
    thread = hub.start()["thread_id"]
    hub.send(thread, "Enroll me")
    view = hub.decide(thread, False)
    assert gateway.calls == [] and view["role"] == "guest"
    assert view["answer"] == "Okay, not enrolled."


def test_read_only_tools_run_without_approval_and_failures_reach_the_model(
    hub: HubService, gateway: FakeGateway, script: Script
) -> None:
    gateway.fail_search = True
    script.replies += [
        AIMessage(
            "",
            tool_calls=[
                {"name": "iit__list_offered_courses", "args": {}, "id": "a"},
                {"name": "search_iit_docs", "args": {"query": "fees"}, "id": "b"},
                {"name": "iit__get_my_grades", "args": {}, "id": "c"},
            ],
        ),
        AIMessage("Here is what I found."),
    ]
    thread = hub.start()["thread_id"]
    view = hub.send(thread, "What courses and fees?")
    assert view["tools_called"] == [
        "iit__list_offered_courses",
        "search_iit_docs",
        "iit__get_my_grades",
    ]
    snapshot = hub.graph.get_state({"configurable": {"thread_id": thread}})
    results = [m.content for m in snapshot.values["messages"] if m.type == "tool"]
    assert results[0] == "iit__list_offered_courses ok"
    assert results[1].startswith("The request failed") and "503" in results[1]
    assert results[2] == "Tool iit__get_my_grades is not available to you."


def test_teacher_sign_in(hub: HubService, gateway: FakeGateway, script: Script) -> None:
    view = hub.start(teacher_id="t-03", pin="3333")
    assert (view["role"], view["subject_id"], view["name"]) == ("teacher", "T-03", "Nusrat")
    assert gateway.calls[0]["end_user"] == "T-03" and gateway.calls[0]["arguments"] == {
        "pin": "3333"
    }
    with pytest.raises(SignInFailed, match="wrong PIN"):
        hub.start(teacher_id="T-03", pin="0000")
    script.replies += [
        _call("iit__get_course_progress", course_code="SE 302"),
        AIMessage("SE 302 is on track."),
    ]
    hub.send(view["thread_id"], "How is SE 302 doing?")
    assert gateway.calls[-1]["end_user"] == "T-03" and gateway.calls[-1]["role"] == "teacher"


def test_unknown_threads(hub: HubService) -> None:
    with pytest.raises(ThreadNotFound):
        hub.view("missing")


def test_helpers() -> None:
    assert _render_chunks([]) == "No matching IIT documents."
    assert _render_chunks([{"text": "t"}]) == "[1] (IIT document) t"
    assert _promotion("not json", {}) is None
    assert _promotion(json.dumps({"student_id": "S-1", "new_student": False}), {}) is None
    assert _promotion(json.dumps({"student_id": "S-1", "new_student": True}), {"name": "A"}) == {
        "role": "student",
        "subject_id": "S-1",
        "facts": {"name": "A"},
    }
    assert "teacher T-01 (?)" in system_prompt("teacher", "T-01", {})


@pytest.fixture
def api(hub: HubService) -> Iterator[TestClient]:
    with TestClient(create_app(lambda: hub)) as client:
        yield client


def test_api_flow(api: TestClient, script: Script) -> None:
    assert "IIT Information Hub" in api.get("/").text
    thread = api.post("/api/threads", json={}).json()["thread_id"]
    script.replies += [
        _call(
            "iit__enroll_course", "c1", course_code="SE 101", name="A", email="a@x", program="BSSE"
        )
    ]
    pending = api.post(f"/api/threads/{thread}/messages", json={"content": "enroll me"}).json()
    assert pending["pending_action"]["actions"][0]["tool"] == "iit__enroll_course"
    assert api.post(f"/api/threads/{thread}/messages", json={"content": "x"}).status_code == 409
    script.replies += [AIMessage("Done.")]
    assert api.post(f"/api/threads/{thread}/approve").json()["role"] == "student"
    assert api.post(f"/api/threads/{thread}/reject").status_code == 409
    assert api.get(f"/api/threads/{thread}").json()["answer"] == "Done."
    assert api.get("/api/threads/nope").status_code == 404
    assert api.post("/api/threads", json={"teacher_id": "T-03", "pin": "bad"}).status_code == 401
    assert api.post(f"/api/threads/{thread}/messages", json={"content": ""}).status_code == 422
