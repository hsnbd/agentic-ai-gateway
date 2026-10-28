"""LangChain and LangGraph run against the gateway as an ordinary OpenAI endpoint.

The TestClient is handed to LangChain as its HTTP client, so the real
OpenAI SDK underneath talks to the in-process gateway, and FakeProvider plays
the model (its ``tool_queue`` scripts the tool calls).
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from tests.integration.conftest import FakeProvider, create_virtual_key


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


class Weather(BaseModel):
    """The weather in a city."""

    city: str


def _llm(client: TestClient, api_key: str, **kwargs: Any) -> ChatOpenAI:
    return ChatOpenAI(
        model="test-model",
        base_url="http://testserver/v1",
        api_key=api_key,  # type: ignore[arg-type]
        http_client=client,
        max_retries=0,
        **kwargs,
    )


def test_chat_stream_tools_and_structured_output(
    client: TestClient, auth_headers: dict[str, str], primary: FakeProvider
) -> None:
    key = auth_headers["Authorization"].removeprefix("Bearer ")
    llm = _llm(client, key)

    reply = llm.invoke("Say hello.")
    assert reply.content == primary.reply
    assert reply.usage_metadata and reply.usage_metadata["total_tokens"] == 15

    chunks = list(llm.stream("Count."))
    assert len(chunks) > 1 and "".join(str(c.content) for c in chunks).strip() == primary.reply

    primary.tool_queue = [("get_weather", {"city": "Berlin"})]
    called = llm.bind_tools([get_weather]).invoke("Weather in Berlin?")
    assert called.tool_calls[0]["name"] == "get_weather"  # type: ignore[attr-defined]
    assert called.tool_calls[0]["args"] == {"city": "Berlin"}  # type: ignore[attr-defined]
    assert [t.function.name for t in primary.seen_requests[-1].tools] == ["get_weather"]

    primary.tool_queue = [("Weather", {"city": "Dhaka"})]
    structured = llm.with_structured_output(Weather, method="function_calling")
    assert structured.invoke("Weather in Dhaka?") == Weather(city="Dhaka")


def test_langgraph_agent_feeds_tool_results_back(
    client: TestClient, auth_headers: dict[str, str], primary: FakeProvider
) -> None:
    key = auth_headers["Authorization"].removeprefix("Bearer ")
    primary.tool_queue = [("get_weather", {"city": "Berlin"})]
    agent = create_react_agent(_llm(client, key), [get_weather])
    state = agent.invoke({"messages": [HumanMessage("Weather in Berlin?")]})
    tool_message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert tool_message.content == "Sunny in Berlin"
    assert state["messages"][-1].content == "Tool result: Sunny in Berlin"


def test_end_user_tags_and_virtual_keys_flow_through_langchain(
    client: TestClient, admin_headers: dict[str, str], primary: FakeProvider
) -> None:
    """What an app sets on its ChatOpenAI reaches the gateway's controls."""
    secret, record = create_virtual_key(client, admin_headers, allowed_models=["test-model"])
    llm = _llm(
        client,
        secret,
        model_kwargs={"user": "student-7"},
        extra_body={"aigw": {"tags": ["iit-hub"], "no_cache": True}},
    )
    llm.invoke("hi")
    seen = primary.seen_requests[-1]
    assert seen.user == "student-7"
    logs = client.get(
        "/admin/api/logs", params={"key_id": record["id"]}, headers=admin_headers
    ).json()["items"]
    assert logs and logs[0]["status"] == "success"


def test_langchain_embeddings(client: TestClient, auth_headers: dict[str, str]) -> None:
    key = auth_headers["Authorization"].removeprefix("Bearer ")
    embedder = OpenAIEmbeddings(
        model="embed-model",
        base_url="http://testserver/v1",
        api_key=key,  # type: ignore[arg-type]
        http_client=client,
        check_embedding_ctx_length=False,
        max_retries=0,
    )
    vectors = embedder.embed_documents(["alpha", "beta"])
    assert len(vectors) == 2 and len(vectors[0]) > 0
    assert embedder.embed_query("alpha") == vectors[0]
