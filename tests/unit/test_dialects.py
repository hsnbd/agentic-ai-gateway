"""Client dialect compatibility without external services."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import chat, messages
from app.core.errors import InvalidRequestError
from app.core.schemas import (
    ChatResponse,
    Choice,
    FinishReason,
    Message,
    Role,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from app.dialects.anthropic_dialect import AnthropicDialect
from app.dialects.openai_dialect import OpenAIDialect


def _response(reason: FinishReason = FinishReason.TOOL_CALLS) -> ChatResponse:
    return ChatResponse(
        id="msg_123",
        model="test-model",
        created=12,
        choices=[
            Choice(
                message=Message(
                    role=Role.ASSISTANT,
                    content="Hello",
                    tool_calls=[ToolCall(id="call_1", name="read", arguments='{"path":"foo"}')],
                ),
                finish_reason=reason,
            )
        ],
        usage=Usage.of(10, 5),
        provider="openai",
        cache_hit=True,
        latency_ms=1.5,
        cost_usd=0.01,
    )


def _events(frames: list[str]) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    for frame in frames:
        assert frame.endswith("\n\n")
        lines = frame.strip().split("\n")
        result.append((lines[0].removeprefix("event: "), json.loads(lines[-1][6:])))
    return result


@pytest.mark.parametrize("content", ["hello", [{"type": "text", "text": "hello"}]])
def test_openai_content_forms(content: Any) -> None:
    request = OpenAIDialect().decode_chat(
        {"model": "m", "messages": [{"role": "user", "content": content}]}
    )
    assert request.messages[0].text() == "hello"


def test_openai_image_and_parameters() -> None:
    request = OpenAIDialect().decode_chat(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}}
                    ],
                }
            ],
            "max_tokens": 100,
            "max_completion_tokens": 40,
            "stop": "STOP",
            "n": 2,
            "response_format": {"type": "json_schema", "json_schema": {"name": "result"}},
            "stream_options": {"include_usage": True},
            "aigw": {"no_cache": True},
            "metadata": {"tags": ["agent"]},
            "cache_ttl": 60,
        }
    )
    assert request.max_tokens == 40
    assert request.stop == ["STOP"]
    assert request.requires_vision()
    assert request.no_cache and request.cache_ttl == 60 and request.tags == ["agent"]
    assert request.n == 2
    assert request.response_format == {"type": "json_schema", "json_schema": {"name": "result"}}


def test_openai_tools_round_trip() -> None:
    dialect = OpenAIDialect()
    request = dialect.decode_chat(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path":"foo"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "read"}},
        }
    )
    assert request.tool_choice is not None
    assert request.tool_choice.function_name == "read"
    assert request.tools[0].function.name == "read"
    assert request.messages[1].tool_call_id == "call_1"
    encoded = dialect.encode_chat(_response())
    assert encoded["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == (
        '{"path":"foo"}'
    )
    assert encoded["choices"][0]["finish_reason"] == "tool_calls"
    assert encoded["object"] == "chat.completion"
    assert encoded["aigw"]["cache_hit"] is True
    assert encoded["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


@pytest.mark.parametrize(
    ("system", "expected"),
    [
        ("You are helpful", "You are helpful"),
        (
            [{"type": "text", "text": "You are "}, {"type": "text", "text": "helpful"}],
            "You are helpful",
        ),
    ],
)
def test_anthropic_system_hoisting(system: Any, expected: str) -> None:
    request = AnthropicDialect().decode_chat(
        {
            "model": "claude",
            "max_tokens": 32,
            "system": system,
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"user_id": "bob"},
            "top_k": 5,
        }
    )
    assert request.messages[0].role == Role.SYSTEM
    assert request.messages[0].text() == expected
    assert request.user == "bob"
    assert request.model_extra is not None
    assert request.model_extra["top_k"] == 5


def test_anthropic_tool_object_both_directions() -> None:
    dialect = AnthropicDialect()
    request = dialect.decode_chat(
        {
            "model": "claude",
            "max_tokens": 20,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "read",
                            "input": {"path": "foo"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "ok"}],
                },
            ],
            "tools": [{"name": "read", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "read"},
        }
    )
    assert json.loads(request.messages[0].tool_calls[0].arguments) == {"path": "foo"}
    assert request.messages[1].role == Role.TOOL
    assert request.messages[1].tool_call_id == "call_1"
    assert request.tool_choice is not None
    assert request.tool_choice.function_name == "read"
    encoded = dialect.encode_chat(_response())
    assert encoded["content"][1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "read",
        "input": {"path": "foo"},
    }
    assert encoded["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_anthropic_max_tokens_required() -> None:
    with pytest.raises(InvalidRequestError) as error:
        AnthropicDialect().decode_chat({"model": "claude", "messages": []})
    assert error.value.status_code == 400


@pytest.mark.parametrize(
    ("reason", "openai", "anthropic"),
    [
        (FinishReason.STOP, "stop", "end_turn"),
        (FinishReason.LENGTH, "length", "max_tokens"),
        (FinishReason.TOOL_CALLS, "tool_calls", "tool_use"),
        (FinishReason.CONTENT_FILTER, "content_filter", "refusal"),
        (FinishReason.ERROR, "error", "end_turn"),
    ],
)
def test_finish_reason_mapping(reason: FinishReason, openai: str, anthropic: str) -> None:
    assert OpenAIDialect().encode_chat(_response(reason))["choices"][0]["finish_reason"] == openai
    assert AnthropicDialect().encode_chat(_response(reason))["stop_reason"] == anthropic


def test_anthropic_named_event_order_and_tool_json() -> None:
    dialect = AnthropicDialect()
    state: dict[str, Any] = {}
    frames = dialect.encode_stream_start("msg_1", "claude", state)
    frames += dialect.encode_chunk(StreamChunk(content="Hi"), state)
    frames += dialect.encode_chunk(
        StreamChunk(
            tool_calls=[ToolCallDelta(index=0, id="call_1", name="read", arguments='{"path":')]
        ),
        state,
    )
    frames += dialect.encode_chunk(
        StreamChunk(tool_calls=[ToolCallDelta(index=0, arguments='"foo"}')]), state
    )
    frames += dialect.encode_chunk(
        StreamChunk(finish_reason=FinishReason.TOOL_CALLS, usage=Usage.of(10, 5)), state
    )
    frames += dialect.encode_stream_end(state)
    events = _events(frames)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[5][1]["delta"] == {"type": "input_json_delta", "partial_json": '{"path":'}
    assert events[8][1]["delta"]["stop_reason"] == "tool_use"
    assert events[8][1]["usage"]["output_tokens"] == 5


def test_openai_stream_shape_and_done() -> None:
    dialect = OpenAIDialect()
    state: dict[str, Any] = {"include_usage": True}
    frames = dialect.encode_stream_start("chat_1", "m", state)
    frames += dialect.encode_chunk(
        StreamChunk(id="chat_1", model="m", role=Role.ASSISTANT, content="hi"), state
    )
    frames += dialect.encode_chunk(
        StreamChunk(
            id="chat_1",
            model="m",
            tool_calls=[ToolCallDelta(index=0, id="call_1", name="read", arguments="{")],
        ),
        state,
    )
    frames += dialect.encode_chunk(
        StreamChunk(
            id="chat_1", model="m", finish_reason=FinishReason.TOOL_CALLS, usage=Usage.of(1, 2)
        ),
        state,
    )
    frames += dialect.encode_stream_end(state)
    assert frames[-1] == "data: [DONE]\n\n"
    body = json.loads(frames[0][6:])
    assert body["object"] == "chat.completion.chunk"
    assert body["choices"][0]["delta"] == {"role": "assistant", "content": "hi"}
    assert json.loads(frames[-2][6:])["choices"] == []
    assert json.loads(frames[-2][6:])["usage"]["total_tokens"] == 3


def test_error_envelopes() -> None:
    error = InvalidRequestError("Bad request")
    assert OpenAIDialect().encode_error(error)["error"]["type"] == "invalid_request"
    assert AnthropicDialect().encode_error(error) == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "Bad request"},
    }


def test_routes_stream_preflight_and_midstream_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubPipeline:
        async def run_stream(self, ctx: Any) -> Any:
            if ctx.request.model == "before":
                raise InvalidRequestError("before first chunk")
            yield StreamChunk(id="chat_1", model="m", content="hi")
            raise InvalidRequestError("after first chunk")

    class StubState:
        def __init__(self) -> None:
            self.pipeline = StubPipeline()

        def require_pipeline(self) -> Any:
            return self.pipeline

    monkeypatch.setattr(chat, "get_state", lambda app: StubState())
    app = FastAPI()
    app.include_router(chat.router)
    app.include_router(messages.router)
    client = TestClient(app)
    bad = client.post(
        "/v1/chat/completions", json={"model": "before", "messages": [], "stream": True}
    )
    assert bad.status_code == 400
    assert bad.json()["error"]["message"] == "before first chunk"
    streamed = client.post(
        "/v1/chat/completions", json={"model": "after", "messages": [], "stream": True}
    )
    assert streamed.status_code == 200
    assert streamed.headers["x-accel-buffering"] == "no"
    assert "after first chunk" in streamed.text
    assert streamed.text.endswith("data: [DONE]\n\n")
    anthropic = client.post(
        "/v1/messages", json={"model": "after", "max_tokens": 10, "messages": [], "stream": True}
    )
    assert "event: error\n" in anthropic.text


def test_routes_unary_models_and_legacy_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubPipeline:
        async def run(self, ctx: Any) -> ChatResponse:
            assert ctx.request.metadata["api_key"] == "secret"
            return _response(FinishReason.STOP)

        async def run_stream(self, ctx: Any) -> Any:
            yield StreamChunk(id="chat_1", model="m", role=Role.ASSISTANT, content="Hi")
            yield StreamChunk(id="chat_1", model="m", finish_reason=FinishReason.STOP)

    class Deployment:
        provider = "openai"

    class Registry:
        def list_models(self) -> list[str]:
            return ["m"]

        def deployments_for(self, model: str) -> list[Deployment]:
            assert model == "m"
            return [Deployment()]

    class StubState:
        def __init__(self) -> None:
            self.pipeline = StubPipeline()
            self.registry = Registry()

        def require_pipeline(self) -> Any:
            return self.pipeline

    monkeypatch.setattr(chat, "get_state", lambda app: StubState())
    app = FastAPI()
    app.include_router(chat.router)
    app.include_router(messages.router)
    client = TestClient(app)
    result = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret"},
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert result.status_code == 200
    assert result.json()["object"] == "chat.completion"
    assert result.headers["x-gateway-provider"] == "openai"
    assert result.headers["x-gateway-cache"] == "miss"
    anthropic = client.post(
        "/v1/messages",
        headers={"x-api-key": "secret"},
        json={"model": "m", "max_tokens": 10, "messages": []},
    )
    assert anthropic.status_code == 200
    assert anthropic.json()["type"] == "message"
    assert client.get("/v1/models").json()["data"][0]["id"] == "m"
    assert client.get("/v1/models/m").json()["object"] == "model"
    legacy = client.post("/v1/completions", json={"model": "m", "prompt": "hi", "stream": True})
    assert legacy.status_code == 200
    chunks = [
        json.loads(line[6:]) for line in legacy.text.splitlines() if line.startswith("data: {")
    ]
    assert chunks[0]["object"] == "text_completion"
    assert chunks[0]["choices"][0]["text"] == "Hi"
    assert legacy.text.endswith("data: [DONE]\n\n")


def test_invalid_payloads_and_anthropic_count_tokens() -> None:
    with pytest.raises(InvalidRequestError):
        OpenAIDialect().decode_chat({"model": "m", "messages": [], "tool_choice": "bad"})
    with pytest.raises(InvalidRequestError):
        OpenAIDialect().decode_chat(
            {
                "model": "m",
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
            }
        )
    with pytest.raises(InvalidRequestError):
        AnthropicDialect().decode_chat(
            {
                "model": "claude",
                "max_tokens": 5,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "call_1", "name": "read", "input": "invalid"}
                        ],
                    }
                ],
            }
        )
    app = FastAPI()
    app.include_router(chat.router)
    app.include_router(messages.router)
    client = TestClient(app)
    invalid = client.post("/v1/messages", content="{", headers={"content-type": "application/json"})
    assert invalid.status_code == 400
    assert invalid.json()["error"]["type"] == "invalid_request_error"
    counted = client.post(
        "/v1/messages/count_tokens",
        json={"model": "claude", "messages": [{"role": "user", "content": "Hello there"}]},
    )
    assert counted.status_code == 200
    assert counted.json()["input_tokens"] > 0
