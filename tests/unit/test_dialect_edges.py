"""Validation and encoding edge cases for the OpenAI and Anthropic wire dialects."""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.core.errors import ErrorCode, GatewayError, InvalidRequestError, ProviderError
from app.core.schemas import (
    ChatResponse,
    Choice,
    FinishReason,
    ImagePart,
    Message,
    Role,
    StreamChunk,
    TextPart,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from app.dialects.anthropic_dialect import AnthropicDialect
from app.dialects.openai_dialect import OpenAIDialect

USER = {"role": "user", "content": "hi"}


# -- OpenAI -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"model": "m"}, "'messages' must be an array"),
        ({"model": "", "messages": []}, "'model' must be a non-empty string"),
        ({"model": "m", "messages": [USER], "tool_choice": {"type": "x"}}, "Unsupported"),
        ({"model": "m", "messages": [USER], "tool_choice": 5}, "Unsupported tool_choice value"),
        ({"model": "m", "messages": [USER], "tool_choice": {"type": "function"}}, "Invalid"),
        ({"model": "m", "messages": [USER], "tool_choice": "sometimes"}, "Invalid tool_choice"),
        ({"model": "m", "messages": [USER], "temperature": "hot"}, "Invalid OpenAI chat"),
        ({"model": "m", "messages": ["not-an-object"]}, "Each message must be an object"),
        ({"model": "m", "messages": [{"role": "user", "content": 5}]}, "string, array, or null"),
        ({"model": "m", "messages": [{"role": "user", "content": ["x"]}]}, "must be objects"),
        (
            {"model": "m", "messages": [{"role": "user", "content": [{"type": "audio"}]}]},
            "Unsupported content part",
        ),
        (
            {"model": "m", "messages": [{"role": "user", "content": [{"type": "text"}]}]},
            "Invalid content parts",
        ),
        ({"model": "m", "messages": [{"role": "wizard", "content": "x"}]}, "Invalid message"),
    ],
)
def test_openai_rejects_malformed_requests(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidRequestError, match=message):
        OpenAIDialect().decode_chat(payload)


def test_openai_decodes_string_image_urls_and_data_urls() -> None:
    request = OpenAIDialect().decode_chat(
        {
            "model": "m",
            "stop": "END",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": "https://example.com/a.png"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                    ],
                }
            ],
        }
    )
    first, second = request.messages[0].content  # type: ignore[misc]
    assert isinstance(first, ImagePart) and first.media_type is None
    assert isinstance(second, ImagePart) and second.media_type == "image/png"
    assert request.stop == ["END"]


def test_openai_message_encoding_includes_parts_names_and_tool_ids() -> None:
    parts = Message(
        role=Role.USER,
        content=[TextPart(text="look"), ImagePart(url="https://example.com/a.png")],
        name="alice",
    )
    encoded = OpenAIDialect._message(parts)
    assert encoded["name"] == "alice"
    assert encoded["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/a.png", "detail": "auto"},
    }
    tool = OpenAIDialect._message(Message(role=Role.TOOL, content="42", tool_call_id="call_1"))
    assert tool["tool_call_id"] == "call_1"


def test_openai_usage_only_chunk_without_include_usage_emits_nothing() -> None:
    chunk = StreamChunk(model="m", usage=Usage(total_tokens=3))
    assert OpenAIDialect().encode_chunk(chunk, {"include_usage": False}) == []


# -- Anthropic: request validation ------------------------------------------


def _anthropic(**fields: Any) -> dict[str, Any]:
    return {"model": "claude", "max_tokens": 10, "messages": [USER], **fields}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"model": "claude", "messages": []}, "'max_tokens' is required"),
        (_anthropic(max_tokens=True), "positive integer"),
        (_anthropic(max_tokens=0), "positive integer"),
        (_anthropic(model=""), "'model' must be a non-empty string"),
        (_anthropic(messages="nope"), "'messages' must be an array"),
        (_anthropic(messages=["nope"]), "Each message must be an object"),
        (_anthropic(messages=[{"role": "system", "content": "x"}]), "'user' or 'assistant'"),
        (_anthropic(metadata="nope"), "'metadata' must be an object"),
        (_anthropic(temperature="hot"), "Invalid Anthropic chat request"),
        (_anthropic(system=5), "'system' must be a string"),
        (_anthropic(messages=[{"role": "user", "content": 5}]), "string or array"),
        (_anthropic(messages=[{"role": "user", "content": ["x"]}]), "blocks must be objects"),
        (
            _anthropic(messages=[{"role": "user", "content": [{"type": "audio"}]}]),
            "Unsupported content block",
        ),
        (
            _anthropic(messages=[{"role": "assistant", "content": [{"type": "tool_use"}]}]),
            "require string 'id' and 'name'",
        ),
        (
            _anthropic(
                messages=[
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "t", "name": "n", "input": []}],
                    }
                ]
            ),
            "input must be a JSON object",
        ),
        (
            _anthropic(messages=[{"role": "user", "content": [{"type": "tool_result"}]}]),
            "require string 'tool_use_id'",
        ),
        (
            _anthropic(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "t", "content": 5}],
                    }
                ]
            ),
            "string or array",
        ),
        (
            _anthropic(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "t", "content": [1]}],
                    }
                ]
            ),
            "content blocks must be objects",
        ),
        (
            _anthropic(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t",
                                "content": [{"type": "audio"}],
                            }
                        ],
                    }
                ]
            ),
            "Unsupported tool-result content block",
        ),
        (
            _anthropic(messages=[{"role": "user", "content": [{"type": "image"}]}]),
            "require a source object",
        ),
        (
            _anthropic(
                messages=[
                    {"role": "user", "content": [{"type": "image", "source": {"type": "base64"}}]}
                ]
            ),
            "require media_type and data",
        ),
        (
            _anthropic(
                messages=[
                    {"role": "user", "content": [{"type": "image", "source": {"type": "file"}}]}
                ]
            ),
            "base64 or url",
        ),
        (_anthropic(tools="nope"), "'tools' must be an array"),
        (_anthropic(tools=["nope"]), "must be objects"),
        (_anthropic(tools=[{"name": ""}]), "non-empty name"),
        (_anthropic(tool_choice="auto"), "'tool_choice' must be an object"),
        (_anthropic(tool_choice={"type": "tool"}), "Unsupported tool_choice"),
    ],
)
def test_anthropic_rejects_malformed_requests(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidRequestError, match=message):
        AnthropicDialect().decode_chat(payload)


def test_anthropic_decodes_rich_content() -> None:
    request = AnthropicDialect().decode_chat(
        _anthropic(
            system=[{"type": "text", "text": "be "}, "skip", {"type": "text", "text": "brief"}],
            metadata={"user_id": "u1"},
            tools=[{"name": "lookup"}],
            tool_choice={"type": "none"},
            messages=[
                {"role": "user", "content": None},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is"},
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.com/a.png"},
                        },
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "AA"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "t2",
                            "content": [
                                {"type": "text", "text": "result"},
                                {
                                    "type": "image",
                                    "source": {"type": "url", "url": "https://example.com/b.png"},
                                },
                            ],
                        },
                        {"type": "tool_result", "tool_use_id": "t3", "content": []},
                    ],
                },
            ],
        )
    )
    assert request.messages[0].content == "be brief"
    assert request.user == "u1"
    assert request.tool_choice is not None and request.tool_choice.mode == "none"
    assert request.tools[0].function.parameters == {"type": "object", "properties": {}}
    roles = [message.role for message in request.messages]
    assert roles == [Role.SYSTEM, Role.USER, Role.USER, Role.TOOL, Role.TOOL, Role.TOOL]
    image = request.messages[2].content[2]  # type: ignore[index]
    assert isinstance(image, ImagePart) and image.url == "data:image/png;base64,AA"
    assert request.messages[3].content is None
    assert request.messages[5].content is None


@pytest.mark.parametrize(
    ("choice", "mode"),
    [
        ({"type": "auto"}, "auto"),
        ({"type": "any"}, "required"),
        ({"type": "tool", "name": "x"}, "function"),
    ],
)
def test_anthropic_tool_choice_modes(choice: dict[str, Any], mode: str) -> None:
    request = AnthropicDialect().decode_chat(_anthropic(tool_choice=choice))
    assert request.tool_choice is not None and request.tool_choice.mode == mode


# -- Anthropic: response encoding -------------------------------------------


def _anthropic_response(message: Message, **fields: Any) -> ChatResponse:
    return ChatResponse(model="claude", choices=[Choice(message=message)], **fields)


def test_anthropic_encodes_parts_images_and_gateway_extras() -> None:
    message = Message(
        role=Role.ASSISTANT,
        content=[
            TextPart(text="see"),
            ImagePart(url="data:image/png;base64,AAA"),
            ImagePart(url="data:;base64,BBB"),
            ImagePart(url="data:nocomma"),
            ImagePart(url="https://example.com/a.png"),
        ],
    )
    body = AnthropicDialect().encode_chat(
        _anthropic_response(
            message, sources=[{"id": "s"}], stop_reason="max_iterations", tool_calls_executed=2
        )
    )
    blocks = body["content"]
    assert blocks[0] == {"type": "text", "text": "see"}
    assert blocks[1]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAA"}
    assert blocks[2]["source"]["media_type"] == "application/octet-stream"
    assert blocks[3]["source"] == {"type": "url", "url": "data:nocomma"}
    assert blocks[4]["source"] == {"type": "url", "url": "https://example.com/a.png"}
    assert body["aigw"] == {
        "sources": [{"id": "s"}],
        "stop_reason": "max_iterations",
        "tool_calls_executed": 2,
    }


def test_anthropic_encodes_empty_responses() -> None:
    body = AnthropicDialect().encode_chat(ChatResponse(model="claude", choices=[]))
    assert body["content"] == []
    assert body["stop_reason"] == "end_turn"
    assert "aigw" not in body


@pytest.mark.parametrize("arguments", ["not json", "[1, 2]"])
def test_anthropic_rejects_non_object_tool_arguments(arguments: str) -> None:
    message = Message(
        role=Role.ASSISTANT, tool_calls=[ToolCall(id="t", name="n", arguments=arguments)]
    )
    with pytest.raises(InvalidRequestError, match="JSON object"):
        AnthropicDialect().encode_chat(_anthropic_response(message))


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (InvalidRequestError("x"), "invalid_request_error"),
        (ProviderError(ErrorCode.PROVIDER_OVERLOADED, "x"), "overloaded_error"),
        (GatewayError(ErrorCode.RAG_UNAVAILABLE, "x"), "api_error"),
    ],
)
def test_anthropic_error_types(error: GatewayError, kind: str) -> None:
    assert AnthropicDialect().encode_error(error)["error"]["type"] == kind


# -- Anthropic: streaming ---------------------------------------------------


def _events(frames: list[str]) -> list[dict[str, Any]]:
    return [json.loads(frame.split("data: ", 1)[1]) for frame in frames]


def test_anthropic_stream_interleaves_text_and_tool_blocks() -> None:
    dialect = AnthropicDialect()
    state: dict[str, Any] = {}
    frames = dialect.encode_stream_start("msg_1", "claude", state)
    frames += dialect.encode_chunk(StreamChunk(model="claude", content="Hel"), state)
    frames += dialect.encode_chunk(StreamChunk(model="claude", content="lo"), state)
    frames += dialect.encode_chunk(
        StreamChunk(model="claude", tool_calls=[ToolCallDelta(index=0, name="a")]), state
    )
    frames += dialect.encode_chunk(
        StreamChunk(model="claude", tool_calls=[ToolCallDelta(index=1, id="t_b", name="b")]),
        state,
    )
    # Back to tool 0: its block is reopened as the active one and gets late metadata.
    frames += dialect.encode_chunk(
        StreamChunk(
            model="claude",
            tool_calls=[ToolCallDelta(index=0, id="t_a", name="a2", arguments="{}")],
        ),
        state,
    )
    frames += dialect.encode_chunk(
        StreamChunk(model="claude", finish_reason=FinishReason.TOOL_CALLS), state
    )
    frames += dialect.encode_stream_end(state)

    events = _events(frames)
    types = [event["type"] for event in events]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_stop",
        "content_block_start",
        "content_block_stop",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    tool_a = events[5]["content_block"]
    assert tool_a == {"type": "tool_use", "id": "toolu_0", "name": "a", "input": {}}
    assert events[9]["index"] == 1
    assert state["tool_ids"][0] == "t_a" and state["tool_names"][0] == "a2"
    assert events[11]["delta"]["stop_reason"] == "tool_use"


def test_anthropic_stream_end_emits_message_delta_when_missing() -> None:
    dialect = AnthropicDialect()
    state: dict[str, Any] = {}
    dialect.encode_stream_start("msg_1", "claude", state)
    dialect.encode_chunk(StreamChunk(model="claude", content="x"), state)
    types = [event["type"] for event in _events(dialect.encode_stream_end(state))]
    assert types == ["content_block_stop", "message_delta", "message_stop"]
