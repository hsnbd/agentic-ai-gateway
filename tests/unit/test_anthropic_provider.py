from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from app.core.errors import ErrorCode, ProviderError
from app.core.schemas import (
    ChatRequest,
    ImagePart,
    Message,
    Role,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolDef,
)
from app.providers.anthropic import AnthropicProvider
from app.providers.base import Deployment


@pytest.fixture
async def deployment() -> Deployment:
    return Deployment(
        id="anthropic/claude",
        model_name="claude-sonnet-4",
        provider="anthropic",
        provider_model="claude-sonnet-4-20250514",
        api_key="sk-ant-test",
        base_url="https://api.anthropic.com/v1",
    )


@pytest.fixture
async def provider() -> AsyncIterator[AnthropicProvider]:
    async with httpx.AsyncClient() as client:
        yield AnthropicProvider(client)


@respx.mock
async def test_hoists_system_and_defaults_max_tokens(
    provider: AnthropicProvider, deployment: Deployment
) -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg-1",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )
    )
    request = ChatRequest(
        model="claude-sonnet-4",
        messages=[
            Message(role=Role.SYSTEM, content="system one"),
            Message(role=Role.USER, content="hello"),
            Message(role=Role.SYSTEM, content="system two"),
        ],
        seed=42,
        presence_penalty=0.2,
        n=3,
    )

    await provider.chat(request, deployment)

    sent = json.loads(route.calls.last.request.content)
    assert sent["system"] == "system one\n\nsystem two"
    assert sent["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert sent["max_tokens"] == 4096
    assert "seed" not in sent and "presence_penalty" not in sent and "n" not in sent
    assert route.calls.last.request.headers["x-api-key"] == "sk-ant-test"
    assert route.calls.last.request.headers["anthropic-version"] == "2023-06-01"


@respx.mock
async def test_max_tokens_defaults_to_deployment_limit(
    provider: AnthropicProvider, deployment: Deployment
) -> None:
    deployment.capabilities.max_output_tokens = 8192
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={"content": [], "usage": {"input_tokens": 0, "output_tokens": 0}},
        )
    )

    await provider.chat(ChatRequest(model="claude-sonnet-4", messages=[]), deployment)

    assert json.loads(route.calls.last.request.content)["max_tokens"] == 8192


@respx.mock
async def test_tool_use_response_and_usage_translation(
    provider: AnthropicProvider, deployment: Deployment
) -> None:
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg-tool",
                "content": [
                    {"type": "text", "text": "Searching"},
                    {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"q": "x"}},
                ],
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cache_read_input_tokens": 3,
                },
            },
        )
    )

    response = await provider.chat(ChatRequest(model="claude-sonnet-4", messages=[]), deployment)

    assert response.text == "Searching"
    assert response.choices[0].finish_reason.value == "tool_calls"
    assert response.tool_calls[0].id == "tool-1"
    assert response.tool_calls[0].name == "lookup"
    assert json.loads(response.tool_calls[0].arguments) == {"q": "x"}
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 4
    assert response.usage.total_tokens == 14
    assert response.usage.cached_tokens == 3


@respx.mock
async def test_tool_results_and_adjacent_roles_are_merged(
    provider: AnthropicProvider, deployment: Deployment
) -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={"content": [], "usage": {"input_tokens": 0, "output_tokens": 0}},
        )
    )
    request = ChatRequest(
        model="claude-sonnet-4",
        messages=[
            Message(role=Role.USER, content="first"),
            Message(role=Role.USER, content="second"),
            Message(role=Role.ASSISTANT, content="working"),
            Message(role=Role.TOOL, tool_call_id="tool-1", content="result one"),
            Message(role=Role.TOOL, tool_call_id="tool-2", content="result two"),
            Message(role=Role.USER, content="next"),
        ],
    )

    await provider.chat(request, deployment)

    messages = json.loads(route.calls.last.request.content)["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert [block["text"] for block in messages[0]["content"]] == ["first", "second"]
    results = [block for block in messages[2]["content"] if block["type"] == "tool_result"]
    assert [block["tool_use_id"] for block in results] == ["tool-1", "tool-2"]
    assert messages[2]["content"][-1] == {"type": "text", "text": "next"}


@respx.mock
async def test_tools_tool_choice_and_assistant_tool_calls_are_translated(
    provider: AnthropicProvider, deployment: Deployment
) -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={"content": [], "usage": {"input_tokens": 0, "output_tokens": 0}},
        )
    )
    request = ChatRequest(
        model="claude-sonnet-4",
        messages=[
            Message(
                role=Role.ASSISTANT,
                tool_calls=[ToolCall(id="call-1", name="lookup", arguments='{"key": 1}')],
            )
        ],
        tools=[ToolDef(function={"name": "lookup", "description": "Find", "parameters": {"type": "object"}})],
        tool_choice=ToolChoice(mode="function", function_name="lookup"),
    )

    await provider.chat(request, deployment)

    sent = json.loads(route.calls.last.request.content)
    assert sent["tools"] == [
        {"name": "lookup", "description": "Find", "input_schema": {"type": "object"}}
    ]
    assert sent["tool_choice"] == {"type": "tool", "name": "lookup"}
    assert sent["messages"][0]["content"] == [
        {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"key": 1}}
    ]


@respx.mock
async def test_image_content_blocks_support_data_and_http_urls(
    provider: AnthropicProvider, deployment: Deployment
) -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={"content": [], "usage": {"input_tokens": 0, "output_tokens": 0}},
        )
    )
    request = ChatRequest(
        model="claude-sonnet-4",
        messages=[
            Message(
                role=Role.USER,
                content=[
                    TextPart(text="look"),
                    ImagePart(url="data:image/png;base64,aGVsbG8="),
                    ImagePart(url="https://example.com/image.png"),
                ],
            )
        ],
    )

    await provider.chat(request, deployment)

    content = json.loads(route.calls.last.request.content)["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "aGVsbG8=",
    }
    assert content[2]["source"] == {"type": "url", "url": "https://example.com/image.png"}


@respx.mock
async def test_stream_translates_text_tool_json_deltas_and_final_usage(
    provider: AnthropicProvider, deployment: Deployment
) -> None:
    events = [
        ('message_start', {"message": {"id": "msg-stream", "usage": {"input_tokens": 8, "cache_read_input_tokens": 2}}}),
        ('content_block_start', {"index": 0, "content_block": {"type": "text", "text": ""}}),
        ('content_block_delta', {"index": 0, "delta": {"type": "text_delta", "text": "hello"}}),
        ('content_block_stop', {"index": 0}),
        ('content_block_start', {"index": 1, "content_block": {"type": "tool_use", "id": "tool-2", "name": "lookup", "input": {}}}),
        ('content_block_delta', {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"q":'}}),
        ('content_block_delta', {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '"x"}'}}),
        ('content_block_stop', {"index": 1}),
        ('message_delta', {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
        ('message_stop', {}),
    ]
    sse = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, text=sse)
    )

    chunks = [
        chunk
        async for chunk in provider.stream(
            ChatRequest(model="claude-sonnet-4", messages=[], stream=True), deployment
        )
    ]

    assert any(chunk.content == "hello" for chunk in chunks)
    tool_chunks = [chunk.tool_calls[0] for chunk in chunks if chunk.tool_calls]
    assert tool_chunks[0].index == 1
    assert tool_chunks[0].id == "tool-2" and tool_chunks[0].name == "lookup"
    assert "".join(delta.arguments or "" for delta in tool_chunks[1:]) == '{"q":"x"}'
    final = chunks[-1]
    assert final.finish_reason.value == "tool_calls"
    assert final.usage.prompt_tokens == 8
    assert final.usage.completion_tokens == 5
    assert final.usage.total_tokens == 13
    assert final.usage.cached_tokens == 2
    assert all(chunk.id == "msg-stream" for chunk in chunks)


@pytest.mark.parametrize(
    ("status", "code"),
    [(429, ErrorCode.PROVIDER_RATE_LIMIT), (529, ErrorCode.PROVIDER_OVERLOADED)],
)
@respx.mock
async def test_http_errors_are_mapped_and_retryable(
    status: int,
    code: ErrorCode,
    provider: AnthropicProvider,
    deployment: Deployment,
) -> None:
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(status, json={"error": {"message": "try again"}})
    )

    with pytest.raises(ProviderError) as error:
        await provider.chat(ChatRequest(model="claude-sonnet-4", messages=[]), deployment)

    assert error.value.code == code
    assert error.value.retryable is True
