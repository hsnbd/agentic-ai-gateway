from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx

from app.core.errors import ErrorCode, ProviderError
from app.core.schemas import (
    ChatRequest,
    EmbeddingRequest,
    FunctionDef,
    ImagePart,
    Message,
    Role,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolDef,
)
from app.providers.base import Deployment
from app.providers.openai import OpenAIProvider


@pytest.fixture
def deployment() -> Deployment:
    return Deployment(
        id="openai/gpt-4o",
        model_name="gpt-4o",
        provider="openai",
        provider_model="gpt-4o",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
    )


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_chat_translation_and_response(
    client: httpx.AsyncClient, deployment: Deployment
) -> None:
    with respx.mock(base_url="https://api.openai.com/v1") as router:
        route = router.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "model": "gpt-4o",
                    "created": 123,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                },
            )
        )
        request = ChatRequest(
            model="public-model",
            messages=[Message(role=Role.USER, content="Hi")],
            temperature=0.3,
            top_p=0.8,
            max_tokens=20,
            seed=7,
            presence_penalty=0.1,
            frequency_penalty=0.2,
            stop=["END"],
            n=2,
            user="user-1",
        )
        result = await OpenAIProvider(client).chat(request, deployment)

    sent = route.calls[0].request
    body = json.loads(sent.content)
    assert sent.headers["Authorization"] == "Bearer sk-test"
    assert body == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.3,
        "top_p": 0.8,
        "max_tokens": 20,
        "seed": 7,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "stop": ["END"],
        "n": 2,
        "user": "user-1",
    }
    assert result.text == "Hello"
    assert result.choices[0].finish_reason.value == "stop"
    assert result.usage.prompt_tokens == 5
    assert result.usage.total_tokens == 7
    assert result.provider == "openai"
    assert result.deployment_id == deployment.id
    assert result.latency_ms is not None


@pytest.mark.asyncio
async def test_usage_maps_cache_and_reasoning_tokens(
    client: httpx.AsyncClient, deployment: Deployment
) -> None:
    with respx.mock(base_url="https://api.openai.com/v1") as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 8,
                        "total_tokens": 18,
                        "prompt_tokens_details": {"cached_tokens": 4},
                        "completion_tokens_details": {"reasoning_tokens": 3},
                    },
                },
            )
        )
        result = await OpenAIProvider(client).chat(
            ChatRequest(model="gpt-4o", messages=[Message(role=Role.USER, content="Hi")]),
            deployment,
        )
    assert result.usage.cached_tokens == 4
    assert result.usage.reasoning_tokens == 3


@pytest.mark.asyncio
async def test_tool_call_translation_and_parsing(
    client: httpx.AsyncClient, deployment: Deployment
) -> None:
    with respx.mock(base_url="https://api.openai.com/v1") as router:
        route = router.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Paris"}',
                                },
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }]
                },
            )
        )
        request = ChatRequest(
            model="gpt-4o",
            messages=[
                Message(
                    role=Role.ASSISTANT,
                    content=None,
                    tool_calls=[ToolCall(id="call-0", name="weather", arguments="{}")],
                ),
                Message(role=Role.TOOL, content="sunny", tool_call_id="call-0"),
            ],
            tools=[ToolDef(function=FunctionDef(
                name="weather",
                description="Get weather",
                parameters={"type": "object", "properties": {}},
            ))],
            tool_choice=ToolChoice(mode="function", function_name="weather"),
        )
        result = await OpenAIProvider(client).chat(request, deployment)

    body = json.loads(route.calls[0].request.content)
    assert body["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-0",
                "type": "function",
                "function": {"name": "weather", "arguments": "{}"},
            }],
        },
        {"role": "tool", "content": "sunny", "tool_call_id": "call-0"},
    ]
    assert body["tools"][0]["function"]["name"] == "weather"
    assert body["tool_choice"] == {"type": "function", "function": {"name": "weather"}}
    assert result.choices[0].finish_reason.value == "tool_calls"
    assert result.tool_calls[0].name == "weather"
    assert result.tool_calls[0].arguments == '{"city":"Paris"}'


@pytest.mark.asyncio
async def test_multimodal_image_translation(
    client: httpx.AsyncClient, deployment: Deployment
) -> None:
    with respx.mock(base_url="https://api.openai.com/v1") as router:
        route = router.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "seen"}}]},
            )
        )
        await OpenAIProvider(client).chat(
            ChatRequest(
                model="gpt-4o",
                messages=[Message(role=Role.USER, content=[
                    TextPart(text="Describe"),
                    ImagePart(url="https://example.com/image.png", detail="high"),
                ])],
            ),
            deployment,
        )
    body = json.loads(route.calls[0].request.content)
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "Describe"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png", "detail": "high"},
        },
    ]


@pytest.mark.asyncio
async def test_stream_tool_call_deltas_and_final_usage(
    client: httpx.AsyncClient, deployment: Deployment
) -> None:
    chunks = [
        {"id": "stream-1", "model": "gpt-4o", "choices": [{
            "index": 0, "delta": {"role": "assistant"},
        }]},
        {"id": "stream-1", "model": "gpt-4o", "choices": [{
            "index": 0, "delta": {"tool_calls": [{
                "index": 0,
                "id": "call-1",
                "function": {"name": "weather", "arguments": '{"city":'},
            }]},
        }]},
        {"id": "stream-1", "model": "gpt-4o", "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": '"Paris"}'},
            }]},
            "finish_reason": "tool_calls",
        }]},
        {"id": "stream-1", "model": "gpt-4o", "choices": [], "usage": {
            "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6,
        }},
    ]
    stream_body = "\n\n".join(f"data: {json.dumps(chunk)}" for chunk in chunks)
    with respx.mock(base_url="https://api.openai.com/v1") as router:
        route = router.post("/chat/completions").mock(return_value=httpx.Response(
            200, text=f": keepalive\n{stream_body}\ndata: [DONE]\n"
        ))
        provider = OpenAIProvider(client)
        request = ChatRequest(model="gpt-4o", messages=[Message(role=Role.USER, content="Hi")])
        iterator = provider.stream(request, deployment)
        assert hasattr(iterator, "__aiter__")
        received = [chunk async for chunk in iterator]

    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body["stream"] is True
    assert sent_body["stream_options"] == {"include_usage": True}
    tool_deltas = [delta for chunk in received for delta in chunk.tool_calls]
    assert tool_deltas[0].id == "call-1"
    assert tool_deltas[0].name == "weather"
    assert "".join(delta.arguments or "" for delta in tool_deltas) == '{"city":"Paris"}'
    assert received[-1].usage is not None
    assert received[-1].usage.total_tokens == 6
    assert received[-2].finish_reason is not None


@pytest.mark.asyncio
async def test_embeddings_are_sorted_by_index(
    client: httpx.AsyncClient, deployment: Deployment
) -> None:
    with respx.mock(base_url="https://api.openai.com/v1") as router:
        route = router.post("/embeddings").mock(return_value=httpx.Response(
            200,
            json={
                "model": "text-embedding-3-small",
                "data": [
                    {"index": 1, "embedding": [0.2, 0.3]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        ))
        result = await OpenAIProvider(client).embed(
            EmbeddingRequest(
                model="text-embedding-3-small", input=["a", "b"], dimensions=2
            ),
            deployment,
        )
    body = json.loads(route.calls[0].request.content)
    assert body == {"model": "gpt-4o", "input": ["a", "b"], "dimensions": 2}
    assert [vector.index for vector in result.data] == [0, 1]
    assert result.usage.prompt_tokens == 5
    assert result.provider == "openai"
    assert result.latency_ms is not None


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (429, {"error": {"message": "rate limited"}}, ErrorCode.PROVIDER_RATE_LIMIT),
        (401, {"error": {"message": "bad key"}}, ErrorCode.AUTHENTICATION_ERROR),
        (400, {"error": {"message": "maximum context length exceeded"}},
         ErrorCode.CONTEXT_LENGTH_EXCEEDED),
    ],
)
@pytest.mark.asyncio
async def test_http_errors_are_mapped(
    client: httpx.AsyncClient,
    deployment: Deployment,
    status: int,
    body: dict[str, Any],
    expected: ErrorCode,
) -> None:
    with respx.mock(base_url="https://api.openai.com/v1") as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(status, json=body)
        )
        with pytest.raises(ProviderError) as error:
            await OpenAIProvider(client).chat(
                ChatRequest(model="gpt-4o", messages=[Message(role=Role.USER, content="x")]),
                deployment,
            )
    assert error.value.code == expected
    if status == 429:
        assert error.value.retryable is True


@pytest.mark.asyncio
async def test_gateway_fields_are_not_forwarded(
    client: httpx.AsyncClient, deployment: Deployment
) -> None:
    with respx.mock(base_url="https://api.openai.com/v1") as router:
        route = router.post("/chat/completions").mock(return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        ))
        request = ChatRequest(
            model="gpt-4o",
            messages=[Message(role=Role.USER, content="x")],
            no_cache=True,
            fallbacks=["other"],
            routing_strategy="priority",
            guardrail_policy="strict",
            tags=["tag"],
            metadata={"key": "value"},
            cache_ttl=30,
        )
        await OpenAIProvider(client).chat(request, deployment)
    body = json.loads(route.calls[0].request.content)
    forbidden = {
        "no_cache", "fallbacks", "routing_strategy", "guardrail_policy",
        "tags", "metadata", "cache_ttl",
    }
    assert forbidden.isdisjoint(body)
