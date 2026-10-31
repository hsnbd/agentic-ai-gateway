from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.core.errors import ErrorCode, ProviderError
from app.core.schemas import (
    ChatRequest,
    EmbeddingRequest,
    FinishReason,
    ImagePart,
    Message,
    Role,
    TextPart,
    ToolCall,
    ToolDef,
)
from app.providers.base import Deployment
from app.providers.ollama import OllamaProvider


@pytest.fixture
def deployment() -> Deployment:
    return Deployment(
        id="ollama/llama3.1",
        model_name="llama3.1",
        provider="ollama",
        provider_model="llama3.1:8b",
        api_key=None,
        base_url="http://localhost:11434",
    )


@pytest.mark.asyncio
@respx.mock
async def test_chat_translation_and_response_parsing(deployment: Deployment) -> None:
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "Hello!"},
                "done_reason": "stop",
                "prompt_eval_count": 5,
                "eval_count": 2,
            },
        )
    )
    async with httpx.AsyncClient() as client:
        result = await OllamaProvider(client).chat(
            ChatRequest(model="llama3.1", messages=[Message(role=Role.USER, content="Hi")]),
            deployment,
        )

    sent = json.loads(route.calls[0].request.content)
    assert sent == {
        "model": "llama3.1:8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }
    assert result.text == "Hello!"
    assert result.choices[0].finish_reason == FinishReason.STOP
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 2


@pytest.mark.asyncio
@respx.mock
async def test_sampling_options_and_no_default_authorization(deployment: Deployment) -> None:
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})
    )
    async with httpx.AsyncClient() as client:
        await OllamaProvider(client).chat(
            ChatRequest(
                model="llama3.1",
                messages=[Message(role=Role.USER, content="test")],
                temperature=0.2,
                top_p=0.8,
                max_tokens=64,
                stop=["END"],
                seed=4,
                presence_penalty=0.1,
                frequency_penalty=0.3,
            ),
            deployment,
        )

    sent = json.loads(route.calls[0].request.content)
    assert sent["options"] == {
        "temperature": 0.2,
        "top_p": 0.8,
        "num_predict": 64,
        "stop": ["END"],
        "seed": 4,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.3,
    }
    assert "authorization" not in route.calls[0].request.headers


@pytest.mark.asyncio
@respx.mock
async def test_a_configured_context_window_is_sent_as_num_ctx(deployment: Deployment) -> None:
    # Ollama otherwise loads the model with a small default context and
    # silently drops the start of longer prompts.
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})
    )
    windowed = deployment.model_copy(update={"default_params": {"num_ctx": 32768}})
    async with httpx.AsyncClient() as client:
        await OllamaProvider(client).chat(
            ChatRequest(model="llama3.1", messages=[Message(role=Role.USER, content="test")]),
            windowed,
        )

    assert json.loads(route.calls[0].request.content)["options"] == {"num_ctx": 32768}


@pytest.mark.asyncio
@respx.mock
async def test_tool_call_request_and_response_translation(deployment: Deployment) -> None:
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "lookup", "arguments": {"q": "x"}}}],
                },
                "done_reason": "stop",
            },
        )
    )
    tool = ToolDef.model_validate(
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Find a value",
                "parameters": {"type": "object", "properties": {"q": {}}},
            },
        }
    )
    async with httpx.AsyncClient() as client:
        result = await OllamaProvider(client).chat(
            ChatRequest(
                model="llama3.1",
                messages=[
                    Message(
                        role=Role.ASSISTANT,
                        tool_calls=[ToolCall(name="lookup", arguments='{"q":"x"}')],
                    )
                ],
                tools=[tool],
            ),
            deployment,
        )

    sent = json.loads(route.calls[0].request.content)
    assert sent["messages"][0]["tool_calls"] == [
        {"function": {"name": "lookup", "arguments": {"q": "x"}}}
    ]
    assert sent["tools"][0]["function"]["name"] == "lookup"
    assert result.choices[0].finish_reason == FinishReason.TOOL_CALLS
    assert result.tool_calls[0].arguments == '{"q": "x"}'


@pytest.mark.asyncio
@respx.mock
async def test_image_data_uri_is_sent_as_raw_base64(deployment: Deployment) -> None:
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"role": "assistant", "content": "seen"}})
    )
    async with httpx.AsyncClient() as client:
        await OllamaProvider(client).chat(
            ChatRequest(
                model="llama3.1",
                messages=[
                    Message(
                        role=Role.USER,
                        content=[
                            TextPart(text="describe"),
                            ImagePart(url="data:image/png;base64,aGVsbG8="),
                        ],
                    )
                ],
            ),
            deployment,
        )

    sent = json.loads(route.calls[0].request.content)
    assert sent["messages"] == [{"role": "user", "content": "describe", "images": ["aGVsbG8="]}]


@pytest.mark.asyncio
@respx.mock
async def test_jsonl_stream_yields_content_and_final_usage(deployment: Deployment) -> None:
    body = (
        "\n".join(
            [
                json.dumps({"message": {"role": "assistant", "content": "Hel"}, "done": False}),
                json.dumps({"message": {"role": "assistant", "content": "lo"}, "done": False}),
                json.dumps(
                    {
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                        "done_reason": "stop",
                        "prompt_eval_count": 3,
                        "eval_count": 2,
                    }
                ),
            ]
        )
        + "\n"
    )
    respx.post("http://localhost:11434/api/chat").mock(return_value=httpx.Response(200, text=body))
    async with httpx.AsyncClient() as client:
        chunks = [
            chunk
            async for chunk in OllamaProvider(client).stream(
                ChatRequest(
                    model="llama3.1",
                    messages=[Message(role=Role.USER, content="Hi")],
                ),
                deployment,
            )
        ]

    assert "".join(chunk.content or "" for chunk in chunks) == "Hello"
    final = chunks[-1]
    assert final.finish_reason == FinishReason.STOP
    assert final.usage is not None
    assert final.usage.prompt_tokens == 3
    assert final.usage.completion_tokens == 2


@pytest.mark.asyncio
@respx.mock
async def test_stream_emits_tool_call_deltas(deployment: Deployment) -> None:
    body = (
        json.dumps(
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "lookup", "arguments": {"q": "x"}}}],
                },
                "done": True,
                "done_reason": "stop",
            }
        )
        + "\n"
    )
    respx.post("http://localhost:11434/api/chat").mock(return_value=httpx.Response(200, text=body))
    async with httpx.AsyncClient() as client:
        chunks = [
            chunk
            async for chunk in OllamaProvider(client).stream(
                ChatRequest(
                    model="llama3.1",
                    messages=[Message(role=Role.USER, content="use tool")],
                ),
                deployment,
            )
        ]

    assert chunks[0].tool_calls[0].name == "lookup"
    assert chunks[0].tool_calls[0].arguments == '{"q": "x"}'
    assert chunks[0].tool_calls[0].id is not None
    assert chunks[-1].finish_reason == FinishReason.TOOL_CALLS


@pytest.mark.asyncio
@respx.mock
async def test_tool_calls_in_separate_stream_lines_get_their_own_index(
    deployment: Deployment,
) -> None:
    def line(name: str) -> str:
        call = {"function": {"name": name, "arguments": {}}}
        return json.dumps({"message": {"role": "assistant", "tool_calls": [call]}}) + "\n"

    body = line("first") + line("second") + json.dumps({"done": True}) + "\n"
    respx.post("http://localhost:11434/api/chat").mock(return_value=httpx.Response(200, text=body))
    request = ChatRequest(model="llama3.1", messages=[Message(role=Role.USER, content="go")])
    async with httpx.AsyncClient() as client:
        chunks = [chunk async for chunk in OllamaProvider(client).stream(request, deployment)]

    calls = [call for chunk in chunks for call in chunk.tool_calls]
    assert [(call.index, call.name) for call in calls] == [(0, "first"), (1, "second")]
    assert len({call.id for call in calls}) == 2


_BASH_TOOL = ToolDef.model_validate(
    {
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "number"},
                    "workdir": {"type": ["string", "null"]},
                    "env": {"anyOf": [{"type": "object"}, {"type": "null"}]},
                    "shell": {"type": "string", "nullable": True},
                    "description": {"type": "string"},
                },
                "required": ["command", "description"],
            },
        }
    }
)
_NULLY_CALL = {
    "function": {
        "name": "bash",
        "arguments": {
            "command": "ls",
            "description": None,
            "timeout": None,
            "workdir": None,
            "env": None,
            "shell": None,
            "unknown": None,
        },
    }
}


@pytest.mark.asyncio
@respx.mock
async def test_null_optional_tool_arguments_are_dropped(deployment: Deployment) -> None:
    # llama3.2 fills skipped optional parameters with null; strict clients reject that.
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"role": "assistant", "tool_calls": [_NULLY_CALL]}}
        )
    )
    request = ChatRequest(
        model="llama3.1", messages=[Message(role=Role.USER, content="go")], tools=[_BASH_TOOL]
    )
    async with httpx.AsyncClient() as client:
        result = await OllamaProvider(client).chat(request, deployment)

    [call] = result.choices[0].message.tool_calls
    # Required and nullable parameters keep their null; the rest are omitted.
    assert json.loads(call.arguments) == {
        "command": "ls",
        "description": None,
        "workdir": None,
        "env": None,
        "shell": None,
    }


@pytest.mark.asyncio
@respx.mock
async def test_null_optional_tool_arguments_are_dropped_when_streaming(
    deployment: Deployment,
) -> None:
    unlisted = {"function": {"name": "other", "arguments": {"x": None}}}
    line = {"message": {"role": "assistant", "tool_calls": [_NULLY_CALL, unlisted]}}
    body = json.dumps(line) + "\n" + json.dumps({"done": True}) + "\n"
    respx.post("http://localhost:11434/api/chat").mock(return_value=httpx.Response(200, text=body))
    request = ChatRequest(
        model="llama3.1", messages=[Message(role=Role.USER, content="go")], tools=[_BASH_TOOL]
    )
    async with httpx.AsyncClient() as client:
        chunks = [chunk async for chunk in OllamaProvider(client).stream(request, deployment)]

    bash, other = (call for chunk in chunks for call in chunk.tool_calls)
    assert "timeout" not in json.loads(bash.arguments or "{}")
    # A tool the request did not declare is passed through untouched.
    assert json.loads(other.arguments or "{}") == {"x": None}


@pytest.mark.asyncio
@respx.mock
async def test_embedding_batch_order_and_legacy_shape(deployment: Deployment) -> None:
    route = respx.post("http://localhost:11434/api/embed").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "embeddings": [[0.1, 0.2], [0.3, 0.4]],
                    "prompt_eval_count": 7,
                },
            ),
            httpx.Response(200, json={"embedding": [0.5, 0.6]}),
        ]
    )
    async with httpx.AsyncClient() as client:
        provider = OllamaProvider(client)
        batch = await provider.embed(
            EmbeddingRequest(model="nomic-embed-text", input=["first", "second"]),
            deployment,
        )
        legacy = await provider.embed(
            EmbeddingRequest(model="nomic-embed-text", input=["legacy"]), deployment
        )

    assert [item.embedding for item in batch.data] == [[0.1, 0.2], [0.3, 0.4]]
    assert [item.index for item in batch.data] == [0, 1]
    assert batch.usage.prompt_tokens == 7
    assert [(item.index, item.embedding) for item in legacy.data] == [(0, [0.5, 0.6])]
    assert json.loads(route.calls[0].request.content) == {
        "model": "llama3.1:8b",
        "input": ["first", "second"],
    }


@pytest.mark.asyncio
@respx.mock
async def test_health_check_reflects_http_status(deployment: Deployment) -> None:
    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=[httpx.Response(200), httpx.Response(503)]
    )
    async with httpx.AsyncClient() as client:
        provider = OllamaProvider(client)
        assert await provider.health_check(deployment) is True
        assert await provider.health_check(deployment) is False


@pytest.mark.asyncio
@respx.mock
async def test_model_not_found_error_recommends_pull(deployment: Deployment) -> None:
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(404, json={"error": "model 'llama3.1:8b' not found"})
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as raised:
            await OllamaProvider(client).chat(
                ChatRequest(
                    model="llama3.1",
                    messages=[Message(role=Role.USER, content="Hi")],
                ),
                deployment,
            )

    assert raised.value.code == ErrorCode.NOT_FOUND
    assert "ollama pull llama3.1:8b" in raised.value.message


@pytest.mark.asyncio
@respx.mock
async def test_gateway_only_fields_are_not_forwarded(deployment: Deployment) -> None:
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})
    )
    async with httpx.AsyncClient() as client:
        await OllamaProvider(client).chat(
            ChatRequest(
                model="llama3.1",
                messages=[Message(role=Role.USER, content="test")],
                no_cache=True,
                fallbacks=["other"],
                routing_strategy="priority",
                guardrail_policy="strict",
                tags=["internal"],
                metadata={"trace": "hidden"},
                cache_ttl=30,
            ),
            deployment,
        )

    sent = json.loads(route.calls[0].request.content)
    assert set(sent) == {"model", "messages", "stream"}
    assert not {
        "no_cache",
        "fallbacks",
        "routing_strategy",
        "guardrail_policy",
        "tags",
        "metadata",
        "cache_ttl",
    }.intersection(sent)


# What Ollama returns when the model fails mid-run, e.g. out of GPU memory.
_UNFINISHED = {"model": "", "message": {"role": "", "content": ""}, "done": False}


@pytest.mark.asyncio
@respx.mock
async def test_an_unfinished_reply_is_a_clear_provider_error(deployment: Deployment) -> None:
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=_UNFINISHED)
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError, match="Ollama server log") as caught:
            await OllamaProvider(client).chat(
                ChatRequest(model="llama3.1", messages=[Message(role=Role.USER, content="Hi")]),
                deployment,
            )
    assert caught.value.code == ErrorCode.PROVIDER_UNAVAILABLE


@pytest.mark.asyncio
@respx.mock
async def test_an_error_in_the_body_is_reported(deployment: Deployment) -> None:
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"error": "model runner has unexpectedly stopped"})
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError, match="unexpectedly stopped"):
            await OllamaProvider(client).chat(
                ChatRequest(model="llama3.1", messages=[Message(role=Role.USER, content="Hi")]),
                deployment,
            )


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([_UNFINISHED], "Ollama server log"),
        ([{"message": {"role": "assistant", "content": "par"}}], "Ollama server log"),
        ([{"error": "out of memory"}], "out of memory"),
    ],
)
async def test_a_stream_that_never_finishes_is_an_error(
    deployment: Deployment, lines: list[dict[str, object]], message: str
) -> None:
    body = "".join(json.dumps(line) + "\n" for line in lines)
    respx.post("http://localhost:11434/api/chat").mock(return_value=httpx.Response(200, text=body))
    request = ChatRequest(
        model="llama3.1", messages=[Message(role=Role.USER, content="Hi")], stream=True
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError, match=message):
            async for _ in OllamaProvider(client).stream(request, deployment):
                pass
