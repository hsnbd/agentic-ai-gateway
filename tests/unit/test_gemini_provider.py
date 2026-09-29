from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.core.errors import ErrorCode, ProviderError
from app.core.schemas import (
    ChatRequest,
    EmbeddingRequest,
    FunctionDef,
    Message,
    Role,
    ToolCall,
    ToolChoice,
    ToolDef,
)
from app.providers.base import Deployment
from app.providers.gemini import GeminiProvider

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
CHAT_URL = f"{BASE_URL}/models/gemini-2.0-flash:generateContent"
STREAM_URL = f"{BASE_URL}/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
EMBED_URL = f"{BASE_URL}/models/gemini-2.0-flash:embedContent"


@pytest.fixture
def deployment() -> Deployment:
    return Deployment(
        id="gemini/flash",
        model_name="gemini-2-flash",
        provider="gemini",
        provider_model="gemini-2.0-flash",
        api_key="gm-test",
        base_url=BASE_URL,
    )


def make_provider() -> tuple[GeminiProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient()
    return GeminiProvider(client), client


async def test_chat_translates_system_role_and_function_schema(
    deployment: Deployment,
) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.post(CHAT_URL).mock(
            return_value=httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
            )
        )
        provider, client = make_provider()
        try:
            request = ChatRequest(
                model="gemini-2-flash",
                messages=[
                    Message(role=Role.SYSTEM, content="Be helpful"),
                    Message(role=Role.USER, content="Question"),
                    Message(role=Role.ASSISTANT, content="Answer"),
                ],
                tools=[
                    ToolDef(
                        function=FunctionDef(
                            name="lookup",
                            description="Look up a value",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "$ref": "#/defs/query"}
                                },
                                "additionalProperties": False,
                                "$schema": "draft-07",
                                "$defs": {"query": {"type": "string"}},
                                "exclusiveMinimum": 1,
                            },
                        )
                    )
                ],
                tool_choice=ToolChoice(mode="function", function_name="lookup"),
                temperature=0.2,
                top_p=0.8,
                max_tokens=64,
                stop=["END"],
                seed=3,
                n=2,
                response_format={"type": "json_object"},
            )
            response = await provider.chat(request, deployment)
        finally:
            await client.aclose()

    payload = json.loads(route.calls[0].request.content)
    assert route.calls[0].request.headers["x-goog-api-key"] == "gm-test"
    assert payload["systemInstruction"] == {"parts": [{"text": "Be helpful"}]}
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "Question"}]},
        {"role": "model", "parts": [{"text": "Answer"}]},
    ]
    declaration = payload["tools"][0]["functionDeclarations"][0]
    assert declaration["parameters"] == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    }
    assert payload["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["lookup"],
    }
    assert payload["generationConfig"] == {
        "temperature": 0.2,
        "topP": 0.8,
        "maxOutputTokens": 64,
        "seed": 3,
        "stopSequences": ["END"],
        "candidateCount": 2,
        "responseMimeType": "application/json",
    }
    assert response.text == "ok"


async def test_chat_tool_call_response_has_tool_finish_reason(deployment: Deployment) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.post(CHAT_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "candidates": [{
                        "content": {"parts": [{
                            "functionCall": {"name": "lookup", "args": {"id": 7}}
                        }]},
                        "finishReason": "STOP",
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 4,
                        "candidatesTokenCount": 2,
                        "totalTokenCount": 6,
                        "cachedContentTokenCount": 1,
                    },
                },
            )
        )
        provider, client = make_provider()
        try:
            result = await provider.chat(
                ChatRequest(model="gemini-2-flash", messages=[Message(role=Role.USER, content="go")]),
                deployment,
            )
        finally:
            await client.aclose()
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "lookup"
    assert json.loads(result.tool_calls[0].arguments) == {"id": 7}
    assert result.choices[0].finish_reason.value == "tool_calls"
    assert result.usage.prompt_tokens == 4
    assert result.usage.cached_tokens == 1


async def test_tool_result_resolves_name_from_prior_call(deployment: Deployment) -> None:
    with respx.mock(assert_all_called=True) as router:
        route = router.post(CHAT_URL).mock(
            return_value=httpx.Response(200, json={"candidates": []})
        )
        provider, client = make_provider()
        try:
            request = ChatRequest(
                model="gemini-2-flash",
                messages=[
                    Message(role=Role.ASSISTANT, tool_calls=[
                        ToolCall(id="call-1", name="search", arguments='{"term":"x"}')
                    ]),
                    Message(role=Role.TOOL, tool_call_id="call-1", content="found"),
                ],
            )
            await provider.chat(request, deployment)
        finally:
            await client.aclose()
    payload = json.loads(route.calls[0].request.content)
    assert payload["contents"] == [
        {"role": "model", "parts": [{"functionCall": {"name": "search", "args": {"term": "x"}}}]},
        {"role": "user", "parts": [{
            "functionResponse": {"name": "search", "response": {"result": "found"}}
        }]},
    ]


async def test_stream_yields_text_deltas_and_final_usage(deployment: Deployment) -> None:
    body = (
        'data: {"candidates":[{"content":{"parts":[{"text":"Hello "}]}}]}\n\n'
        'data: {"candidates":[{"content":{"parts":[{"text":"world"}]},"finishReason":"STOP"}],'
        '"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}\n\n'
    )
    with respx.mock(assert_all_called=True) as router:
        router.post(STREAM_URL).mock(
            return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
        )
        provider, client = make_provider()
        try:
            chunks = [
                chunk async for chunk in provider.stream(
                    ChatRequest(model="gemini-2-flash", messages=[Message(role=Role.USER, content="hi")]),
                    deployment,
                )
            ]
        finally:
            await client.aclose()
    assert [chunk.content for chunk in chunks if chunk.content] == ["Hello ", "world"]
    assert any(chunk.usage and chunk.usage.total_tokens == 5 for chunk in chunks)
    assert any(chunk.finish_reason and chunk.finish_reason.value == "stop" for chunk in chunks)


async def test_embeddings_preserve_input_order_and_send_dimensions(
    deployment: Deployment,
) -> None:
    seen: list[dict[str, object]] = []

    def response(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        text = body["content"]["parts"][0]["text"]
        vector = [1.0, 0.0] if text == "first" else [0.0, 1.0]
        return httpx.Response(200, json={"embedding": {"values": vector}})

    with respx.mock(assert_all_called=True) as router:
        router.post(EMBED_URL).mock(side_effect=response)
        provider, client = make_provider()
        try:
            result = await provider.embed(
                EmbeddingRequest(model="gemini-2-flash", input=["first", "second"], dimensions=2),
                deployment,
            )
        finally:
            await client.aclose()
    assert [entry.index for entry in result.data] == [0, 1]
    assert [entry.embedding for entry in result.data] == [[1.0, 0.0], [0.0, 1.0]]
    assert all(body["outputDimensionality"] == 2 for body in seen)
    assert all(body["model"] == "models/gemini-2.0-flash" for body in seen)


@pytest.mark.parametrize(
    ("status", "code"),
    [(429, ErrorCode.PROVIDER_RATE_LIMIT), (503, ErrorCode.PROVIDER_UNAVAILABLE)],
)
async def test_chat_maps_http_errors(
    deployment: Deployment, status: int, code: ErrorCode
) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.post(CHAT_URL).mock(return_value=httpx.Response(status, json={"error": "failure"}))
        provider, client = make_provider()
        try:
            with pytest.raises(ProviderError) as raised:
                await provider.chat(
                    ChatRequest(
                        model="gemini-2-flash",
                        messages=[Message(role=Role.USER, content="hello")],
                    ),
                    deployment,
                )
        finally:
            await client.aclose()
    assert raised.value.code == code
