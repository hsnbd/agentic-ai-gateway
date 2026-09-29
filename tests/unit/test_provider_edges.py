"""Edge cases shared by, and specific to, the provider adapters and their registry."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from app.config.settings import Settings
from app.core.errors import ConfigurationError, ErrorCode, NotFoundError, ProviderError
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
from app.providers.anthropic import AnthropicProvider
from app.providers.base import Capabilities, Deployment, Pricing, Provider
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider
from app.providers.registry import ProviderRegistry, _expand_env

BASE = "https://llm.example"


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as async_client:
        yield async_client


def _deployment(provider: str = "openai", **fields: Any) -> Deployment:
    defaults: dict[str, Any] = {
        "id": f"{provider}/m",
        "model_name": "m",
        "provider": provider,
        "provider_model": "pm",
        "api_key": "key",
        "base_url": BASE,
    }
    return Deployment(**{**defaults, **fields})


def _request(*messages: Message, **fields: Any) -> ChatRequest:
    return ChatRequest(
        model="m", messages=list(messages) or [Message(role=Role.USER, content="hi")], **fields
    )


def _tool() -> ToolDef:
    return ToolDef(function=FunctionDef(name="lookup", parameters={"type": "object"}))


def _sse(*events: str) -> bytes:
    return "".join(events).encode()


async def _collect(stream: AsyncIterator[Any]) -> list[Any]:
    return [item async for item in stream]


# -- Base provider ----------------------------------------------------------


@pytest.mark.parametrize(
    ("capabilities", "request_fields", "supported"),
    [
        (Capabilities(streaming=False), {"stream": True}, False),
        (Capabilities(tools=False), {"tools": [_tool()]}, False),
        (Capabilities(json_mode=False), {"response_format": {"type": "json_object"}}, False),
        (Capabilities(max_output_tokens=10), {"max_tokens": 20}, False),
        (Capabilities(max_output_tokens=10), {"max_tokens": 5}, True),
    ],
)
def test_capabilities_filter_requests(
    capabilities: Capabilities, request_fields: dict[str, Any], supported: bool
) -> None:
    assert capabilities.supports(_request(**request_fields)) is supported


def test_vision_requires_capability() -> None:
    image = Message(role=Role.USER, content=[ImagePart(url="https://example.com/a.png")])
    assert not Capabilities(vision=False).supports(_request(image))
    assert Capabilities(vision=True).supports(_request(image))


def test_cached_tokens_fall_back_to_input_rate() -> None:
    assert Pricing(input_per_mtok=2.0).estimate(1_000_000, 0, 500_000) == pytest.approx(2.0)


def test_deployments_hash_by_id() -> None:
    assert hash(_deployment()) == hash(_deployment(model_name="other"))
    assert len({_deployment(), _deployment()}) == 1


class _Minimal(Provider):
    name = "minimal"

    async def chat(self, request: ChatRequest, deployment: Deployment) -> Any:
        raise NotImplementedError

    def stream(self, request: ChatRequest, deployment: Deployment) -> Any:
        raise NotImplementedError


async def test_base_provider_defaults(client: httpx.AsyncClient) -> None:
    provider = _Minimal(client)
    with pytest.raises(ProviderError, match="does not support embeddings"):
        await provider.embed(EmbeddingRequest(model="m", input=["x"]), _deployment())
    assert await provider.health_check(_deployment())
    with pytest.raises(ProviderError, match="has no base_url") as raised:
        provider._base_url(_deployment(base_url=None))
    assert raised.value.code is ErrorCode.CONFIGURATION_ERROR


def _status_error(status: int, body: Any = None, headers: dict[str, str] | None = None) -> Any:
    request = httpx.Request("POST", BASE)
    if isinstance(body, str):
        response = httpx.Response(status, text=body, headers=headers, request=request)
    else:
        response = httpx.Response(status, json=body, headers=headers, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ProviderError(ErrorCode.PROVIDER_ERROR, "already mapped"), ErrorCode.PROVIDER_ERROR),
        (httpx.ReadTimeout("slow"), ErrorCode.PROVIDER_TIMEOUT),
        (httpx.ConnectError("refused"), ErrorCode.PROVIDER_UNAVAILABLE),
        (ValueError("bad json"), ErrorCode.PROVIDER_ERROR),
        (_status_error(418, {"message": "teapot"}), ErrorCode.PROVIDER_ERROR),
        (_status_error(503, {"detail": "down"}), ErrorCode.PROVIDER_UNAVAILABLE),
        (_status_error(529, {"error": "overloaded"}), ErrorCode.PROVIDER_OVERLOADED),
        (
            _status_error(400, {"error": {"message": "maximum context length is 8k"}}),
            ErrorCode.CONTEXT_LENGTH_EXCEEDED,
        ),
    ],
)
def test_error_mapping(client: httpx.AsyncClient, error: Exception, code: ErrorCode) -> None:
    mapped = _Minimal(client).map_error(error, _deployment())
    assert mapped.code is code
    if isinstance(error, ProviderError):
        assert mapped is error


def test_retry_after_and_error_message_extraction(client: httpx.AsyncClient) -> None:
    provider = _Minimal(client)
    limited = provider.map_error(_status_error(429, {}, {"retry-after": "2.5"}), _deployment())
    assert limited.retry_after == 2.5
    unparseable = provider.map_error(
        _status_error(429, {}, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}), _deployment()
    )
    assert unparseable.retry_after is None
    assert (
        "plain failure"
        in provider.map_error(_status_error(500, "plain failure"), _deployment()).message
    )
    assert "[1, 2]" in provider.map_error(_status_error(500, [1, 2]), _deployment()).message
    assert (
        "{'error': 5}"
        in provider.map_error(_status_error(500, {"error": 5}), _deployment()).message
    )


# -- OpenAI -----------------------------------------------------------------


def test_openai_payload_covers_names_and_tool_choices(client: httpx.AsyncClient) -> None:
    provider = OpenAIProvider(client)
    named = Message(role=Role.USER, content="hi", name="alice")
    payload = provider._chat_payload(
        _request(named, tools=[_tool()], tool_choice=ToolChoice(mode="function")), _deployment()
    )
    assert payload["messages"][0]["name"] == "alice"
    assert payload["tool_choice"] == {"type": "function", "function": {}}
    auto = provider._chat_payload(_request(tool_choice=ToolChoice(mode="auto")), _deployment())
    assert auto["tool_choice"] == "auto"


async def test_openai_stream_skips_noise_and_ends_without_done(
    client: httpx.AsyncClient,
) -> None:
    body = _sse(
        ": keep-alive\n\n",
        "event: ping\n\n",
        "data: {not json\n\n",
        'data: {"id":"c","choices":[{"delta":{"content":"hi"}}]}\n\n',
    )
    with respx.mock(base_url=BASE) as router:
        router.post("/chat/completions").mock(return_value=httpx.Response(200, content=body))
        chunks = await _collect(OpenAIProvider(client).stream(_request(), _deployment()))
    assert [chunk.content for chunk in chunks] == ["hi"]


@pytest.mark.parametrize(
    ("provider_cls", "name", "path"),
    [
        (OpenAIProvider, "openai", r".*/chat/completions"),
        (AnthropicProvider, "anthropic", r".*/messages"),
        (GeminiProvider, "gemini", r".*streamGenerateContent.*"),
        (OllamaProvider, "ollama", r".*/api/chat"),
    ],
)
async def test_stream_http_errors_keep_upstream_detail(
    client: httpx.AsyncClient, provider_cls: type[Provider], name: str, path: str
) -> None:
    """Regression: a streamed error body was unread, so mapping it crashed."""
    with respx.mock(base_url=BASE) as router:
        router.post(url__regex=path).mock(
            return_value=httpx.Response(
                400, json={"error": {"message": "maximum context length exceeded"}}
            )
        )
        with pytest.raises(ProviderError) as raised:
            await _collect(provider_cls(client).stream(_request(), _deployment(name)))
    assert raised.value.code is ErrorCode.CONTEXT_LENGTH_EXCEEDED
    assert "maximum context length" in raised.value.message


async def test_openai_embeddings_with_user_and_errors(client: httpx.AsyncClient) -> None:
    with respx.mock(base_url=BASE) as router:
        route = router.post("/embeddings").mock(
            return_value=httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
        )
        result = await OpenAIProvider(client).embed(
            EmbeddingRequest(model="m", input=["x"], user="u"), _deployment()
        )
        assert json.loads(route.calls[0].request.content)["user"] == "u"
        assert result.usage.total_tokens == 0

        route.mock(return_value=httpx.Response(500, json={}))
        with pytest.raises(ProviderError):
            await OpenAIProvider(client).embed(
                EmbeddingRequest(model="m", input=["x"]), _deployment()
            )


@pytest.mark.parametrize(
    ("provider_cls", "path"),
    [
        (OpenAIProvider, "/models"),
        (GeminiProvider, "/models"),
        (OllamaProvider, "/api/tags"),
    ],
)
async def test_health_checks(
    client: httpx.AsyncClient, provider_cls: type[Provider], path: str
) -> None:
    provider = provider_cls(client)
    with respx.mock(base_url=BASE) as router:
        route = router.get(path).mock(return_value=httpx.Response(200))
        assert await provider.health_check(_deployment())
        route.mock(return_value=httpx.Response(503))
        assert not await provider.health_check(_deployment())
        route.mock(side_effect=httpx.ConnectError("down"))
        assert not await provider.health_check(_deployment())


# -- Anthropic --------------------------------------------------------------


def test_anthropic_content_blocks(client: httpx.AsyncClient) -> None:
    provider = AnthropicProvider(client)
    blocks = provider._content_blocks(
        [
            TextPart(text="look"),
            ImagePart(url="data:image/png,raw%20bytes"),
            ImagePart(url="data:;base64,QUJD"),
        ]
    )
    assert blocks[1]["source"]["data"] == "cmF3IGJ5dGVz"
    assert blocks[1]["source"]["media_type"] == "image/png"
    assert blocks[2]["source"]["media_type"] == "application/octet-stream"
    assert provider._content_blocks("") == []
    with pytest.raises(ValueError, match="Invalid data URI"):
        provider._content_blocks([ImagePart(url="data:nocomma")])
    with pytest.raises(ValueError, match="Unsupported image URI"):
        provider._content_blocks([ImagePart(url="ftp://example.com/a.png")])


def test_anthropic_request_body_options(client: httpx.AsyncClient) -> None:
    provider = AnthropicProvider(client)
    assistant = Message(
        role=Role.ASSISTANT,
        content="calling",
        tool_calls=[ToolCall(id="t1", name="lookup", arguments="not json")],
    )
    deployment = _deployment("anthropic", default_params={"stop": ["END"]})
    for mode, expected in [
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        ("function", None),
    ]:
        body = provider._request_body(
            _request(
                Message(role=Role.USER, content="hi"),
                assistant,
                tools=[_tool()],
                tool_choice=ToolChoice(mode=mode),
                temperature=0.2,
                top_p=0.9,
            ),
            deployment,
        )
        assert body.get("tool_choice") == expected
    assert body["messages"][1]["content"][1]["input"] == {}
    assert body["temperature"] == 0.2 and body["top_p"] == 0.9
    assert body["stop_sequences"] == ["END"]
    plain = provider._request_body(_request(tools=[_tool()]), _deployment("anthropic"))
    assert "tool_choice" not in plain and "stop_sequences" not in plain


async def test_anthropic_chat_ignores_unknown_blocks_and_maps_config_errors(
    client: httpx.AsyncClient,
) -> None:
    provider = AnthropicProvider(client)
    with respx.mock(base_url=BASE) as router:
        router.post("/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg",
                    "content": [
                        {"type": "thinking", "thinking": "..."},
                        {"type": "text", "text": "ok"},
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        )
        response = await provider.chat(_request(), _deployment("anthropic"))
    assert response.text == "ok"
    with pytest.raises(ProviderError) as raised:
        await provider.chat(_request(), _deployment("anthropic", base_url=None))
    assert raised.value.__cause__ is None


async def test_anthropic_stream_edge_events(client: httpx.AsyncClient) -> None:
    body = _sse(
        ": comment\n",
        "event: message_start\n",
        'data: {"message":{"id":"m1","usage":{"input_tokens":3}}}\n\n',
        "event: content_block_start\n",
        'data: {"index":0,"content_block":{"type":"text"}}\n\n',
        "event: content_block_delta\n",
        'data: {"index":0,"delta":{"type":"thinking_delta"}}\n\n',
        "event: message_delta\n",
        'data: {"delta":{},"usage":{}}\n\n',
        "event: ping\n",
        "data: {}",  # trailing event without a blank line, and no message_stop
    )
    with respx.mock(base_url=BASE) as router:
        router.post("/messages").mock(return_value=httpx.Response(200, content=body))
        chunks = await _collect(
            AnthropicProvider(client).stream(_request(), _deployment("anthropic"))
        )
    assert chunks[0].role == Role.ASSISTANT
    assert chunks[-1].usage.prompt_tokens == 3
    assert chunks[-1].finish_reason is not None


async def test_anthropic_stream_rejects_non_object_events(client: httpx.AsyncClient) -> None:
    with respx.mock(base_url=BASE) as router:
        router.post("/messages").mock(
            return_value=httpx.Response(
                200, content=_sse("event: message_start\n", "data: [1]\n\n")
            )
        )
        with pytest.raises(ProviderError, match="JSON object"):
            await _collect(AnthropicProvider(client).stream(_request(), _deployment("anthropic")))
    with pytest.raises(ProviderError) as raised:
        await _collect(
            AnthropicProvider(client).stream(_request(), _deployment("anthropic", base_url=None))
        )
    assert raised.value.code is ErrorCode.CONFIGURATION_ERROR


# -- Gemini -----------------------------------------------------------------


def test_gemini_schema_images_and_contents(client: httpx.AsyncClient) -> None:
    assert GeminiProvider._clean_schema({"anyOf": [{"$ref": "x", "type": "string"}]}) == {
        "anyOf": [{"type": "string"}]
    }
    assert GeminiProvider._image_part(ImagePart(url="data:image/png,raw")) == {
        "inlineData": {"mimeType": "image/png", "data": "cmF3"}
    }
    assert GeminiProvider._image_part(ImagePart(url="data:;base64,QUJD"))["inlineData"] == {
        "mimeType": "application/octet-stream",
        "data": "QUJD",
    }
    assert GeminiProvider._image_part(ImagePart(url="https://example.com/a.png")) == {
        "fileData": {"mimeType": "application/octet-stream", "fileUri": "https://example.com/a.png"}
    }
    with pytest.raises(ValueError, match="Malformed"):
        GeminiProvider._image_part(ImagePart(url="data:nocomma"))
    with pytest.raises(ValueError, match="data URI or an http"):
        GeminiProvider._image_part(ImagePart(url="ftp://example.com/a.png"))

    contents = GeminiProvider._contents(
        _request(
            Message(role=Role.USER, content=[TextPart(text="a"), ImagePart(url="https://x/a.png")]),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[ToolCall(id="c1", name="lookup", arguments="not json")],
            ),
            Message(role=Role.TOOL, content="42", tool_call_id="c1"),
            Message(role=Role.TOOL, content="?", tool_call_id="unknown", name="named"),
            Message(role=Role.USER, content=""),
        )
    )
    assert contents[1]["parts"] == [{"functionCall": {"name": "lookup", "args": {}}}]
    tool_parts = contents[2]["parts"]
    assert tool_parts[0]["functionResponse"]["name"] == "lookup"
    assert tool_parts[1]["functionResponse"]["name"] == "named"
    assert tool_parts[2] == {"text": ""}


def test_gemini_payload_generation_config(client: httpx.AsyncClient) -> None:
    payload = GeminiProvider(client)._payload(
        _request(
            tool_choice=ToolChoice(mode="function", function_name="lookup"),
            tools=[_tool()],
            stop=["END"],
            n=2,
            response_format={"type": "json_object"},
            seed=7,
        ),
        _deployment("gemini"),
    )
    assert payload["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["lookup"],
    }
    generation = payload["generationConfig"]
    assert generation["stopSequences"] == ["END"]
    assert generation["candidateCount"] == 2
    assert generation["responseMimeType"] == "application/json"
    assert generation["seed"] == 7
    bare = GeminiProvider(client)._payload(
        _request().model_copy(update={"n": 0}), _deployment("gemini")
    )
    assert "generationConfig" not in bare


def test_gemini_usage_and_finish_reasons() -> None:
    assert GeminiProvider._usage(None).total_tokens == 0
    assert GeminiProvider._finish_reason("MAX_TOKENS").value == "length"
    assert GeminiProvider._finish_reason("SAFETY").value == "content_filter"
    response = GeminiProvider._response({}, _request(), _deployment("gemini"))
    assert response.text == ""


async def test_gemini_stream_edge_cases(client: httpx.AsyncClient) -> None:
    body = _sse(
        "event: noise\n",
        'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"a","args":{"x":1}}},'
        '{"text":5}]}}]}\n\n',
        'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"a"}},'
        '{"inline":"x"}]}}]}\n\n',
        'data: {"candidates":[{"content":{"parts":[]}}]}\n\n',
        "\n",
        'data: {"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":3}}\n\n',
        "data: [DONE]",
    )
    with respx.mock(base_url=BASE) as router:
        router.post(url__regex=r".*streamGenerateContent.*").mock(
            return_value=httpx.Response(200, content=body)
        )
        chunks = await _collect(GeminiProvider(client).stream(_request(), _deployment("gemini")))
    first, second, finished, final = chunks
    assert first.tool_calls[0].index == second.tool_calls[0].index == 0
    assert first.tool_calls[0].id == second.tool_calls[0].id
    assert finished.finish_reason.value == "tool_calls"
    assert final.usage.total_tokens == 3


async def test_gemini_stream_trailing_event_and_nothing_to_finish(
    client: httpx.AsyncClient,
) -> None:
    with respx.mock(base_url=BASE) as router:
        route = router.post(url__regex=r".*streamGenerateContent.*")
        route.mock(return_value=httpx.Response(200, content=_sse('data: {"candidates":[]}')))
        assert (
            await _collect(GeminiProvider(client).stream(_request(), _deployment("gemini"))) == []
        )
        route.mock(
            return_value=httpx.Response(
                200, content=_sse('data: {"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}')
            )
        )
        chunks = await _collect(GeminiProvider(client).stream(_request(), _deployment("gemini")))
        assert [chunk.content for chunk in chunks] == ["hi"]
        route.mock(return_value=httpx.Response(500, json={}))
        with pytest.raises(ProviderError):
            await _collect(GeminiProvider(client).stream(_request(), _deployment("gemini")))


async def test_gemini_embeddings_with_dimensions(client: httpx.AsyncClient) -> None:
    with respx.mock(base_url=BASE) as router:
        route = router.post(url__regex=r".*embedContent").mock(
            return_value=httpx.Response(
                200,
                json={"embedding": {"values": [1.0]}, "usageMetadata": "not-a-dict"},
            )
        )
        result = await GeminiProvider(client).embed(
            EmbeddingRequest(model="m", input=["a"], dimensions=8), _deployment("gemini")
        )
        assert json.loads(route.calls[0].request.content)["outputDimensionality"] == 8
        assert result.usage.total_tokens == 0
        route.mock(return_value=httpx.Response(400, json={}))
        with pytest.raises(ProviderError):
            await GeminiProvider(client).embed(
                EmbeddingRequest(model="m", input=["a"]), _deployment("gemini")
            )


# -- Ollama -----------------------------------------------------------------


def test_ollama_message_payload(client: httpx.AsyncClient) -> None:
    payload = OllamaProvider._message_payload(
        Message(
            role=Role.USER,
            content=[
                TextPart(text="look"),
                ImagePart(url="data:image/png;base64,QUJD"),
                ImagePart(url="data:nocomma"),
                ImagePart(url="https://example.com/a.png"),
            ],
        )
    )
    assert payload == {"role": "user", "content": "look", "images": ["QUJD"]}
    assert (
        OllamaProvider._message_payload(Message(role=Role.ASSISTANT, content=None))["content"] == ""
    )
    with pytest.raises(ValueError, match="JSON object"):
        OllamaProvider._message_payload(
            Message(role=Role.ASSISTANT, tool_calls=[ToolCall(name="t", arguments="[1]")])
        )
    request = OllamaProvider(client)._request_payload(
        _request(stop=["END"], tools=[_tool()], response_format={"type": "json_object"}),
        _deployment("ollama", api_key=None),
    )
    assert request["options"] == {"stop": ["END"]}
    assert request["format"] == "json"
    assert "Authorization" not in OllamaProvider(client)._headers(
        _deployment("ollama", api_key=None)
    )


async def test_ollama_stream_skips_blank_lines(client: httpx.AsyncClient) -> None:
    body = (
        b"\n"
        + json.dumps({"message": {"content": ""}}).encode()
        + b"\n"
        + json.dumps({"message": {"content": "hi"}, "done": False}).encode()
        + b"\n"
    )
    with respx.mock(base_url=BASE) as router:
        router.post("/api/chat").mock(return_value=httpx.Response(200, content=body))
        chunks = await _collect(OllamaProvider(client).stream(_request(), _deployment("ollama")))
    assert [chunk.content for chunk in chunks] == ["hi"]


async def test_ollama_404_without_model_hint_and_missing_embeddings(
    client: httpx.AsyncClient,
) -> None:
    provider = OllamaProvider(client)
    with respx.mock(base_url=BASE) as router:
        router.post("/api/chat").mock(return_value=httpx.Response(404, json={"error": "no route"}))
        with pytest.raises(ProviderError) as raised:
            await provider.chat(_request(), _deployment("ollama"))
        assert "ollama pull" not in raised.value.message

        embed = router.post("/api/embed")
        embed.mock(return_value=httpx.Response(200, json={"embedding": [1.0]}))
        result = await provider.embed(
            EmbeddingRequest(model="m", input=["x"]), _deployment("ollama")
        )
        assert result.data[0].embedding == [1.0]
        embed.mock(return_value=httpx.Response(200, json={}))
        with pytest.raises(ProviderError, match="no embeddings"):
            await provider.embed(EmbeddingRequest(model="m", input=["x"]), _deployment("ollama"))


# -- Registry ---------------------------------------------------------------


def _registry(tmp_path: Path, yaml_text: str, **settings: Any) -> ProviderRegistry:
    path = tmp_path / "models.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    registry = ProviderRegistry(Settings(models_config_path=str(path), **settings))
    registry._register_providers(httpx.AsyncClient())
    registry.load_config(str(path))
    return registry


def test_env_expansion_in_nested_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIGW_TEST_KEY", "secret")
    monkeypatch.delenv("AIGW_TEST_MISSING", raising=False)
    assert _expand_env({"a": ["${AIGW_TEST_KEY}", "${AIGW_TEST_MISSING:-dflt}"], "b": 1}) == {
        "a": ["secret", "dflt"],
        "b": 1,
    }
    assert _expand_env("${AIGW_TEST_MISSING}") == ""


async def test_registry_lifecycle_and_lookups(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        """
model_list:
  - model_name: chat
    params: {provider: openai}
  - model_name: chat
    params: {provider: openai}
  - model_name: chat
    params: {provider: openai}
  - model_name: disabled
    id: explicit-off
    enabled: false
    params: {provider: ollama}
aliases:
  loop-a: loop-b
  loop-b: loop-a
""",
        encoding="utf-8",
    )
    registry = ProviderRegistry(Settings(models_config_path=str(path), openai_api_key="sk-x"))
    await registry.startup()
    try:
        assert [d.id for d in registry.list_deployments()] == [
            "openai/chat",
            "openai/chat#2",
            "openai/chat#3",
            "explicit-off",
        ]
        assert registry.get_deployment("openai/chat").api_key == "sk-x"
        assert registry.deployments_for("disabled") == []
        assert len(registry.deployments_for("disabled", include_disabled=True)) == 1
        assert [d.id for d in registry.deployments_for("openai/chat#2")] == ["openai/chat#2"]
        assert registry.resolve_alias("loop-a") in {"loop-a", "loop-b"}
        assert registry.list_models() == ["chat", "disabled", "loop-a", "loop-b"]
        assert registry.provider_names == ["anthropic", "gemini", "ollama", "openai"]
        assert registry.get_deployment("explicit-off").api_key is None
        with pytest.raises(NotFoundError):
            registry.deployments_for("missing")
        with pytest.raises(NotFoundError):
            registry.get_deployment("missing")
        with pytest.raises(ConfigurationError):
            registry.get_provider("missing")
    finally:
        await registry.shutdown()
        await registry.shutdown()


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("model_list:\n  - params: {provider: openai}\n", "needs model_name"),
        ("model_list:\n  - model_name: m\n    params: {provider: nope}\n", "Unknown provider"),
        (
            "model_list:\n  - {model_name: m, id: x, params: {provider: openai}}\n"
            "  - {model_name: n, id: x, params: {provider: openai}}\n",
            "Duplicate deployment id",
        ),
    ],
)
def test_registry_config_errors(tmp_path: Path, yaml_text: str, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        _registry(tmp_path, yaml_text)


def test_registry_missing_and_empty_config(tmp_path: Path) -> None:
    registry = ProviderRegistry(Settings())
    with pytest.raises(ConfigurationError, match="not found"):
        registry.load_config(str(tmp_path / "missing.yaml"))
    assert _registry(tmp_path, "").list_deployments() == []


def test_error_dict_without_message_falls_back_to_payload(client: httpx.AsyncClient) -> None:
    mapped = _Minimal(client).map_error(_status_error(500, {"error": {"code": 7}}), _deployment())
    assert "{'error': {'code': 7}}" in mapped.message


def test_gemini_function_choice_without_a_name(client: httpx.AsyncClient) -> None:
    payload = GeminiProvider(client)._payload(
        _request(tool_choice=ToolChoice(mode="function")), _deployment("gemini")
    )
    assert payload["toolConfig"]["functionCallingConfig"] == {"mode": "ANY"}


def test_ollama_length_finish_reason() -> None:
    assert OllamaProvider._finish_reason({"done_reason": "length"}).value == "length"


async def test_anthropic_trailing_message_stop_without_blank_line(
    client: httpx.AsyncClient,
) -> None:
    body = _sse("event: message_stop\n", "data: {}")
    with respx.mock(base_url=BASE) as router:
        router.post("/messages").mock(return_value=httpx.Response(200, content=body))
        chunks = await _collect(
            AnthropicProvider(client).stream(_request(), _deployment("anthropic"))
        )
    assert len(chunks) == 1 and chunks[0].finish_reason is not None
