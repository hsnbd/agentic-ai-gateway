"""Cross-provider conformance.

Four adapters were written independently against `Provider`. These tests pin
the parts of the contract the pipeline actually relies on, so a future adapter
cannot quietly diverge.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from app.core.schemas import ChatRequest, ChatResponse, Choice, Message, Role
from app.providers.anthropic import AnthropicProvider
from app.providers.base import Capabilities, Deployment, Pricing, Provider
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider

PROVIDER_CLASSES = [OpenAIProvider, AnthropicProvider, GeminiProvider, OllamaProvider]


@pytest.fixture
def client() -> httpx.AsyncClient:
    return httpx.AsyncClient()


@pytest.mark.parametrize("cls", PROVIDER_CLASSES, ids=lambda c: c.name)
def test_provider_declares_a_unique_name(cls: type[Provider]) -> None:
    assert isinstance(cls.name, str) and cls.name
    assert cls.name != "base"


def test_provider_names_are_unique() -> None:
    names = [c.name for c in PROVIDER_CLASSES]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("cls", PROVIDER_CLASSES, ids=lambda c: c.name)
def test_chat_embed_health_are_coroutines(cls: type[Provider]) -> None:
    for method in ("chat", "embed", "health_check"):
        assert inspect.iscoroutinefunction(getattr(cls, method)), (
            f"{cls.name}.{method} must be an async def"
        )


@pytest.mark.parametrize("cls", PROVIDER_CLASSES, ids=lambda c: c.name)
def test_stream_returns_an_async_iterator_without_awaiting(
    cls: type[Provider], client: httpx.AsyncClient
) -> None:
    """`Pipeline.run_stream` does `async for c in provider.stream(...)`.

    That works whether `stream` is an async-generator function or a plain
    function returning an async generator, but it breaks if someone makes it
    `async def` returning an iterator, because the call would need awaiting.
    """
    assert not inspect.iscoroutinefunction(cls.stream), (
        f"{cls.name}.stream must not be a coroutine function; "
        "return an AsyncIterator directly so callers can `async for` it"
    )

    provider = cls(client)
    request = ChatRequest(model="m", messages=[Message(role=Role.USER, content="hi")])
    result = provider.stream(request, _deployment(cls.name))

    assert isinstance(result, AsyncIterator), (
        f"{cls.name}.stream must return an AsyncIterator"
    )
    assert hasattr(result, "__anext__")


@pytest.mark.parametrize("cls", PROVIDER_CLASSES, ids=lambda c: c.name)
def test_stream_is_lazy(cls: type[Provider], client: httpx.AsyncClient) -> None:
    """No network call may happen until the first `__anext__`.

    The resilient executor relies on this: it only treats a failure as
    recoverable if nothing has been emitted yet, and it starts its
    time-to-first-token clock at iteration time, not at call time.
    """
    provider = cls(client)
    request = ChatRequest(model="m", messages=[Message(role=Role.USER, content="hi")])

    # Constructing the iterator against a deployment with a bogus URL must not
    # raise; only iterating it would try to connect.
    iterator = provider.stream(request, _deployment(cls.name, base_url="http://127.0.0.1:1"))
    assert iterator is not None


@pytest.mark.parametrize("cls", PROVIDER_CLASSES, ids=lambda c: c.name)
async def test_errors_are_mapped_to_gateway_errors(
    cls: type[Provider], client: httpx.AsyncClient
) -> None:
    """Adapters must never leak httpx or vendor exceptions upward."""
    from app.core.errors import ErrorCode, GatewayError

    provider = cls(client)
    deployment = _deployment(cls.name)

    mapped = provider.map_error(httpx.ConnectError("refused"), deployment)
    assert isinstance(mapped, GatewayError)
    assert mapped.code in {
        ErrorCode.PROVIDER_UNAVAILABLE,
        ErrorCode.PROVIDER_TIMEOUT,
        ErrorCode.PROVIDER_ERROR,
    }
    assert mapped.provider == cls.name

    timeout = provider.map_error(httpx.ReadTimeout("slow"), deployment)
    assert timeout.code is ErrorCode.PROVIDER_TIMEOUT
    assert timeout.retryable


@pytest.mark.parametrize("cls", PROVIDER_CLASSES, ids=lambda c: c.name)
async def test_rate_limit_is_retryable_and_auth_is_not(
    cls: type[Provider], client: httpx.AsyncClient
) -> None:
    from app.core.errors import ErrorCode

    provider = cls(client)
    deployment = _deployment(cls.name)

    rate_limited = provider.map_error(
        _status_error(429, {"error": {"message": "slow down"}}), deployment
    )
    assert rate_limited.code is ErrorCode.PROVIDER_RATE_LIMIT
    assert rate_limited.retryable and rate_limited.fallbackable

    unauthorized = provider.map_error(
        _status_error(401, {"error": {"message": "bad key"}}), deployment
    )
    assert unauthorized.code is ErrorCode.AUTHENTICATION_ERROR
    # A bad credential will still be bad on the next attempt.
    assert not unauthorized.retryable


@pytest.mark.parametrize("cls", PROVIDER_CLASSES, ids=lambda c: c.name)
async def test_context_overflow_is_detected_and_is_fallbackable(
    cls: type[Provider], client: httpx.AsyncClient
) -> None:
    """Providers report overflow as a generic 400; routing needs the specific code."""
    from app.core.errors import ErrorCode

    provider = cls(client)
    mapped = provider.map_error(
        _status_error(
            400, {"error": {"message": "This model's maximum context length is 8192 tokens"}}
        ),
        _deployment(cls.name),
    )
    assert mapped.code is ErrorCode.CONTEXT_LENGTH_EXCEEDED
    assert mapped.fallbackable
    # Retrying the identical oversized prompt cannot succeed.
    assert not mapped.retryable


def test_chat_response_is_attributable() -> None:
    """Every response must identify where it came from, for cost and logs."""
    response = ChatResponse(
        model="gpt-4o",
        provider="openai",
        choices=[Choice(index=0, message=Message(role=Role.ASSISTANT, content="hi"))],
    )
    assert response.provider == "openai"
    assert response.deployment_id is None


def _deployment(provider: str, base_url: str | None = None) -> Deployment:
    return Deployment(
        id=f"{provider}/test",
        model_name="test-model",
        provider=provider,
        provider_model="test-model",
        api_key="test-key",
        base_url=base_url or "http://localhost:9",
        capabilities=Capabilities(tools=True, vision=True, embeddings=True),
        pricing=Pricing(input_per_mtok=1.0, output_per_mtok=2.0),
    )


def _status_error(status: int, payload: dict[str, Any]) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:9/v1/chat")
    response = httpx.Response(status, json=payload, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)
