"""Provider abstraction.

A `Provider` is a stateless adapter that translates the canonical IR into one
vendor's HTTP API and back. A `Deployment` binds a provider to a concrete
model name plus credentials, pricing, and limits.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.core.errors import ErrorCode, ProviderError
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
)


class Capabilities(BaseModel):
    """What a deployment can actually do, used to filter routing candidates."""

    chat: bool = True
    streaming: bool = True
    tools: bool = False
    parallel_tool_calls: bool = False
    vision: bool = False
    json_mode: bool = False
    embeddings: bool = False
    max_context_tokens: int = 8192
    max_output_tokens: int | None = None

    def supports(self, request: ChatRequest) -> bool:
        if request.stream and not self.streaming:
            return False
        if request.requires_tools() and not self.tools:
            return False
        if request.requires_vision() and not self.vision:
            return False
        if request.response_format and not self.json_mode:
            return False
        return not (
            request.max_tokens
            and self.max_output_tokens
            and request.max_tokens > self.max_output_tokens
        )


class Pricing(BaseModel):
    """USD per one million tokens."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cached_input_per_mtok: float | None = None

    def estimate(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
        billable_prompt = max(prompt_tokens - cached_tokens, 0)
        cost = billable_prompt / 1_000_000 * self.input_per_mtok
        cost += completion_tokens / 1_000_000 * self.output_per_mtok
        if cached_tokens:
            rate = self.cached_input_per_mtok
            if rate is None:
                rate = self.input_per_mtok
            cost += cached_tokens / 1_000_000 * rate
        return cost


class Deployment(BaseModel):
    """A routable (provider, model) pair with its own credentials and limits."""

    id: str
    #: Public name clients request, e.g. "gpt-4o". Many deployments may share it.
    model_name: str
    provider: str
    #: Provider-side model identifier, which may differ from `model_name`.
    provider_model: str

    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    default_params: dict[str, Any] = Field(default_factory=dict)

    capabilities: Capabilities = Field(default_factory=Capabilities)
    pricing: Pricing = Field(default_factory=Pricing)

    weight: int = 1
    priority: int = 0
    rpm_limit: int | None = None
    tpm_limit: int | None = None
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.id)


class Provider(abc.ABC):
    """Base class for vendor adapters.

    Subclasses must be safe to share across concurrent requests; per-request
    state belongs in local variables, not on the instance.
    """

    #: Short stable identifier, e.g. "openai".
    name: str = "base"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    # -- Required surface -------------------------------------------------

    @abc.abstractmethod
    async def chat(self, request: ChatRequest, deployment: Deployment) -> ChatResponse:
        """Perform a unary chat completion."""

    @abc.abstractmethod
    def stream(self, request: ChatRequest, deployment: Deployment) -> AsyncIterator[StreamChunk]:
        """Perform a streaming chat completion."""

    # -- Optional surface -------------------------------------------------

    async def embed(self, request: EmbeddingRequest, deployment: Deployment) -> EmbeddingResponse:
        raise ProviderError(
            ErrorCode.INVALID_REQUEST,
            f"Provider {self.name!r} does not support embeddings",
            provider=self.name,
        )

    async def health_check(self, deployment: Deployment) -> bool:
        """Cheap liveness probe. Default: assume healthy."""
        return True

    # -- Shared helpers ---------------------------------------------------

    def _headers(self, deployment: Deployment) -> dict[str, str]:
        """Auth and content headers. Subclasses override and extend."""
        return {"Content-Type": "application/json", **deployment.extra_headers}

    def _base_url(self, deployment: Deployment) -> str:
        if not deployment.base_url:
            raise ProviderError(
                ErrorCode.CONFIGURATION_ERROR,
                f"Deployment {deployment.id!r} has no base_url",
                provider=self.name,
            )
        return deployment.base_url.rstrip("/")

    def _merge_params(self, request: ChatRequest, deployment: Deployment) -> dict[str, Any]:
        """Apply deployment defaults beneath explicit request values."""
        params = dict(deployment.default_params)
        for key in (
            "temperature",
            "top_p",
            "max_tokens",
            "seed",
            "presence_penalty",
            "frequency_penalty",
        ):
            value = getattr(request, key, None)
            if value is not None:
                params[key] = value
        return params

    @staticmethod
    async def _raise_for_stream_status(response: httpx.Response) -> None:
        """`raise_for_status` for a streamed response.

        A streamed body is not read up front, so the error detail (and the
        mapping that depends on it, e.g. context-length vs. generic 400) would
        be unreadable in `map_error`. Read it first on failure.
        """
        if response.is_error:
            await response.aread()
        response.raise_for_status()

    def map_error(self, exc: Exception, deployment: Deployment) -> ProviderError:
        """Translate a transport or HTTP error into the unified taxonomy."""
        model = deployment.provider_model

        if isinstance(exc, ProviderError):
            return exc

        if isinstance(exc, httpx.TimeoutException):
            return ProviderError(
                ErrorCode.PROVIDER_TIMEOUT,
                f"{self.name} request timed out",
                provider=self.name,
                model=model,
                cause=exc,
            )

        if isinstance(exc, httpx.ConnectError | httpx.NetworkError):
            return ProviderError(
                ErrorCode.PROVIDER_UNAVAILABLE,
                f"Cannot reach {self.name}: {exc}",
                provider=self.name,
                model=model,
                cause=exc,
            )

        if isinstance(exc, httpx.HTTPStatusError):
            return self._map_status_error(exc, deployment)

        return ProviderError(
            ErrorCode.PROVIDER_ERROR,
            f"{self.name} call failed: {exc}",
            provider=self.name,
            model=model,
            cause=exc,
        )

    def _map_status_error(
        self, exc: httpx.HTTPStatusError, deployment: Deployment
    ) -> ProviderError:
        status = exc.response.status_code
        detail = self._extract_error_message(exc.response)
        model = deployment.provider_model

        code = {
            400: ErrorCode.INVALID_REQUEST,
            401: ErrorCode.AUTHENTICATION_ERROR,
            403: ErrorCode.PERMISSION_DENIED,
            404: ErrorCode.NOT_FOUND,
            408: ErrorCode.PROVIDER_TIMEOUT,
            413: ErrorCode.CONTEXT_LENGTH_EXCEEDED,
            422: ErrorCode.INVALID_REQUEST,
            429: ErrorCode.PROVIDER_RATE_LIMIT,
            529: ErrorCode.PROVIDER_OVERLOADED,
        }.get(status)

        if code is None:
            code = ErrorCode.PROVIDER_UNAVAILABLE if status >= 500 else ErrorCode.PROVIDER_ERROR

        # Context-length failures arrive as generic 400s; detect them by text
        # so that routing can fail over to a larger-context model.
        lowered = detail.lower()
        if code is ErrorCode.INVALID_REQUEST and (
            "context length" in lowered
            or "context_length" in lowered
            or "too many tokens" in lowered
            or "maximum context" in lowered
        ):
            code = ErrorCode.CONTEXT_LENGTH_EXCEEDED

        retry_after: float | None = None
        raw_retry = exc.response.headers.get("retry-after")
        if raw_retry:
            try:
                retry_after = float(raw_retry)
            except ValueError:
                retry_after = None

        return ProviderError(
            code,
            f"{self.name} returned {status}: {detail}",
            provider=self.name,
            model=model,
            status_code=status,
            retry_after=retry_after,
            cause=exc,
        )

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except Exception:
            return (response.text or "")[:500]

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str):
                    return message
            if isinstance(error, str):
                return error
            for key in ("message", "detail"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value
        return str(payload)[:500]
