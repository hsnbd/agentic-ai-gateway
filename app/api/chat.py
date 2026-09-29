"""OpenAI-compatible HTTP API and shared dialect execution helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.core.errors import (
    ConfigurationError,
    ErrorCode,
    GatewayError,
    InvalidRequestError,
    NotFoundError,
)
from app.core.pipeline import Pipeline, RequestContext
from app.core.schemas import ChatRequest, EmbeddingRequest, Message, Role
from app.core.state import GatewayState, get_state
from app.dialects.base import Dialect
from app.dialects.openai_dialect import OpenAIDialect

router = APIRouter()
logger = logging.getLogger(__name__)
_STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidRequestError(f"Invalid JSON request body: {exc}") from exc
    if not isinstance(payload, dict):
        raise InvalidRequestError("Request body must be an object")
    return payload


def _pipeline(state: GatewayState) -> Pipeline:
    """Startup owns pipeline construction; routes must never build one lazily."""
    return state.require_pipeline()


def _context(request: Request, chat: ChatRequest, dialect: Dialect) -> RequestContext:
    # Credentials from headers override client-controlled metadata.
    key = request.headers.get("x-api-key")
    authorization = request.headers.get("authorization", "")
    if not key and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
    chat.metadata["api_key"] = key
    return RequestContext(
        request=chat, state=get_state(request.app), dialect=dialect.name, route=request.url.path
    )


def _headers(ctx: RequestContext) -> dict[str, str]:
    result = {
        "X-Gateway-Request-Id": ctx.request_id,
        "X-Gateway-Cache": "hit" if ctx.cache_hit else "miss",
        "X-Gateway-Cost-USD": str(ctx.cost_usd),
        "X-Gateway-Latency-Ms": str(round(ctx.elapsed_ms(), 3)),
    }
    provider = ctx.response.provider if ctx.response else None
    deployment_id = ctx.response.deployment_id if ctx.response else None
    if ctx.routing:
        provider = provider or ctx.routing.deployment.provider
        deployment_id = deployment_id or ctx.routing.deployment.id
    if provider:
        result["X-Gateway-Provider"] = provider
    if deployment_id:
        result["X-Gateway-Deployment"] = deployment_id
    return result


def _error(
    error: GatewayError, dialect: Dialect, ctx: RequestContext | None = None
) -> JSONResponse:
    headers = _headers(ctx) if ctx else {}
    if error.retry_after is not None:
        headers["Retry-After"] = str(int(error.retry_after))
    return JSONResponse(dialect.encode_error(error), status_code=error.status_code, headers=headers)


def _stream_error(error: GatewayError, dialect: Dialect) -> str:
    body = json.dumps(dialect.encode_error(error), separators=(",", ":"))
    if dialect.name == "anthropic":
        return f"event: error\ndata: {body}\n\n"
    return f"data: {body}\n\n"


async def serve_chat(
    request: Request, chat: ChatRequest, dialect: Dialect, *, include_usage: bool = False
) -> JSONResponse | StreamingResponse:
    ctx = _context(request, chat, dialect)
    try:
        pipeline = _pipeline(ctx.state)
        if not chat.stream:
            response = await pipeline.run(ctx)
            ctx.response = response
            return JSONResponse(dialect.encode_chat(response), headers=_headers(ctx))

        iterator = pipeline.run_stream(ctx).__aiter__()
        # Pre-stages and the provider are lazy: advance before sending HTTP 200.
        try:
            first = await anext(iterator)
        except StopAsyncIteration:
            raise GatewayError(
                code=ErrorCode.INTERNAL_ERROR,
                message="Provider returned an empty stream",
            ) from None
        state: dict[str, Any] = {"include_usage": include_usage}
        start = dialect.encode_stream_start(first.id, first.model, state)
        state["include_usage"] = include_usage
        headers = _headers(ctx)
        if first.provider:
            headers["X-Gateway-Provider"] = first.provider
        if first.deployment_id:
            headers["X-Gateway-Deployment"] = first.deployment_id
        if first.cache_hit:
            headers["X-Gateway-Cache"] = "hit"

        async def frames() -> AsyncIterator[str]:
            try:
                for frame in start:
                    yield frame
                for frame in dialect.encode_chunk(first, state):
                    yield frame
                async for chunk in iterator:
                    for frame in dialect.encode_chunk(chunk, state):
                        yield frame
                for frame in dialect.encode_stream_end(state):
                    yield frame
            except GatewayError as exc:
                logger.warning("Stream failed after response start: %s", exc)
                yield _stream_error(exc, dialect)
                if dialect.name == "openai":
                    yield "data: [DONE]\n\n"
            finally:
                if isinstance(iterator, AsyncGenerator):
                    await iterator.aclose()

        return StreamingResponse(
            frames(), media_type="text/event-stream", headers={**_STREAM_HEADERS, **headers}
        )
    except GatewayError as exc:
        return _error(exc, dialect, ctx)


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    dialect = OpenAIDialect()
    try:
        payload = await _json_body(request)
        chat = dialect.decode_chat(payload)
        return await serve_chat(
            request,
            chat,
            dialect,
            include_usage=bool((payload.get("stream_options") or {}).get("include_usage")),
        )
    except GatewayError as exc:
        return _error(exc, dialect)


@router.post("/v1/completions", response_model=None)
async def completions(request: Request) -> JSONResponse | StreamingResponse:
    dialect = OpenAIDialect()
    try:
        payload = await _json_body(request)
        prompt = payload.get("prompt")
        if isinstance(prompt, list) and all(isinstance(p, str) for p in prompt):
            prompt = "\n".join(prompt)
        if not isinstance(prompt, str):
            raise InvalidRequestError("'prompt' must be a string or array of strings")
        chat = dialect.decode_chat({**payload, "messages": [{"role": "user", "content": prompt}]})
        if chat.stream:
            streamed = await serve_chat(request, chat, dialect)
            if not isinstance(streamed, StreamingResponse):
                return streamed

            async def legacy_frames() -> AsyncIterator[str]:
                async for frame in streamed.body_iterator:
                    if not isinstance(frame, str):
                        frame = bytes(frame).decode("utf-8")
                    if frame == "data: [DONE]\n\n":
                        yield frame
                        continue
                    body = json.loads(frame.removeprefix("data: "))
                    if "error" in body:
                        yield frame
                        continue
                    for choice in body.get("choices", []):
                        delta = choice["delta"]
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "id": body["id"],
                                    "object": "text_completion",
                                    "created": body["created"],
                                    "model": body["model"],
                                    "choices": [
                                        {
                                            "text": delta.get("content", ""),
                                            "index": choice["index"],
                                            "logprobs": None,
                                            "finish_reason": choice["finish_reason"],
                                        }
                                    ],
                                },
                                separators=(",", ":"),
                            )
                            + "\n\n"
                        )

            return StreamingResponse(
                legacy_frames(), media_type="text/event-stream", headers=dict(streamed.headers)
            )
        ctx = _context(request, chat, dialect)
        response = await _pipeline(ctx.state).run(ctx)
        ctx.response = response
        return JSONResponse(
            {
                "id": response.id,
                "object": "text_completion",
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "text": choice.message.text(),
                        "index": choice.index,
                        "logprobs": None,
                        "finish_reason": choice.finish_reason.value,
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            },
            headers=_headers(ctx),
        )
    except GatewayError as exc:
        return _error(exc, dialect)


@router.post("/v1/embeddings", response_model=None)
async def embeddings(request: Request) -> JSONResponse:
    dialect = OpenAIDialect()
    try:
        payload = await _json_body(request)
        raw = payload.get("input")
        inputs = [raw] if isinstance(raw, str) else raw
        if (
            not isinstance(inputs, list)
            or not inputs
            or not all(isinstance(value, str) for value in inputs)
        ):
            raise InvalidRequestError("'input' must be a string or array of strings")
        try:
            embedding = EmbeddingRequest(
                model=payload["model"],
                input=inputs,
                dimensions=payload.get("dimensions"),
                user=payload.get("user"),
            )
        except (ValidationError, KeyError) as exc:
            raise InvalidRequestError(f"Invalid embedding request: {exc}") from exc
        state = get_state(request.app)
        ctx = _context(
            request,
            ChatRequest(model=embedding.model, messages=[Message(role=Role.USER, content="")]),
            dialect,
        )
        # Embeddings bypass chat-specific cache and guardrails, but not key authentication.
        try:
            from app.auth.stage import AuthStage
        except ImportError as exc:
            raise ConfigurationError("Authentication stage is unavailable") from exc

        await AuthStage().process(ctx)
        deployments = [
            d for d in state.registry.deployments_for(embedding.model) if d.capabilities.embeddings
        ]
        if not deployments:
            raise NotFoundError(f"No embedding deployment for {embedding.model!r}")
        deployment = deployments[0]
        result = await state.registry.provider_for(deployment).embed(embedding, deployment)
        return JSONResponse(
            {
                "object": "list",
                "model": result.model,
                "data": [
                    {"object": "embedding", "index": vector.index, "embedding": vector.embedding}
                    for vector in result.data
                ],
                "usage": {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "total_tokens": result.usage.total_tokens,
                },
            },
            headers={**_headers(ctx), "X-Gateway-Provider": deployment.provider},
        )
    except GatewayError as exc:
        return _error(exc, dialect)


def _model(model_id: str, state: GatewayState) -> dict[str, Any]:
    deployments = state.registry.deployments_for(model_id)
    if not deployments:
        raise NotFoundError(f"Model {model_id!r} is not available")
    return {"id": model_id, "object": "model", "created": 0, "owned_by": deployments[0].provider}


@router.get("/v1/models", response_model=None)
async def models(request: Request) -> JSONResponse:
    state = get_state(request.app)
    return JSONResponse(
        {
            "object": "list",
            "data": [
                _model(model_id, state)
                for model_id in state.registry.list_models()
                if state.registry.deployments_for(model_id)
            ],
        }
    )


@router.get("/v1/models/{model:path}", response_model=None)
async def model(request: Request, model: str) -> JSONResponse:
    try:
        return JSONResponse(_model(model, get_state(request.app)))
    except GatewayError as exc:
        return _error(exc, OpenAIDialect())
