"""Anthropic-compatible Messages API."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.chat import _error, _json_body, serve_chat
from app.core.errors import GatewayError
from app.dialects.anthropic_dialect import AnthropicDialect

router = APIRouter()


@router.post("/v1/messages", response_model=None)
async def messages(request: Request) -> JSONResponse | StreamingResponse:
    dialect = AnthropicDialect()
    try:
        payload = await _json_body(request)
        return await serve_chat(request, dialect.decode_chat(payload), dialect)
    except GatewayError as exc:
        return _error(exc, dialect)


@router.post("/v1/messages/count_tokens", response_model=None)
async def count_tokens(request: Request) -> JSONResponse:
    dialect = AnthropicDialect()
    try:
        payload = await _json_body(request)
        # Counting does not require max_tokens; decoding still validates the messages.
        chat = dialect.decode_chat({**payload, "max_tokens": payload.get("max_tokens", 1)})
        try:
            from app.accounting.tokens import count_message_tokens
        except ImportError:
            count = sum(len(message.text()) // 4 for message in chat.messages)
        else:
            count = count_message_tokens(chat.messages, chat.model)
        return JSONResponse({"input_tokens": count})
    except GatewayError as exc:
        return _error(exc, dialect)
