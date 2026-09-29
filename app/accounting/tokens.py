from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import tiktoken

from app.core.schemas import ChatRequest, Message


@lru_cache(maxsize=128)
def _encoder_for_model(model: str) -> Any | None:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return None


def count_tokens(text: str, model: str) -> int:
    encoder = _encoder_for_model(model)
    if encoder is None:
        return len(text) // 4
    try:
        return len(encoder.encode(text, disallowed_special=()))
    except (KeyError, ValueError):
        return len(text) // 4


def count_message_tokens(messages: list[Message], model: str) -> int:
    """Estimate chat tokens using message/name overhead and a priming allowance."""
    total = 3
    for message in messages:
        total += 3
        total += count_tokens(message.text(), model)
        if message.name is not None:
            total += 1 + count_tokens(message.name, model)
        if message.tool_call_id:
            total += count_tokens(message.tool_call_id, model)
        for tool_call in message.tool_calls:
            total += 3 + count_tokens(tool_call.name, model)
            total += count_tokens(tool_call.arguments, model)
    return total


def estimate_request_tokens(request: ChatRequest) -> int:
    total = count_message_tokens(request.messages, request.model)
    if request.tools:
        serialized_tools = json.dumps(
            [tool.model_dump(mode="json") for tool in request.tools],
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
        total += count_tokens(serialized_tools, request.model)
    return total


def _json_default(value: Any) -> str:
    return str(value)
