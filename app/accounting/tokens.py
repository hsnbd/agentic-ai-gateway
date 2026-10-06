from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import tiktoken

from app.core.schemas import ChatRequest, Message

logger = logging.getLogger(__name__)

#: Longest a request waits for a tokenizer to load before falling back to an estimate.
ENCODER_WAIT_SECONDS = 2.0
#: After a failed load (e.g. no network), how long before trying again.
ENCODER_RETRY_SECONDS = 300.0

# tiktoken downloads its BPE files on first use, with no timeout, unless they
# are cached (TIKTOKEN_CACHE_DIR; the Docker image bakes them in). Loading runs
# in a background thread so an offline or slow network bounds a request's
# wait instead of hanging it; until the encoding arrives, counts are estimated.
_encodings: dict[str, Any] = {}
_loaders: dict[str, threading.Thread] = {}
_failed_at: dict[str, float] = {}
_lock = threading.Lock()


def _load(name: str) -> None:
    try:
        _encodings[name] = tiktoken.get_encoding(name)
    except Exception as exc:
        _failed_at[name] = time.monotonic()
        logger.warning("Tokenizer %s unavailable (%s); estimating token counts", name, exc)


def _encoding(name: str) -> Any | None:
    loaded = _encodings.get(name)
    if loaded is not None:
        return loaded
    with _lock:
        loader = _loaders.get(name)
        retry_due = time.monotonic() - _failed_at.get(name, float("-inf")) >= ENCODER_RETRY_SECONDS
        start = loader is None or (not loader.is_alive() and retry_due)
        if start:
            loader = threading.Thread(
                target=_load, args=(name,), name=f"tiktoken-{name}", daemon=True
            )
            _loaders[name] = loader
            loader.start()
    if start and loader is not None:
        loader.join(ENCODER_WAIT_SECONDS)
    return _encodings.get(name)


def _encoder_for_model(model: str) -> Any | None:
    try:
        name = tiktoken.encoding_name_for_model(model)
    except KeyError:
        return None
    return _encoding(name)


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
