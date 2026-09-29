"""Client-facing wire format contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.errors import GatewayError
from app.core.schemas import ChatRequest, ChatResponse, StreamChunk

#: Request fields the gateway consumes itself and never forwards upstream.
GATEWAY_FIELDS: tuple[str, ...] = (
    "no_cache",
    "cache_ttl",
    "fallbacks",
    "routing_strategy",
    "guardrail_policy",
    "tags",
    "rag",
    "mcp",
)


def extract_gateway_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Collect gateway controls from `metadata`, then `aigw`, then the top level.

    Later sources win, so an explicit top-level field overrides the same key
    nested in `aigw`, which overrides one in `metadata`. SDKs that reject
    unknown top-level fields can always use `aigw` (e.g. OpenAI's
    `extra_body={"aigw": {...}}`).
    """
    extensions: dict[str, Any] = {}
    for key in ("metadata", "aigw"):
        value = payload.get(key)
        if isinstance(value, dict):
            extensions.update(value)
    extensions.update({key: payload[key] for key in GATEWAY_FIELDS if key in payload})
    return {key: extensions[key] for key in GATEWAY_FIELDS if key in extensions}


class Dialect(ABC):
    name: str

    @abstractmethod
    def decode_chat(self, payload: dict[str, Any]) -> ChatRequest: ...

    @abstractmethod
    def encode_chat(self, response: ChatResponse) -> dict[str, Any]: ...

    @abstractmethod
    def encode_chunk(self, chunk: StreamChunk, state: dict[str, Any]) -> list[str]:
        """Return already-formatted SSE frames, each ending with a blank line.

        Anthropic requires named ``event:`` frames; OpenAI uses bare ``data:`` frames.
        """

    @abstractmethod
    def encode_stream_start(
        self, response_id: str, model: str, state: dict[str, Any]
    ) -> list[str]: ...

    @abstractmethod
    def encode_stream_end(self, state: dict[str, Any]) -> list[str]: ...

    @abstractmethod
    def encode_error(self, error: GatewayError) -> dict[str, Any]: ...
