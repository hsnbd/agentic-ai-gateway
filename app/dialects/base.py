"""Client-facing wire format contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.errors import GatewayError
from app.core.schemas import ChatRequest, ChatResponse, StreamChunk


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
