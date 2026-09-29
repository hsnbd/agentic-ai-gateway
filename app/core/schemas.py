"""Provider-neutral request/response schemas.

Every inbound dialect (OpenAI, Anthropic, native) is translated into these
types, and every provider adapter translates them onward. Nothing in the
pipeline should reference a provider-specific shape.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


# --------------------------------------------------------------------------
# Content parts
# --------------------------------------------------------------------------


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    type: Literal["image"] = "image"
    #: Either a URL or a `data:` URI. Adapters convert to provider form.
    url: str
    media_type: str | None = None
    detail: Literal["auto", "low", "high"] = "auto"


ContentPart = TextPart | ImagePart


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


class FunctionDef(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolDef(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    type: Literal["function"] = "function"
    name: str
    #: Raw JSON string, kept verbatim so streaming deltas can append to it.
    arguments: str = "{}"


class ToolChoice(BaseModel):
    """Normalized tool choice: auto | none | required | a specific function."""

    mode: Literal["auto", "none", "required", "function"] = "auto"
    function_name: str | None = None


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------


class Message(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    role: Role
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    #: Set when role == TOOL, linking the result back to its call.
    tool_call_id: str | None = None

    def text(self) -> str:
        """Flatten content to plain text, ignoring non-text parts."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(p.text for p in self.content if isinstance(p, TextPart))


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message]
    stream: bool = False

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    n: int = 1

    tools: list[ToolDef] = Field(default_factory=list)
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None

    user: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # --- Gateway-specific controls (never forwarded to providers) ---
    no_cache: bool = False
    cache_ttl: int | None = None
    fallbacks: list[str] = Field(default_factory=list)
    routing_strategy: str | None = None
    guardrail_policy: str | None = None
    tags: list[str] = Field(default_factory=list)

    def system_prompt(self) -> str:
        """Concatenate all system messages, which some providers need hoisted out."""
        return "\n\n".join(m.text() for m in self.messages if m.role == Role.SYSTEM)

    def non_system_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role != Role.SYSTEM]

    def requires_tools(self) -> bool:
        return bool(self.tools)

    def requires_vision(self) -> bool:
        return any(
            isinstance(m.content, list) and any(isinstance(p, ImagePart) for p in m.content)
            for m in self.messages
        )


class EmbeddingRequest(BaseModel):
    model: str
    input: list[str]
    dimensions: int | None = None
    user: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    #: Provider-reported cache reads, when available.
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    @classmethod
    def of(cls, prompt: int, completion: int, **kw: int) -> Usage:
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            **kw,
        )


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: FinishReason = FinishReason.STOP


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    model: str
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)

    # --- Gateway annotations, surfaced to clients as headers/extras ---
    provider: str | None = None
    deployment_id: str | None = None
    cache_hit: bool = False
    cache_similarity: float | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    attempt_count: int = 1
    fallback_used: bool = False

    @property
    def text(self) -> str:
        return self.choices[0].message.text() if self.choices else ""

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.choices[0].message.tool_calls if self.choices else []


class EmbeddingVector(BaseModel):
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    model: str
    data: list[EmbeddingVector]
    usage: Usage = Field(default_factory=Usage)
    provider: str | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


class ToolCallDelta(BaseModel):
    index: int = 0
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


class StreamChunk(BaseModel):
    """One incremental update. Adapters emit these; dialects re-encode them."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    model: str = ""
    created: int = Field(default_factory=lambda: int(time.time()))
    index: int = 0

    role: Role | None = None
    content: str | None = None
    tool_calls: list[ToolCallDelta] = Field(default_factory=list)
    finish_reason: FinishReason | None = None
    usage: Usage | None = None

    provider: str | None = None
    deployment_id: str | None = None
    cache_hit: bool = False
