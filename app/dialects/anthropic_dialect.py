"""Anthropic Messages API wire-format translation."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.core.errors import GatewayError, InvalidRequestError
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    FinishReason,
    FunctionDef,
    ImagePart,
    Message,
    Role,
    StreamChunk,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolDef,
)
from app.dialects.base import Dialect, extract_gateway_fields


class AnthropicDialect(Dialect):
    name = "anthropic"

    """Translate between the Anthropic Messages API and the gateway schema."""

    def decode_chat(self, payload: dict[str, Any]) -> ChatRequest:
        max_tokens = payload.get("max_tokens")
        if max_tokens is None:
            raise InvalidRequestError("Field 'max_tokens' is required")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise InvalidRequestError("Field 'max_tokens' must be a positive integer")

        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise InvalidRequestError("Field 'model' must be a non-empty string")

        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list):
            raise InvalidRequestError("Field 'messages' must be an array")

        messages: list[Message] = []
        system = self._decode_system(payload.get("system"))
        if system:
            messages.append(Message(role=Role.SYSTEM, content=system))

        for message in raw_messages:
            if not isinstance(message, dict):
                raise InvalidRequestError("Each message must be an object")
            role = message.get("role")
            if role not in (Role.USER.value, Role.ASSISTANT.value):
                raise InvalidRequestError("Message role must be 'user' or 'assistant'")
            content, tool_calls, tool_results = self._decode_message_content(message.get("content"))
            if content is not None or tool_calls or not tool_results:
                messages.append(Message(role=Role(role), content=content, tool_calls=tool_calls))
            messages.extend(tool_results)

        tools = self._decode_tools(payload.get("tools", []))
        tool_choice = self._decode_tool_choice(payload.get("tool_choice"))
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise InvalidRequestError("Field 'metadata' must be an object")

        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": payload.get("stream", False),
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "top_k": payload.get("top_k"),
            "stop": payload.get("stop_sequences"),
            "tools": tools,
            "tool_choice": tool_choice,
            "metadata": metadata,
            "user": metadata.get("user_id"),
            **extract_gateway_fields(payload),
        }
        try:
            return ChatRequest.model_validate(request)
        except ValidationError as exc:
            raise InvalidRequestError(
                "Invalid Anthropic chat request", details={"errors": exc.errors()}
            ) from exc

    def encode_chat(self, response: ChatResponse) -> dict[str, Any]:
        message = response.choices[0].message if response.choices else Message(role=Role.ASSISTANT)
        content: list[dict[str, Any]] = []
        if message.content:
            if isinstance(message.content, str):
                content.append({"type": "text", "text": message.content})
            else:
                for part in message.content:
                    if isinstance(part, TextPart):
                        content.append({"type": "text", "text": part.text})
                    else:
                        content.append(self._encode_image(part))

        for tool_call in message.tool_calls:
            try:
                arguments = json.loads(tool_call.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                raise InvalidRequestError("Tool arguments must be a JSON object") from exc
            if not isinstance(arguments, dict):
                raise InvalidRequestError("Tool arguments must be a JSON object")
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": arguments,
                }
            )

        finish_reason = response.choices[0].finish_reason if response.choices else FinishReason.STOP
        return {
            "id": response.id,
            "type": "message",
            "role": "assistant",
            "model": response.model,
            "content": content,
            "stop_reason": self._encode_stop_reason(finish_reason),
            "stop_sequence": None,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            # Gateway extras; Anthropic SDKs ignore unknown top-level fields.
            **self._gateway_extras(response),
        }

    @staticmethod
    def _gateway_extras(response: ChatResponse) -> dict[str, Any]:
        extras: dict[str, Any] = {}
        if response.sources:
            extras["sources"] = response.sources
        if response.stop_reason:
            extras["stop_reason"] = response.stop_reason
            extras["tool_calls_executed"] = response.tool_calls_executed
        return {"aigw": extras} if extras else {}

    def encode_chunk(self, chunk: StreamChunk, state: dict[str, Any]) -> list[str]:
        events: list[str] = []
        if chunk.content:
            active_block = state.get("active_block")
            if active_block is None or active_block[0] != "text":
                self._close_active_block(state, events)
            if state.get("text_block_index") is None:
                block_index = self._next_block_index(state)
                state["text_block_index"] = block_index
                events.append(
                    self._event(
                        "content_block_start",
                        {"index": block_index, "content_block": {"type": "text", "text": ""}},
                    )
                )
            else:
                block_index = state["text_block_index"]
            state["active_block"] = ("text", block_index)
            events.append(
                self._event(
                    "content_block_delta",
                    {
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": chunk.content},
                    },
                )
            )

        for tool_delta in chunk.tool_calls:
            tool_indices: dict[int, int] = state.setdefault("tool_block_indices", {})
            tool_block_index = tool_indices.get(tool_delta.index)
            if tool_block_index is None:
                self._close_active_block(state, events)
                tool_block_index = self._next_block_index(state)
                tool_indices[tool_delta.index] = tool_block_index
                state.setdefault("tool_ids", {})[tool_delta.index] = (
                    tool_delta.id or f"toolu_{tool_delta.index}"
                )
                state.setdefault("tool_names", {})[tool_delta.index] = tool_delta.name or ""
                state["active_block"] = ("tool", tool_delta.index)
                events.append(
                    self._event(
                        "content_block_start",
                        {
                            "index": tool_block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": state["tool_ids"][tool_delta.index],
                                "name": state["tool_names"][tool_delta.index],
                                "input": {},
                            },
                        },
                    )
                )
            elif state.get("active_block") != ("tool", tool_delta.index):
                self._close_active_block(state, events)
                state["active_block"] = ("tool", tool_delta.index)

            if tool_delta.id is not None:
                state.setdefault("tool_ids", {})[tool_delta.index] = tool_delta.id
            if tool_delta.name is not None:
                state.setdefault("tool_names", {})[tool_delta.index] = tool_delta.name
            if tool_delta.arguments is not None:
                events.append(
                    self._event(
                        "content_block_delta",
                        {
                            "index": tool_block_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": tool_delta.arguments,
                            },
                        },
                    )
                )

        if chunk.usage is not None:
            state["usage"] = {
                "input_tokens": chunk.usage.prompt_tokens,
                "output_tokens": chunk.usage.completion_tokens,
                "cache_read_input_tokens": chunk.usage.cached_tokens,
            }
        if chunk.finish_reason is not None:
            state["finish_reason"] = chunk.finish_reason
            self._close_active_block(state, events)
            events.append(self._message_delta(state))
            state["message_delta_emitted"] = True
        return events

    def encode_stream_start(self, response_id: str, model: str, state: dict[str, Any]) -> list[str]:
        state.clear()
        state.update(
            {
                "id": response_id,
                "model": model,
                "next_block_index": 0,
                "text_block_index": None,
                "active_block": None,
                "tool_block_indices": {},
                "tool_ids": {},
                "tool_names": {},
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                "finish_reason": FinishReason.STOP,
                "message_delta_emitted": False,
            }
        )
        return [
            self._event(
                "message_start",
                {
                    "message": {
                        "id": response_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    }
                },
            )
        ]

    def encode_stream_end(self, state: dict[str, Any]) -> list[str]:
        events: list[str] = []
        self._close_active_block(state, events)
        if not state.get("message_delta_emitted", False):
            events.append(self._message_delta(state))
            state["message_delta_emitted"] = True
        events.append(self._event("message_stop", {}))
        return events

    def encode_error(self, error: GatewayError) -> dict[str, Any]:
        error_types = {
            "invalid_request": "invalid_request_error",
            "authentication_error": "authentication_error",
            "permission_denied": "permission_error",
            "not_found": "not_found_error",
            "rate_limit_exceeded": "rate_limit_error",
            "provider_rate_limit": "rate_limit_error",
            "provider_overloaded": "overloaded_error",
            "provider_unavailable": "api_error",
            "provider_timeout": "api_error",
            "provider_error": "api_error",
            "internal_error": "api_error",
        }
        return {
            "type": "error",
            "error": {
                "type": error_types.get(error.code.value, "api_error"),
                "message": error.message,
            },
        }

    @staticmethod
    def _decode_system(system: Any) -> str:
        if system is None:
            return ""
        if isinstance(system, str):
            return system
        if isinstance(system, list):
            return "".join(
                block.get("text", "")
                for block in system
                if isinstance(block, dict) and block.get("type") == "text"
            )
        raise InvalidRequestError("Field 'system' must be a string or array of text blocks")

    def _decode_message_content(
        self, content: Any
    ) -> tuple[str | list[TextPart | ImagePart] | None, list[ToolCall], list[Message]]:
        if content is None:
            return None, [], []
        if isinstance(content, str):
            return content, [], []
        if not isinstance(content, list):
            raise InvalidRequestError("Message content must be a string or array")

        parts: list[TextPart | ImagePart] = []
        tool_calls: list[ToolCall] = []
        tool_results: list[Message] = []
        for block in content:
            if not isinstance(block, dict):
                raise InvalidRequestError("Content blocks must be objects")
            block_type = block.get("type")
            if block_type == "text":
                parts.append(TextPart(text=block.get("text", "")))
            elif block_type == "image":
                parts.append(self._decode_image(block))
            elif block_type == "tool_use":
                name = block.get("name")
                tool_id = block.get("id")
                if not isinstance(name, str) or not isinstance(tool_id, str):
                    raise InvalidRequestError("Tool-use blocks require string 'id' and 'name'")
                tool_input = block.get("input", {})
                if not isinstance(tool_input, dict):
                    raise InvalidRequestError("Tool-use input must be a JSON object")
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=name,
                        arguments=json.dumps(tool_input, separators=(",", ":")),
                    )
                )
            elif block_type == "tool_result":
                tool_use_id = block.get("tool_use_id")
                if not isinstance(tool_use_id, str):
                    raise InvalidRequestError("Tool-result blocks require string 'tool_use_id'")
                tool_results.append(
                    Message(
                        role=Role.TOOL,
                        content=self._decode_tool_result(block.get("content")),
                        tool_call_id=tool_use_id,
                    )
                )
            else:
                raise InvalidRequestError(f"Unsupported content block type: {block_type!r}")
        return parts or None, tool_calls, tool_results

    @staticmethod
    def _decode_image(block: dict[str, Any]) -> ImagePart:
        source = block.get("source")
        if not isinstance(source, dict):
            raise InvalidRequestError("Image blocks require a source object")
        if source.get("type") == "base64":
            media_type = source.get("media_type")
            data = source.get("data")
            if not isinstance(media_type, str) or not isinstance(data, str):
                raise InvalidRequestError("Base64 image sources require media_type and data")
            return ImagePart(url=f"data:{media_type};base64,{data}", media_type=media_type)
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return ImagePart(url=source["url"])
        raise InvalidRequestError("Image source must be base64 or url")

    @staticmethod
    def _decode_tool_result(content: Any) -> str | list[TextPart | ImagePart] | None:
        if content is None or isinstance(content, str):
            return content
        if not isinstance(content, list):
            raise InvalidRequestError("Tool-result content must be a string or array")
        parts: list[TextPart | ImagePart] = []
        for block in content:
            if not isinstance(block, dict):
                raise InvalidRequestError("Tool-result content blocks must be objects")
            if block.get("type") == "text":
                parts.append(TextPart(text=block.get("text", "")))
            elif block.get("type") == "image":
                parts.append(AnthropicDialect._decode_image(block))
            else:
                raise InvalidRequestError(
                    f"Unsupported tool-result content block type: {block.get('type')!r}"
                )
        return parts or None

    @staticmethod
    def _decode_tools(tools: Any) -> list[ToolDef]:
        if not isinstance(tools, list):
            raise InvalidRequestError("Field 'tools' must be an array")
        definitions: list[ToolDef] = []
        for tool in tools:
            if not isinstance(tool, dict):
                raise InvalidRequestError("Tool definitions must be objects")
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                raise InvalidRequestError("Tool definitions require a non-empty name")
            definitions.append(
                ToolDef(
                    function=FunctionDef(
                        name=name,
                        description=tool.get("description", ""),
                        parameters=tool.get("input_schema", {"type": "object", "properties": {}}),
                    )
                )
            )
        return definitions

    @staticmethod
    def _decode_tool_choice(choice: Any) -> ToolChoice | None:
        if choice is None:
            return None
        if not isinstance(choice, dict):
            raise InvalidRequestError("Field 'tool_choice' must be an object")
        choice_type = choice.get("type")
        if choice_type == "auto":
            return ToolChoice(mode="auto")
        if choice_type == "any":
            return ToolChoice(mode="required")
        if choice_type == "tool" and isinstance(choice.get("name"), str):
            return ToolChoice(mode="function", function_name=choice["name"])
        if choice_type == "none":
            return ToolChoice(mode="none")
        raise InvalidRequestError("Unsupported tool_choice type")

    @staticmethod
    def _encode_image(part: ImagePart) -> dict[str, Any]:
        if part.url.startswith("data:"):
            header, separator, data = part.url[5:].partition(",")
            if separator:
                media_type = (
                    part.media_type or header.partition(";")[0] or "application/octet-stream"
                )
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
        return {"type": "image", "source": {"type": "url", "url": part.url}}

    @staticmethod
    def _encode_stop_reason(reason: FinishReason) -> str:
        return {
            FinishReason.STOP: "end_turn",
            FinishReason.LENGTH: "max_tokens",
            FinishReason.TOOL_CALLS: "tool_use",
            FinishReason.CONTENT_FILTER: "refusal",
            FinishReason.ERROR: "end_turn",
        }[reason]

    @staticmethod
    def _event(name: str, data: dict[str, Any]) -> str:
        # Anthropic repeats the event name as "type" inside every payload, and
        # SDKs (notably the TypeScript one) dispatch on it, not the SSE line.
        encoded = json.dumps({"type": name, **data}, separators=(",", ":"), ensure_ascii=False)
        return f"event: {name}\ndata: {encoded}\n\n"

    @staticmethod
    def _next_block_index(state: dict[str, Any]) -> int:
        index: int = state.get("next_block_index", 0)
        state["next_block_index"] = index + 1
        return index

    def _close_active_block(self, state: dict[str, Any], events: list[str]) -> None:
        active = state.get("active_block")
        if active is None:
            return
        block_type, key = active
        block_index = key if block_type == "text" else state["tool_block_indices"][key]
        if block_type == "text":
            state["text_block_index"] = None
        events.append(self._event("content_block_stop", {"index": block_index}))
        state["active_block"] = None

    def _message_delta(self, state: dict[str, Any]) -> str:
        usage = state.get("usage", {})
        return self._event(
            "message_delta",
            {
                "delta": {
                    "stop_reason": self._encode_stop_reason(
                        state.get("finish_reason", FinishReason.STOP)
                    ),
                    "stop_sequence": None,
                },
                "usage": {
                    "output_tokens": usage.get("output_tokens", 0),
                    "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                },
            },
        )
