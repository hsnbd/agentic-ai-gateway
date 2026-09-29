"""OpenAI Chat Completions client wire format."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.core.errors import GatewayError, InvalidRequestError
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
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


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


class OpenAIDialect(Dialect):
    name = "openai"

    def decode_chat(self, payload: dict[str, Any]) -> ChatRequest:
        if not isinstance(payload.get("messages"), list):
            raise InvalidRequestError("'messages' must be an array")
        if not isinstance(payload.get("model"), str) or not payload["model"]:
            raise InvalidRequestError("'model' must be a non-empty string")
        messages = [self._decode_message(message) for message in payload["messages"]]
        raw_choice = payload.get("tool_choice")
        try:
            if isinstance(raw_choice, str):
                choice = ToolChoice(mode=raw_choice)
            elif isinstance(raw_choice, dict):
                if raw_choice.get("type") != "function":
                    raise InvalidRequestError("Unsupported tool_choice type")
                choice = ToolChoice(mode="function", function_name=raw_choice["function"]["name"])
            elif raw_choice is None:
                choice = None
            else:
                raise InvalidRequestError("Unsupported tool_choice value")
        except (ValidationError, KeyError, TypeError) as exc:
            raise InvalidRequestError(f"Invalid tool_choice: {exc}") from exc
        extensions = extract_gateway_fields(payload)
        stop = payload.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        raw_tools = payload.get("tools") or []
        try:
            return ChatRequest.model_validate(
                {
                    "model": payload["model"],
                    "messages": messages,
                    "tools": [ToolDef.model_validate(tool) for tool in raw_tools],
                    "tool_choice": choice,
                    "max_tokens": payload.get("max_completion_tokens", payload.get("max_tokens")),
                    "temperature": payload.get("temperature"),
                    "top_p": payload.get("top_p"),
                    "stop": stop,
                    "stream": payload.get("stream", False),
                    "seed": payload.get("seed"),
                    "n": payload.get("n", 1),
                    "presence_penalty": payload.get("presence_penalty"),
                    "frequency_penalty": payload.get("frequency_penalty"),
                    "user": payload.get("user"),
                    "response_format": payload.get("response_format"),
                    "parallel_tool_calls": payload.get("parallel_tool_calls"),
                    "metadata": payload.get("metadata") or {},
                    **extensions,
                }
            )
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            raise InvalidRequestError(f"Invalid OpenAI chat request: {exc}") from exc

    @staticmethod
    def _decode_message(raw: Any) -> Message:
        if not isinstance(raw, dict):
            raise InvalidRequestError("Each message must be an object")
        content = raw.get("content")
        if isinstance(content, list):
            try:
                return OpenAIDialect._decode_parts(raw, content)
            except (ValidationError, KeyError, TypeError, ValueError) as exc:
                raise InvalidRequestError(f"Invalid content parts: {exc}") from exc
        if content is not None and not isinstance(content, str):
            raise InvalidRequestError("Message content must be a string, array, or null")
        return OpenAIDialect._build_message(raw, content)

    @staticmethod
    def _decode_parts(raw: dict[str, Any], content: list[Any]) -> Message:
        parts: list[TextPart | ImagePart] = []
        for part in content:
            if not isinstance(part, dict):
                raise InvalidRequestError("Content parts must be objects")
            if part.get("type") == "text":
                parts.append(TextPart(text=part["text"]))
            elif part.get("type") == "image_url":
                image = part["image_url"]
                if isinstance(image, str):
                    image = {"url": image}
                parts.append(
                    ImagePart(
                        url=image["url"],
                        detail=image.get("detail", "auto"),
                        media_type=image["url"].split(";", 1)[0][5:]
                        if image["url"].startswith("data:")
                        else None,
                    )
                )
            else:
                raise InvalidRequestError(f"Unsupported content part: {part.get('type')}")
        content = parts
        return OpenAIDialect._build_message(raw, content)

    @staticmethod
    def _build_message(raw: dict[str, Any], content: Any) -> Message:
        try:
            return Message(
                role=Role(raw["role"]),
                content=content,
                name=raw.get("name"),
                tool_call_id=raw.get("tool_call_id"),
                tool_calls=[
                    ToolCall(
                        id=call["id"],
                        name=call["function"]["name"],
                        arguments=call["function"]["arguments"],
                    )
                    for call in raw.get("tool_calls") or []
                ],
            )
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            raise InvalidRequestError(f"Invalid message: {exc}") from exc

    @staticmethod
    def _message(message: Message) -> dict[str, Any]:
        if isinstance(message.content, list):
            content: str | list[dict[str, Any]] | None = [
                {"type": "text", "text": part.text}
                if isinstance(part, TextPart)
                else {"type": "image_url", "image_url": {"url": part.url, "detail": part.detail}}
                for part in message.content
            ]
        else:
            content = message.content
        result: dict[str, Any] = {"role": message.role.value, "content": content}
        if message.name is not None:
            result["name"] = message.name
        if message.tool_call_id is not None:
            result["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
        return result

    def encode_chat(self, response: ChatResponse) -> dict[str, Any]:
        return {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": self._message(choice.message),
                    "finish_reason": choice.finish_reason.value,
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "system_fingerprint": None,
            "aigw": {
                "provider": response.provider,
                "cache_hit": response.cache_hit,
                "latency_ms": response.latency_ms,
                "cost_usd": response.cost_usd,
                **({"sources": response.sources} if response.sources else {}),
                **(
                    {
                        "stop_reason": response.stop_reason,
                        "tool_calls_executed": response.tool_calls_executed,
                    }
                    if response.stop_reason
                    else {}
                ),
            },
        }

    def encode_stream_start(self, response_id: str, model: str, state: dict[str, Any]) -> list[str]:
        state["response_id"] = response_id
        state["model"] = model
        return []

    def encode_chunk(self, chunk: StreamChunk, state: dict[str, Any]) -> list[str]:
        delta: dict[str, Any] = {}
        if chunk.role is not None:
            delta["role"] = chunk.role.value
        if chunk.content is not None:
            delta["content"] = chunk.content
        if chunk.tool_calls:
            delta["tool_calls"] = [
                {
                    "index": call.index,
                    **({"id": call.id, "type": "function"} if call.id is not None else {}),
                    **(
                        {
                            "function": {
                                **({"name": call.name} if call.name is not None else {}),
                                **(
                                    {"arguments": call.arguments}
                                    if call.arguments is not None
                                    else {}
                                ),
                            }
                        }
                        if call.name is not None or call.arguments is not None
                        else {}
                    ),
                }
                for call in chunk.tool_calls
            ]
        common = {
            "id": chunk.id,
            "object": "chat.completion.chunk",
            "created": chunk.created,
            "model": chunk.model,
            "system_fingerprint": None,
        }
        result: list[str] = []
        if delta or chunk.finish_reason is not None or chunk.usage is None:
            result.append(
                _sse(
                    {
                        **common,
                        "choices": [
                            {
                                "index": chunk.index,
                                "delta": delta,
                                "finish_reason": chunk.finish_reason.value
                                if chunk.finish_reason
                                else None,
                            }
                        ],
                    }
                )
            )
        if chunk.usage is not None and state.get("include_usage"):
            result.append(
                _sse(
                    {
                        **common,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        },
                    }
                )
            )
        return result

    def encode_stream_end(self, state: dict[str, Any]) -> list[str]:
        return ["data: [DONE]\n\n"]

    def encode_error(self, error: GatewayError) -> dict[str, Any]:
        return error.to_dict()
