"""Anthropic Messages API provider adapter."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import unquote_to_bytes

from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    Choice,
    FinishReason,
    ImagePart,
    Message,
    Role,
    StreamChunk,
    TextPart,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from app.providers.base import Deployment, Provider


class AnthropicProvider(Provider):
    name = "anthropic"

    def _headers(self, deployment: Deployment) -> dict[str, str]:
        return {
            **super()._headers(deployment),
            "x-api-key": deployment.api_key or "",
            "anthropic-version": deployment.api_version or "2023-06-01",
        }

    def _content_blocks(
        self, content: str | list[TextPart | ImagePart] | None
    ) -> list[dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []

        blocks: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, TextPart):
                blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                if part.url.startswith("data:"):
                    header, separator, data = part.url[5:].partition(",")
                    if not separator:
                        raise ValueError("Invalid data URI for Anthropic image")
                    metadata = header.split(";")
                    media_type = part.media_type or metadata[0] or "application/octet-stream"
                    if "base64" in metadata[1:]:
                        encoded = unquote_to_bytes(data).decode("ascii")
                    else:
                        encoded = base64.b64encode(unquote_to_bytes(data)).decode("ascii")
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": encoded,
                            },
                        }
                    )
                elif part.url.startswith(("http://", "https://")):
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": part.url},
                        }
                    )
                else:
                    raise ValueError(f"Unsupported image URI: {part.url!r}")
        return blocks

    def _request_body(self, request: ChatRequest, deployment: Deployment) -> dict[str, Any]:
        params = self._merge_params(request, deployment)
        body: dict[str, Any] = {
            "model": deployment.provider_model,
            "max_tokens": (
                request.max_tokens
                if request.max_tokens is not None
                else deployment.capabilities.max_output_tokens or 4096
            ),
            "messages": [],
        }

        system = request.system_prompt()
        if system:
            body["system"] = system

        messages: list[dict[str, Any]] = body["messages"]
        for message in request.non_system_messages():
            if message.role == Role.TOOL:
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": self._content_blocks(message.content),
                    }
                ]
                role = "user"
            else:
                role = message.role.value
                blocks = self._content_blocks(message.content)
                if message.role == Role.ASSISTANT:
                    for tool_call in message.tool_calls:
                        try:
                            arguments = json.loads(tool_call.arguments)
                        except (json.JSONDecodeError, TypeError):
                            arguments = {}
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tool_call.id,
                                "name": tool_call.name,
                                "input": arguments,
                            }
                        )

            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": role, "content": blocks})

        if request.tools and not (request.tool_choice and request.tool_choice.mode == "none"):
            body["tools"] = [
                {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "input_schema": tool.function.parameters,
                }
                for tool in request.tools
            ]
            if request.tool_choice is not None:
                choice = request.tool_choice
                if choice.mode == "auto":
                    body["tool_choice"] = {"type": "auto"}
                elif choice.mode == "required":
                    body["tool_choice"] = {"type": "any"}
                elif choice.mode == "function" and choice.function_name:
                    body["tool_choice"] = {"type": "tool", "name": choice.function_name}

        if params.get("temperature") is not None:
            body["temperature"] = params["temperature"]
        if params.get("top_p") is not None:
            body["top_p"] = params["top_p"]
        stop = request.stop if request.stop is not None else deployment.default_params.get("stop")
        if stop is not None:
            body["stop_sequences"] = stop
        return body

    @staticmethod
    def _finish_reason(stop_reason: str | None) -> FinishReason:
        return {
            "end_turn": FinishReason.STOP,
            "stop_sequence": FinishReason.STOP,
            "max_tokens": FinishReason.LENGTH,
            "tool_use": FinishReason.TOOL_CALLS,
            "refusal": FinishReason.CONTENT_FILTER,
        }.get(stop_reason or "", FinishReason.STOP)

    @staticmethod
    def _usage(usage: dict[str, Any]) -> Usage:
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        return Usage.of(
            prompt_tokens,
            completion_tokens,
            cached_tokens=usage.get("cache_read_input_tokens", 0),
        )

    async def chat(self, request: ChatRequest, deployment: Deployment) -> ChatResponse:
        try:
            response = await self._client.post(
                f"{self._base_url(deployment)}/messages",
                headers=self._headers(deployment),
                json=self._request_body(request, deployment),
            )
            response.raise_for_status()
            payload = response.json()
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            for block in payload.get("content", []):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block["id"],
                            name=block["name"],
                            arguments=json.dumps(block.get("input", {})),
                        )
                    )

            message = Message(
                role=Role.ASSISTANT,
                content="".join(text_parts),
                tool_calls=tool_calls,
            )
            return ChatResponse(
                id=payload.get("id", ""),
                model=deployment.provider_model,
                choices=[
                    Choice(
                        message=message,
                        finish_reason=self._finish_reason(payload.get("stop_reason")),
                    )
                ],
                usage=self._usage(payload.get("usage", {})),
                provider=self.name,
                deployment_id=deployment.id,
            )
        except Exception as exc:
            mapped = self.map_error(exc, deployment)
            if mapped is exc:
                raise mapped from None
            raise mapped from exc

    async def stream(
        self, request: ChatRequest, deployment: Deployment
    ) -> AsyncIterator[StreamChunk]:
        stream_id = ""
        finish_reason = FinishReason.STOP
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        final_emitted = False

        def chunk(**kwargs: Any) -> StreamChunk:
            return StreamChunk(
                id=stream_id or "",
                model=deployment.provider_model,
                provider=self.name,
                **kwargs,
            )

        async def process_event(event: str, data: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            nonlocal stream_id, finish_reason, input_tokens, output_tokens, cached_tokens
            nonlocal final_emitted

            if event == "message_start":
                message = data.get("message", {})
                stream_id = message.get("id", "")
                usage = message.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                cached_tokens = usage.get("cache_read_input_tokens", 0)
                yield chunk(role=Role.ASSISTANT)
            elif event == "content_block_start":
                index = data.get("index", 0)
                block = data.get("content_block", {})
                if block.get("type") == "tool_use":
                    yield chunk(
                        tool_calls=[
                            ToolCallDelta(index=index, id=block.get("id"), name=block.get("name"))
                        ]
                    )
            elif event == "content_block_delta":
                index = data.get("index", 0)
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield chunk(content=delta.get("text", ""))
                elif delta.get("type") == "input_json_delta":
                    yield chunk(
                        tool_calls=[
                            ToolCallDelta(index=index, arguments=delta.get("partial_json", ""))
                        ]
                    )
            elif event == "message_delta":
                delta = data.get("delta", {})
                if "stop_reason" in delta:
                    finish_reason = self._finish_reason(delta.get("stop_reason"))
                usage = data.get("usage", {})
                output_tokens = usage.get("output_tokens", output_tokens)
                cached_tokens = usage.get("cache_read_input_tokens", cached_tokens)
            elif event == "message_stop":
                usage = Usage.of(
                    input_tokens,
                    output_tokens,
                    cached_tokens=cached_tokens,
                )
                yield chunk(finish_reason=finish_reason, usage=usage)
                final_emitted = True

        async def parse_event(event: str, lines: list[str]) -> AsyncIterator[StreamChunk]:
            if not lines:
                return
            data = json.loads("\n".join(lines))
            if not isinstance(data, dict):
                raise ValueError("Anthropic SSE event data must be a JSON object")
            async for item in process_event(event, data):
                yield item

        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url(deployment)}/messages",
                headers={**self._headers(deployment), "Accept": "text/event-stream"},
                json={**self._request_body(request, deployment), "stream": True},
            ) as response:
                response.raise_for_status()
                event_name = "message"
                data_lines: list[str] = []
                async for line in response.aiter_lines():
                    if line == "":
                        async for item in parse_event(event_name, data_lines):
                            yield item
                        event_name = "message"
                        data_lines = []
                    elif line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                async for item in parse_event(event_name, data_lines):
                    yield item

            if not final_emitted:
                yield chunk(
                    finish_reason=finish_reason,
                    usage=Usage.of(input_tokens, output_tokens, cached_tokens=cached_tokens),
                )
        except Exception as exc:
            mapped = self.map_error(exc, deployment)
            if mapped is exc:
                raise mapped from None
            raise mapped from exc
