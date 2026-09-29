"""OpenAI-compatible provider adapter."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    Choice,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
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


class OpenAIProvider(Provider):
    name = "openai"

    def _headers(self, deployment: Deployment) -> dict[str, str]:
        return {
            **super()._headers(deployment),
            "Authorization": f"Bearer {deployment.api_key or ''}",
        }

    @staticmethod
    def _message_payload(message: Message) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role.value}
        if isinstance(message.content, list):
            content: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextPart):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": part.url, "detail": part.detail},
                    })
            payload["content"] = content
        else:
            payload["content"] = message.content
        if message.name is not None:
            payload["name"] = message.name
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
        return payload

    def _chat_payload(
        self, request: ChatRequest, deployment: Deployment, *, stream: bool = False
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": deployment.provider_model,
            "messages": [self._message_payload(message) for message in request.messages],
        }
        merged = self._merge_params(request, deployment)
        for key in (
            "temperature", "top_p", "max_tokens", "seed",
            "presence_penalty", "frequency_penalty",
        ):
            if key in merged:
                payload[key] = merged[key]
        for key in ("stop", "n", "response_format", "parallel_tool_calls", "user"):
            value = getattr(request, key)
            if value is not None:
                payload[key] = value
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters,
                    },
                }
                for tool in request.tools
            ]
        if request.tool_choice is not None:
            choice = request.tool_choice
            if choice.mode == "function":
                function: dict[str, str] = {}
                if choice.function_name is not None:
                    function["name"] = choice.function_name
                payload["tool_choice"] = {"type": "function", "function": function}
            else:
                payload["tool_choice"] = choice.mode
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _finish_reason(value: str | None) -> FinishReason:
        return {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALLS,
            "function_call": FinishReason.TOOL_CALLS,
            "content_filter": FinishReason.CONTENT_FILTER,
        }.get(value or "", FinishReason.STOP)

    @staticmethod
    def _usage(payload: dict[str, Any]) -> Usage:
        prompt_details = payload.get("prompt_tokens_details") or {}
        completion_details = payload.get("completion_tokens_details") or {}
        return Usage(
            prompt_tokens=payload.get("prompt_tokens", 0),
            completion_tokens=payload.get("completion_tokens", 0),
            total_tokens=payload.get("total_tokens", 0),
            cached_tokens=prompt_details.get("cached_tokens", 0),
            reasoning_tokens=completion_details.get("reasoning_tokens", 0),
        )

    @staticmethod
    def _response_message(payload: dict[str, Any]) -> Message:
        tool_calls = [
            ToolCall(
                id=call.get("id", ""),
                name=call.get("function", {}).get("name", ""),
                arguments=call.get("function", {}).get("arguments", "{}"),
            )
            for call in payload.get("tool_calls") or []
        ]
        return Message(
            role=Role(payload.get("role", "assistant")),
            content=payload.get("content"),
            tool_calls=tool_calls,
        )

    async def chat(self, request: ChatRequest, deployment: Deployment) -> ChatResponse:
        started = time.perf_counter()
        try:
            response = await self._client.post(
                f"{self._base_url(deployment)}/chat/completions",
                headers=self._headers(deployment),
                json=self._chat_payload(request, deployment),
            )
            response.raise_for_status()
            body = response.json()
            choices = [
                Choice(
                    index=item.get("index", 0),
                    message=self._response_message(item.get("message", {})),
                    finish_reason=self._finish_reason(item.get("finish_reason")),
                )
                for item in body.get("choices", [])
            ]
            return ChatResponse(
                id=body.get("id", ""),
                model=body.get("model", deployment.provider_model),
                created=body.get("created", int(time.time())),
                choices=choices,
                usage=self._usage(body.get("usage") or {}),
                provider=self.name,
                deployment_id=deployment.id,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            raise self.map_error(exc, deployment) from exc

    def stream(
        self, request: ChatRequest, deployment: Deployment
    ) -> AsyncIterator[StreamChunk]:
        return self._stream(request, deployment)

    async def _stream(
        self, request: ChatRequest, deployment: Deployment
    ) -> AsyncIterator[StreamChunk]:
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url(deployment)}/chat/completions",
                headers=self._headers(deployment),
                json=self._chat_payload(request, deployment, stream=True),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or line.startswith(":") or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        body = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if body.get("usage") is not None:
                        yield StreamChunk(
                            id=body.get("id", ""),
                            model=body.get("model", deployment.provider_model),
                            created=body.get("created", int(time.time())),
                            usage=self._usage(body["usage"]),
                            provider=self.name,
                        )
                    for choice in body.get("choices", []):
                        delta = choice.get("delta") or {}
                        tool_calls = [
                            ToolCallDelta(
                                index=call.get("index", 0),
                                id=call.get("id"),
                                name=(call.get("function") or {}).get("name"),
                                arguments=(call.get("function") or {}).get("arguments"),
                            )
                            for call in delta.get("tool_calls") or []
                        ]
                        finish_reason = choice.get("finish_reason")
                        yield StreamChunk(
                            id=body.get("id", ""),
                            model=body.get("model", deployment.provider_model),
                            created=body.get("created", int(time.time())),
                            index=choice.get("index", 0),
                            role=Role(delta["role"]) if delta.get("role") else None,
                            content=delta.get("content"),
                            tool_calls=tool_calls,
                            finish_reason=self._finish_reason(finish_reason)
                            if finish_reason is not None
                            else None,
                            provider=self.name,
                        )
        except Exception as exc:
            raise self.map_error(exc, deployment) from exc

    async def embed(
        self, request: EmbeddingRequest, deployment: Deployment
    ) -> EmbeddingResponse:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": deployment.provider_model,
            "input": request.input,
        }
        if request.dimensions is not None:
            payload["dimensions"] = request.dimensions
        if request.user is not None:
            payload["user"] = request.user
        try:
            response = await self._client.post(
                f"{self._base_url(deployment)}/embeddings",
                headers=self._headers(deployment),
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            data = sorted(body.get("data", []), key=lambda item: item.get("index", 0))
            usage_payload = body.get("usage") or {}
            prompt_tokens = usage_payload.get("prompt_tokens", 0)
            return EmbeddingResponse(
                model=body.get("model", deployment.provider_model),
                data=[
                    EmbeddingVector(index=item.get("index", 0), embedding=item["embedding"])
                    for item in data
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    total_tokens=usage_payload.get("total_tokens", prompt_tokens),
                ),
                provider=self.name,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            raise self.map_error(exc, deployment) from exc

    async def health_check(self, deployment: Deployment) -> bool:
        try:
            response = await self._client.get(
                f"{self._base_url(deployment)}/models", headers=self._headers(deployment)
            )
            return 200 <= response.status_code < 300
        except Exception:
            return False
