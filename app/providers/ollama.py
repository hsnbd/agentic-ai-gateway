"""Ollama provider adapter for local chat and embedding models."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, NoReturn

import httpx

from app.core.errors import ProviderError
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    Choice,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    FinishReason,
    Message,
    Role,
    StreamChunk,
    TextPart,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from app.providers.base import Deployment, Provider


class OllamaProvider(Provider):
    """Translate canonical gateway requests to Ollama's native API."""

    name = "ollama"

    def _headers(self, deployment: Deployment) -> dict[str, str]:
        headers = super()._headers(deployment)
        if deployment.api_key:
            headers["Authorization"] = f"Bearer {deployment.api_key}"
        return headers

    @staticmethod
    def _message_payload(message: Message) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role.value, "content": ""}
        images: list[str] = []

        if isinstance(message.content, str):
            payload["content"] = message.content
        elif message.content is not None:
            text: list[str] = []
            for part in message.content:
                if isinstance(part, TextPart):
                    text.append(part.text)
                elif part.url.startswith("data:"):
                    _, separator, data = part.url.partition(",")
                    if separator:
                        images.append(data)
                # Ollama takes raw image bytes, so remote image URLs are dropped, not fetched.
            payload["content"] = "".join(text)

        if images:
            payload["images"] = images
        if message.tool_calls:
            tool_calls: list[dict[str, Any]] = []
            for call in message.tool_calls:
                arguments = json.loads(call.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("Ollama tool call arguments must be a JSON object")
                tool_calls.append({"function": {"name": call.name, "arguments": arguments}})
            payload["tool_calls"] = tool_calls
        return payload

    def _request_payload(self, request: ChatRequest, deployment: Deployment) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": deployment.provider_model,
            "messages": [self._message_payload(message) for message in request.messages],
            "stream": request.stream,
        }

        merged = self._merge_params(request, deployment)
        if request.stop is not None:
            merged["stop"] = request.stop
        option_names = {
            "temperature": "temperature",
            "top_p": "top_p",
            "max_tokens": "num_predict",
            "stop": "stop",
            "seed": "seed",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
        }
        options = {
            ollama_name: merged[name]
            for name, ollama_name in option_names.items()
            if merged.get(name) is not None
        }
        if options:
            payload["options"] = options

        if request.tools:
            payload["tools"] = [tool.model_dump(exclude_none=True) for tool in request.tools]
        if request.response_format:
            payload["format"] = "json"
        return payload

    @staticmethod
    def _usage(payload: dict[str, Any]) -> Usage:
        return Usage.of(
            int(payload.get("prompt_eval_count", 0) or 0),
            int(payload.get("eval_count", 0) or 0),
        )

    @staticmethod
    def _finish_reason(payload: dict[str, Any], has_tool_calls: bool = False) -> FinishReason:
        if has_tool_calls:
            return FinishReason.TOOL_CALLS
        if payload.get("done_reason") == "length":
            return FinishReason.LENGTH
        return FinishReason.STOP

    @staticmethod
    def _tool_calls(payload: dict[str, Any]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for item in payload.get("tool_calls", []):
            function = item.get("function", {})
            calls.append(
                ToolCall(
                    name=function["name"],
                    arguments=json.dumps(function.get("arguments", {})),
                )
            )
        return calls

    @staticmethod
    def _raise_mapped(exc: Exception, deployment: Deployment, provider: OllamaProvider) -> NoReturn:
        mapped = provider.map_error(exc, deployment)
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
            detail = provider._extract_error_message(exc.response)
            if "model" in detail.lower() and "not found" in detail.lower():
                mapped = ProviderError(
                    mapped.code,
                    f"{mapped.message}. Pull the model with `ollama pull "
                    f"{deployment.provider_model}` and retry.",
                    provider=provider.name,
                    model=deployment.provider_model,
                    status_code=mapped.status_code,
                    retry_after=mapped.retry_after,
                    details=mapped.details,
                    cause=exc,
                )
        raise mapped from exc

    async def chat(self, request: ChatRequest, deployment: Deployment) -> ChatResponse:
        url = f"{self._base_url(deployment)}/api/chat"
        try:
            response = await self._client.post(
                url,
                json=self._request_payload(request, deployment),
                headers=self._headers(deployment),
            )
            response.raise_for_status()
            data = response.json()
            raw_message = data.get("message", {})
            tool_calls = self._tool_calls(raw_message)
            message = Message(
                role=Role(raw_message.get("role", "assistant")),
                content=raw_message.get("content", ""),
                tool_calls=tool_calls,
            )
            return ChatResponse(
                model=request.model,
                choices=[
                    Choice(
                        message=message,
                        finish_reason=self._finish_reason(data, bool(tool_calls)),
                    )
                ],
                usage=self._usage(data),
                provider=self.name,
                deployment_id=deployment.id,
            )
        except Exception as exc:
            self._raise_mapped(exc, deployment, self)

    async def stream(
        self, request: ChatRequest, deployment: Deployment
    ) -> AsyncIterator[StreamChunk]:
        url = f"{self._base_url(deployment)}/api/chat"
        saw_tool_calls = False
        try:
            payload = self._request_payload(request, deployment)
            payload["stream"] = True
            async with self._client.stream(
                "POST", url, json=payload, headers=self._headers(deployment)
            ) as response:
                await self._raise_for_stream_status(response)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    message = data.get("message", {})
                    calls = self._tool_calls(message)
                    saw_tool_calls = saw_tool_calls or bool(calls)
                    deltas = [
                        ToolCallDelta(index=index, name=call.name, arguments=call.arguments)
                        for index, call in enumerate(calls)
                    ]
                    content = message.get("content")
                    if content or deltas:
                        yield StreamChunk(
                            model=request.model,
                            role=Role.ASSISTANT,
                            content=content or None,
                            tool_calls=deltas,
                            provider=self.name,
                        )
                    if data.get("done") is True:
                        yield StreamChunk(
                            model=request.model,
                            finish_reason=self._finish_reason(data, saw_tool_calls),
                            usage=self._usage(data),
                            provider=self.name,
                        )
        except Exception as exc:
            self._raise_mapped(exc, deployment, self)

    async def embed(self, request: EmbeddingRequest, deployment: Deployment) -> EmbeddingResponse:
        url = f"{self._base_url(deployment)}/api/embed"
        try:
            response = await self._client.post(
                url,
                json={"model": deployment.provider_model, "input": request.input},
                headers=self._headers(deployment),
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings")
            if embeddings is None and "embedding" in data:
                embeddings = [data["embedding"]]
            if embeddings is None:
                raise ValueError("Ollama embedding response has no embeddings")
            vectors = [
                EmbeddingVector(index=index, embedding=vector)
                for index, vector in enumerate(embeddings)
            ]
            return EmbeddingResponse(
                model=request.model,
                data=vectors,
                usage=Usage.of(int(data.get("prompt_eval_count", 0) or 0), 0),
                provider=self.name,
            )
        except Exception as exc:
            self._raise_mapped(exc, deployment, self)

    async def health_check(self, deployment: Deployment) -> bool:
        try:
            response = await self._client.get(
                f"{self._base_url(deployment)}/api/tags",
                headers=self._headers(deployment),
            )
            return 200 <= response.status_code < 300
        except Exception:
            return False
