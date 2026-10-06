from __future__ import annotations

import asyncio
import base64
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import unquote_to_bytes

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


class GeminiProvider(Provider):
    name = "gemini"

    def _headers(self, deployment: Deployment) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            **deployment.extra_headers,
            **({"x-goog-api-key": deployment.api_key} if deployment.api_key else {}),
        }

    @staticmethod
    def _clean_schema(value: Any) -> Any:
        unsupported = {
            "additionalProperties",
            "$schema",
            "$ref",
            "definitions",
            "$defs",
            "exclusiveMinimum",
            "exclusiveMaximum",
        }
        if isinstance(value, dict):
            return {
                key: GeminiProvider._clean_schema(item)
                for key, item in value.items()
                if key not in unsupported
            }
        if isinstance(value, list):
            return [GeminiProvider._clean_schema(item) for item in value]
        return value

    @staticmethod
    def _image_part(image: ImagePart) -> dict[str, Any]:
        if image.url.startswith("data:"):
            try:
                header, payload = image.url[5:].split(",", 1)
            except ValueError as exc:
                raise ValueError("Malformed image data URI") from exc
            metadata = header.split(";")
            mime_type = image.media_type or metadata[0] or "application/octet-stream"
            if "base64" in metadata[1:]:
                data = payload
            else:
                data = base64.b64encode(unquote_to_bytes(payload)).decode("ascii")
            return {"inlineData": {"mimeType": mime_type, "data": data}}
        if image.url.startswith(("http://", "https://")):
            return {
                "fileData": {
                    "mimeType": image.media_type or "application/octet-stream",
                    "fileUri": image.url,
                }
            }
        raise ValueError("Image URL must be a data URI or an http(s) URL")

    @staticmethod
    def _message_parts(message: Message) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        if isinstance(message.content, str):
            if message.content:
                parts.append({"text": message.content})
        elif isinstance(message.content, list):
            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append({"text": part.text})
                else:
                    parts.append(GeminiProvider._image_part(part))
        return parts

    @classmethod
    def _contents(cls, request: ChatRequest) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        call_names: dict[str, str] = {}
        parts: list[dict[str, Any]]
        for message in request.non_system_messages():
            if message.role == Role.TOOL:
                call_id = message.tool_call_id or ""
                name = call_names.get(call_id, message.name or "unknown")
                parts = [
                    {
                        "functionResponse": {
                            "name": name,
                            "response": {"result": message.text()},
                        }
                    }
                ]
                role = "user"
            else:
                role = "model" if message.role == Role.ASSISTANT else "user"
                parts = cls._message_parts(message)
                for call in message.tool_calls:
                    call_names[call.id] = call.name
                    try:
                        arguments = json.loads(call.arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}
                    parts.append({"functionCall": {"name": call.name, "args": arguments}})
            if not parts:
                parts = [{"text": ""}]
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})
        return contents

    def _payload(self, request: ChatRequest, deployment: Deployment) -> dict[str, Any]:
        payload: dict[str, Any] = {"contents": self._contents(request)}
        system_prompt = request.system_prompt()
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if request.tools:
            declarations = [
                {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": self._clean_schema(tool.function.parameters),
                }
                for tool in request.tools
            ]
            payload["tools"] = [{"functionDeclarations": declarations}]
        if request.tool_choice is not None:
            choice = request.tool_choice
            mode = {"auto": "AUTO", "none": "NONE", "required": "ANY", "function": "ANY"}[
                choice.mode
            ]
            calling_config: dict[str, Any] = {"mode": mode}
            if choice.mode == "function" and choice.function_name:
                calling_config["allowedFunctionNames"] = [choice.function_name]
            payload["toolConfig"] = {"functionCallingConfig": calling_config}
        params = self._merge_params(request, deployment)
        generation: dict[str, Any] = {}
        for canonical, gemini in (
            ("temperature", "temperature"),
            ("top_p", "topP"),
            ("max_tokens", "maxOutputTokens"),
            ("seed", "seed"),
        ):
            if params.get(canonical) is not None:
                generation[gemini] = params[canonical]
        if request.stop:
            generation["stopSequences"] = request.stop
        if request.n:
            generation["candidateCount"] = request.n
        if request.response_format:
            generation["responseMimeType"] = "application/json"
        if generation:
            payload["generationConfig"] = generation
        return payload

    @staticmethod
    def _usage(metadata: Any) -> Usage:
        if not isinstance(metadata, dict):
            return Usage()
        return Usage(
            prompt_tokens=metadata.get("promptTokenCount", 0),
            completion_tokens=metadata.get("candidatesTokenCount", 0),
            total_tokens=metadata.get("totalTokenCount", 0),
            cached_tokens=metadata.get("cachedContentTokenCount", 0),
        )

    @staticmethod
    def _finish_reason(reason: Any) -> FinishReason:
        if reason == "MAX_TOKENS":
            return FinishReason.LENGTH
        if reason in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION"}:
            return FinishReason.CONTENT_FILTER
        return FinishReason.STOP

    @classmethod
    def _response(
        cls, data: dict[str, Any], request: ChatRequest, deployment: Deployment
    ) -> ChatResponse:
        candidates = data.get("candidates", [])
        candidate = candidates[0] if candidates else {}
        content = candidate.get("content", {})
        text: list[str] = []
        tool_calls: list[ToolCall] = []
        for part in content.get("parts", []):
            if isinstance(part.get("text"), str):
                text.append(part["text"])
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                tool_calls.append(
                    ToolCall(
                        name=function_call.get("name", ""),
                        arguments=json.dumps(function_call.get("args", {})),
                    )
                )
        finish_reason = (
            FinishReason.TOOL_CALLS
            if tool_calls
            else cls._finish_reason(candidate.get("finishReason"))
        )
        message = Message(role=Role.ASSISTANT, content="".join(text), tool_calls=tool_calls)
        return ChatResponse(
            model=request.model,
            choices=[Choice(message=message, finish_reason=finish_reason)],
            usage=cls._usage(data.get("usageMetadata")),
            provider=cls.name,
            deployment_id=deployment.id,
        )

    async def chat(self, request: ChatRequest, deployment: Deployment) -> ChatResponse:
        try:
            url = f"{self._base_url(deployment)}/models/{deployment.provider_model}:generateContent"
            response = await self._client.post(
                url, headers=self._headers(deployment), json=self._payload(request, deployment)
            )
            response.raise_for_status()
            return self._response(response.json(), request, deployment)
        except Exception as exc:
            raise self.map_error(exc, deployment) from exc

    async def stream(
        self, request: ChatRequest, deployment: Deployment
    ) -> AsyncIterator[StreamChunk]:
        async for chunk in self._stream(request, deployment):
            yield chunk

    async def _stream(
        self, request: ChatRequest, deployment: Deployment
    ) -> AsyncIterator[StreamChunk]:
        url = (
            f"{self._base_url(deployment)}/models/{deployment.provider_model}"
            ":streamGenerateContent?alt=sse"
        )
        tool_ids: dict[str, str] = {}
        tool_indices: dict[str, int] = {}
        has_tool_calls = False
        latest_usage: Usage | None = None
        latest_finish: FinishReason | None = None
        event_lines: list[str] = []

        async def emit_event(event_data: str) -> StreamChunk | None:
            nonlocal has_tool_calls, latest_usage, latest_finish
            if event_data == "[DONE]":
                return None
            fragment = json.loads(event_data)
            candidates = fragment.get("candidates", [])
            candidate = candidates[0] if candidates else {}
            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(part["text"] for part in parts if isinstance(part.get("text"), str))
            deltas: list[ToolCallDelta] = []
            for part in parts:
                function_call = part.get("functionCall")
                if not isinstance(function_call, dict):
                    continue
                has_tool_calls = True
                name = function_call.get("name", "")
                if name not in tool_indices:
                    tool_indices[name] = len(tool_indices)
                    tool_ids[name] = f"call_{uuid.uuid4().hex[:24]}"
                deltas.append(
                    ToolCallDelta(
                        index=tool_indices[name],
                        id=tool_ids[name],
                        name=name,
                        arguments=json.dumps(function_call.get("args", {})),
                    )
                )
            reason = candidate.get("finishReason")
            if reason is not None:
                latest_finish = (
                    FinishReason.TOOL_CALLS if has_tool_calls else self._finish_reason(reason)
                )
            if "usageMetadata" in fragment:
                latest_usage = self._usage(fragment["usageMetadata"])
            if not text and not deltas and reason is None and "usageMetadata" not in fragment:
                return None
            return StreamChunk(
                model=request.model,
                role=Role.ASSISTANT if text or deltas else None,
                content=text or None,
                tool_calls=deltas,
                finish_reason=latest_finish if reason is not None else None,
                usage=latest_usage if "usageMetadata" in fragment else None,
                provider=self.name,
            )

        try:
            async with self._client.stream(
                "POST",
                url,
                headers=self._headers(deployment),
                json=self._payload(request, deployment),
            ) as response:
                await self._raise_for_stream_status(response)
                async for line in response.aiter_lines():
                    if not line:
                        if event_lines:
                            event_data = "\n".join(event_lines)
                            event_lines.clear()
                            chunk = await emit_event(event_data)
                            if chunk is not None:
                                yield chunk
                    elif line.startswith("data:"):
                        event_lines.append(line[5:].lstrip())
                if event_lines:
                    chunk = await emit_event("\n".join(event_lines))
                    if chunk is not None:
                        yield chunk
            if latest_usage is not None or latest_finish is not None:
                yield StreamChunk(
                    model=request.model,
                    usage=latest_usage,
                    finish_reason=latest_finish,
                    provider=self.name,
                )
        except Exception as exc:
            raise self.map_error(exc, deployment) from exc

    async def embed(self, request: EmbeddingRequest, deployment: Deployment) -> EmbeddingResponse:
        async def embed_one(text: str) -> dict[str, Any]:
            body: dict[str, Any] = {
                "model": f"models/{deployment.provider_model}",
                "content": {"parts": [{"text": text}]},
            }
            if request.dimensions is not None:
                body["outputDimensionality"] = request.dimensions
            response = await self._client.post(
                f"{self._base_url(deployment)}/models/{deployment.provider_model}:embedContent",
                headers=self._headers(deployment),
                json=body,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            return payload

        try:
            responses = await asyncio.gather(*(embed_one(text) for text in request.input))
            data = [
                EmbeddingVector(index=index, embedding=entry["embedding"]["values"])
                for index, entry in enumerate(responses)
            ]
            usage = Usage()
            for entry in responses:
                metadata = entry.get("usageMetadata", {})
                if isinstance(metadata, dict):
                    prompt_tokens = metadata.get("promptTokenCount", 0)
                    usage += Usage(
                        prompt_tokens=prompt_tokens,
                        total_tokens=metadata.get("totalTokenCount", prompt_tokens),
                        cached_tokens=metadata.get("cachedContentTokenCount", 0),
                    )
            return EmbeddingResponse(
                model=request.model, data=data, usage=usage, provider=self.name
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
