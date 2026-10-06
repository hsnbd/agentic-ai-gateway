"""A tiny multi-provider upstream for end-to-end evaluation.

The evaluation harness needs a provider that answers deterministically, fails
on demand, and costs nothing. Pointing the gateway's OpenAI adapter at this
server exercises the *entire* real path — dialect translation, auth, routing,
retry, fallback, caching, guardrails, accounting — without a network call or an
API key.

It speaks each provider's native API under its own prefix, so every adapter
runs its real translation code against it:

  ``/``             OpenAI (chat completions, embeddings, models)
  ``/anthropic``    Anthropic Messages (``/messages``, unary and SSE)
  ``/gemini``       Gemini (``generateContent``, ``streamGenerateContent``,
                    ``embedContent``, ``models``)
  ``/ollama``       Ollama (``/api/chat`` unary and NDJSON, ``/api/embed``,
                    ``/api/tags``)

Run it, point a deployment's ``base_url`` at it, and drive the gateway normally:

    uv run python scripts/fake_upstream.py --port 4100

Behaviour is controlled by the prompt text, so a test can request a failure
without any out-of-band configuration:

  ``__fail__``      always returns 503, to exercise retry and fallback
  ``__flaky__``     fails twice per prompt, then succeeds
  ``__slow__``      sleeps 1.5s before responding
  ``__tool__``      returns a tool call for the first declared tool
                    (``__tool__:<name>`` picks one; required args are filled in)
  ``__unsafe__``    scored 1.0 by LLM-judge guardrail requests (else 0.0)

A trailing tool message is answered with ``Tool result: <content>``. Every
dialect shares these behaviours; only the wire format differs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
import uuid
from collections import defaultdict
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Fake OpenAI upstream")

#: prompt text -> remaining failures, for the ``__flaky__`` behaviour.
_flaky_budget: dict[str, int] = defaultdict(lambda: 2)


def _prompt_of(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _answer(prompt: str) -> str:
    # Deterministic, so a semantic-cache hit is detectable by comparing bodies.
    return f"Reply to: {prompt.strip()[:180]}"


def _pick_tool(prompt: str, functions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the tool a ``__tool__`` prompt asks for, as ``{name, arguments}``.

    Triggered by ``__tool__`` (the first declared tool) or ``__tool__:<name>``
    (the first tool whose name ends with ``<name>``), so tool-calling and the
    gateway's agent loop can be exercised without a real provider. Required
    arguments are filled from the tool's schema (``parameters``): numbers count
    up from 2, strings are "hi".
    """
    if "__tool__" not in prompt:
        return None
    match = re.search(r"__tool__:([A-Za-z0-9_-]+)", prompt)
    if match:
        functions = [f for f in functions if str(f.get("name", "")).endswith(match.group(1))]
    if not functions or not functions[0].get("name"):
        return None
    function = functions[0]
    return {
        "name": function["name"],
        "arguments": _arguments_for(function.get("parameters") or {}),
    }


def _tool_calls(prompt: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """OpenAI-shaped tool calls for a ``__tool__`` prompt (see ``_pick_tool``)."""
    if _last_role(payload) == "tool":
        return []
    functions = [tool.get("function") or {} for tool in payload.get("tools") or []]
    picked = _pick_tool(prompt, functions)
    if picked is None:
        return []
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:20]}",
            "type": "function",
            "function": {"name": picked["name"], "arguments": json.dumps(picked["arguments"])},
        }
    ]


def _arguments_for(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties") or {}
    arguments: dict[str, Any] = {}
    number = 2
    for name in schema.get("required") or []:
        kind = (properties.get(name) or {}).get("type")
        if kind in ("number", "integer"):
            arguments[name] = number
            number += 1
        elif kind == "boolean":
            arguments[name] = True
        else:
            arguments[name] = "hi"
    return arguments


def _last_role(payload: dict[str, Any]) -> str | None:
    messages = payload.get("messages") or []
    return messages[-1].get("role") if messages else None


def _tool_result(payload: dict[str, Any]) -> str | None:
    """The content of a trailing tool message, which the fake echoes back."""
    messages = payload.get("messages") or []
    if not messages or messages[-1].get("role") != "tool":
        return None
    content = messages[-1].get("content")
    if isinstance(content, list):
        content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def _is_judge_request(payload: dict[str, Any]) -> bool:
    return any(
        message.get("role") == "system" and "guardrail judge" in str(message.get("content"))
        for message in payload.get("messages") or []
    )


def _usage(prompt: str, answer: str) -> dict[str, int]:
    # Roughly four characters per token is close enough for cost arithmetic.
    prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = max(1, len(answer) // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


async def _maybe_fail(prompt: str) -> JSONResponse | None:
    if "__fail__" in prompt:
        return JSONResponse(
            {"error": {"message": "upstream is down", "type": "server_error"}},
            status_code=503,
        )
    if "__flaky__" in prompt:
        if _flaky_budget[prompt] > 0:
            _flaky_budget[prompt] -= 1
            return JSONResponse(
                {"error": {"message": "transient failure", "type": "server_error"}},
                status_code=503,
            )
        # Reset so a later run of the same prompt fails again.
        _flaky_budget[prompt] = 2
    if "__slow__" in prompt:
        await asyncio.sleep(1.5)
    return None


@app.post("/chat/completions")
async def chat_completions(request: Request) -> Any:
    payload = await request.json()
    prompt = _prompt_of(payload)

    failure = await _maybe_fail(prompt)
    if failure is not None:
        return failure

    tool_result = _tool_result(payload)
    answer = _answer(prompt) if tool_result is None else f"Tool result: {tool_result[:180]}"
    if _is_judge_request(payload):
        # Deterministic LLM-judge verdicts: "__unsafe__" is a violation.
        score = 1.0 if "__unsafe__" in prompt else 0.0
        answer = json.dumps({"score": score, "reason": "fake judge"})
    model = payload.get("model", "gpt-4o-mini")
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    tool_calls = _tool_calls(prompt, payload)

    if payload.get("stream"):

        async def event_stream() -> Any:
            first = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first)}\n\n"
            if tool_calls:
                for index, call in enumerate(tool_calls):
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{"index": index, **call}],
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.005)
            else:
                for word in answer.split(" "):
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": word + " "},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.005)
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }
                ],
                "usage": _usage(prompt, answer),
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None if tool_calls else answer,
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": _usage(prompt, answer),
    }


def _vector(text: Any) -> list[float]:
    # A stable bag-of-words pseudo-embedding: identical text embeds
    # identically (what the semantic cache relies on), and texts sharing
    # words land close together (what RAG retrieval relies on).
    dims = 256
    out = [0.0] * dims
    for word in re.findall(r"[a-z0-9]+", str(text).lower()):
        out[int(hashlib.sha256(word.encode()).hexdigest(), 16) % dims] += 1.0
    norm = sum(value * value for value in out) ** 0.5 or 1.0
    return [value / norm for value in out]


@app.post("/embeddings")
async def embeddings(request: Request) -> Any:
    payload = await request.json()
    raw = payload.get("input")
    items = raw if isinstance(raw, list) else [raw]

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": _vector(item)}
            for i, item in enumerate(items)
        ],
        "model": payload.get("model", "text-embedding-3-small"),
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }


@app.get("/models")
async def models() -> Any:
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4o", "object": "model", "owned_by": "fake"},
            {"id": "gpt-4o-mini", "object": "model", "owned_by": "fake"},
        ],
    }


# ---------------------------------------------------------------- Anthropic


def _anthropic_text(content: Any, kind: str = "text") -> str:
    if isinstance(content, str):
        return content if kind == "text" else ""
    parts: list[str] = []
    for block in content or []:
        if block.get("type") != kind:
            continue
        if kind == "text":
            parts.append(block.get("text", ""))
        else:
            parts.append(_anthropic_text(block.get("content"), "text"))
    return " ".join(parts)


def _anthropic_turn(payload: dict[str, Any]) -> tuple[str, str | None]:
    """The prompt (last user text) and the trailing tool result, if any."""
    prompt, tool_result = "", None
    messages = [m for m in payload.get("messages") or [] if m.get("role") == "user"]
    for message in reversed(messages):
        if text := _anthropic_text(message.get("content")):
            prompt = text
            break
    last = (payload.get("messages") or [{}])[-1]
    if last.get("role") == "user" and _has_block(last, "tool_result"):
        tool_result = _anthropic_text(last.get("content"), "tool_result")
    return prompt, tool_result


def _has_block(message: dict[str, Any], kind: str) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(b.get("type") == kind for b in content)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/anthropic/messages")
async def anthropic_messages(request: Request) -> Any:
    payload = await request.json()
    prompt, tool_result = _anthropic_turn(payload)
    failure = await _maybe_fail(prompt)
    if failure is not None:
        return failure

    answer = _answer(prompt) if tool_result is None else f"Tool result: {tool_result[:180]}"
    functions = [
        {"name": tool.get("name"), "parameters": tool.get("input_schema")}
        for tool in payload.get("tools") or []
    ]
    picked = _pick_tool(prompt, functions) if tool_result is None else None
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    model = payload.get("model", "claude")
    usage = _usage(prompt, answer)
    stop_reason = "tool_use" if picked else "end_turn"
    tool_block = (
        {"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:20]}", "name": picked["name"]}
        if picked
        else None
    )
    tool_input = picked["arguments"] if picked else {}

    if payload.get("stream"):

        async def event_stream() -> Any:
            yield _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": usage["prompt_tokens"], "output_tokens": 0},
                    },
                },
            )
            if tool_block is not None:
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {**tool_block, "input": {}},
                    },
                )
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(tool_input),
                        },
                    },
                )
            else:
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                for word in answer.split(" "):
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": word + " "},
                        },
                    )
                    await asyncio.sleep(0.005)
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason},
                    "usage": {"output_tokens": usage["completion_tokens"]},
                },
            )
            yield _sse("message_stop", {"type": "message_stop"})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    content = (
        [{**tool_block, "input": tool_input}]
        if tool_block is not None
        else [{"type": "text", "text": answer}]
    )
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
        },
    }


# ---------------------------------------------------------------- Gemini


def _gemini_turn(payload: dict[str, Any]) -> tuple[str, str | None]:
    """The prompt (last user text) and the trailing function response, if any."""
    contents = payload.get("contents") or []
    prompt, tool_result = "", None
    for content in reversed(contents):
        texts = [p["text"] for p in content.get("parts", []) if p.get("text")]
        if content.get("role") == "user" and texts:
            prompt = " ".join(texts)
            break
    if contents:
        responses = [
            p["functionResponse"] for p in contents[-1].get("parts", []) if "functionResponse" in p
        ]
        if responses:
            tool_result = str((responses[0].get("response") or {}).get("result", ""))
    return prompt, tool_result


def _gemini_usage(prompt: str, answer: str) -> dict[str, int]:
    usage = _usage(prompt, answer)
    return {
        "promptTokenCount": usage["prompt_tokens"],
        "candidatesTokenCount": usage["completion_tokens"],
        "totalTokenCount": usage["total_tokens"],
    }


@app.post("/gemini/models/{target}")
async def gemini_models_action(target: str, request: Request) -> Any:
    model, _, action = target.partition(":")
    payload = await request.json()
    if action == "embedContent":
        text = " ".join(p.get("text", "") for p in payload.get("content", {}).get("parts", []))
        return {
            "embedding": {"values": _vector(text)},
            "usageMetadata": {"promptTokenCount": 8, "totalTokenCount": 8},
        }
    if action not in ("generateContent", "streamGenerateContent"):
        return JSONResponse({"error": {"code": 404, "message": "unknown action"}}, 404)

    prompt, tool_result = _gemini_turn(payload)
    failure = await _maybe_fail(prompt)
    if failure is not None:
        return failure

    answer = _answer(prompt) if tool_result is None else f"Tool result: {tool_result[:180]}"
    functions = [
        declaration
        for tool in payload.get("tools") or []
        for declaration in tool.get("functionDeclarations") or []
    ]
    picked = _pick_tool(prompt, functions) if tool_result is None else None
    usage = _gemini_usage(prompt, answer)
    parts = (
        [{"functionCall": {"name": picked["name"], "args": picked["arguments"]}}]
        if picked
        else [{"text": answer}]
    )

    if action == "streamGenerateContent":

        async def event_stream() -> Any:
            def fragment(parts: list[dict[str, Any]], **extra: Any) -> str:
                candidate = {"content": {"role": "model", "parts": parts}, "index": 0}
                body: dict[str, Any] = {"candidates": [candidate], "modelVersion": model}
                if "finishReason" in extra:
                    candidate["finishReason"] = extra.pop("finishReason")
                return f"data: {json.dumps({**body, **extra})}\r\n\r\n"

            if picked:
                yield fragment(parts)
            else:
                for word in answer.split(" "):
                    yield fragment([{"text": word + " "}])
                    await asyncio.sleep(0.005)
            yield fragment([{"text": ""}], finishReason="STOP", usageMetadata=usage)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "candidates": [
            {"content": {"role": "model", "parts": parts}, "finishReason": "STOP", "index": 0}
        ],
        "usageMetadata": usage,
        "modelVersion": model,
    }


@app.get("/gemini/models")
async def gemini_models() -> Any:
    return {"models": [{"name": "models/gemini-2.5-flash"}, {"name": "models/text-embedding-004"}]}


# ---------------------------------------------------------------- Ollama
# Ollama's chat messages are OpenAI-shaped (roles, a "tool" role, OpenAI tool
# declarations), so the prompt and tool-result helpers above apply as-is.


@app.post("/ollama/api/chat")
async def ollama_chat(request: Request) -> Any:
    payload = await request.json()
    prompt = _prompt_of(payload)
    failure = await _maybe_fail(prompt)
    if failure is not None:
        return failure

    tool_result = _tool_result(payload)
    answer = _answer(prompt) if tool_result is None else f"Tool result: {tool_result[:180]}"
    tool_calls = [
        {
            "function": {
                "name": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"]),
            }
        }
        for call in _tool_calls(prompt, payload)
    ]
    model = payload.get("model", "llama3.2")
    usage = _usage(prompt, answer)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    final = {
        "model": model,
        "created_at": created_at,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": usage["prompt_tokens"],
        "eval_count": usage["completion_tokens"],
    }

    if payload.get("stream", True):

        async def ndjson() -> Any:
            def line(message: dict[str, Any]) -> str:
                body = {"model": model, "created_at": created_at, "done": False}
                return json.dumps({**body, "message": {"role": "assistant", **message}}) + "\n"

            if tool_calls:
                yield line({"content": "", "tool_calls": tool_calls})
            else:
                for word in answer.split(" "):
                    yield line({"content": word + " "})
                    await asyncio.sleep(0.005)
            yield json.dumps({**final, "message": {"role": "assistant", "content": ""}}) + "\n"

        return StreamingResponse(ndjson(), media_type="application/x-ndjson")

    message: dict[str, Any] = {"role": "assistant", "content": "" if tool_calls else answer}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {**final, "message": message}


@app.post("/ollama/api/embed")
async def ollama_embed(request: Request) -> Any:
    payload = await request.json()
    raw = payload.get("input")
    items = raw if isinstance(raw, list) else [raw]
    return {
        "model": payload.get("model", "nomic-embed-text"),
        "embeddings": [_vector(item) for item in items],
        "prompt_eval_count": 8,
    }


@app.get("/ollama/api/tags")
async def ollama_tags() -> Any:
    return {"models": [{"name": "llama3.2:latest"}, {"name": "nomic-embed-text:latest"}]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4100)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
