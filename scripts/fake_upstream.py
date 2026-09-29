"""A tiny OpenAI-compatible upstream for end-to-end evaluation.

The evaluation harness needs a provider that answers deterministically, fails
on demand, and costs nothing. Pointing the gateway's OpenAI adapter at this
server exercises the *entire* real path — dialect translation, auth, routing,
retry, fallback, caching, guardrails, accounting — without a network call or an
API key.

Run it, point ``OPENAI_BASE_URL`` at it, and drive the gateway normally:

    uv run python scripts/fake_upstream.py --port 4100

Behaviour is controlled by the prompt text, so a test can request a failure
without any out-of-band configuration:

  ``__fail__``      always returns 503, to exercise retry and fallback
  ``__flaky__``     fails twice per prompt, then succeeds
  ``__slow__``      sleeps 1.5s before responding
  ``__tool__``      returns a tool call for the first declared tool
                    (``__tool__:<name>`` picks one; required args are filled in)
  ``__unsafe__``    scored 1.0 by LLM-judge guardrail requests (else 0.0)

A trailing tool message is answered with ``Tool result: <content>``.
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
                return " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
    return ""


def _answer(prompt: str) -> str:
    # Deterministic, so a semantic-cache hit is detectable by comparing bodies.
    return f"Reply to: {prompt.strip()[:180]}"


def _tool_calls(prompt: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a synthetic tool call when the prompt asks for one.

    Triggered by ``__tool__`` (the first declared tool) or ``__tool__:<name>``
    (the first tool whose name ends with ``<name>``), so tool-calling and the
    gateway's agent loop can be exercised without a real provider. Required
    arguments are filled from the tool's schema: numbers count up from 2,
    strings are "hi".
    """
    if "__tool__" not in prompt or _last_role(payload) == "tool":
        return []
    tools = payload.get("tools") or []
    if not tools:
        return []
    match = re.search(r"__tool__:([A-Za-z0-9_-]+)", prompt)
    functions = [tool.get("function") or {} for tool in tools]
    if match:
        functions = [f for f in functions if str(f.get("name", "")).endswith(match.group(1))]
    if not functions or not functions[0].get("name"):
        return []
    function = functions[0]
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:20]}",
            "type": "function",
            "function": {
                "name": function["name"],
                "arguments": json.dumps(_arguments_for(function.get("parameters") or {})),
            },
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
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
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


@app.post("/embeddings")
async def embeddings(request: Request) -> Any:
    payload = await request.json()
    raw = payload.get("input")
    items = raw if isinstance(raw, list) else [raw]

    def vector(text: Any) -> list[float]:
        # A stable bag-of-words pseudo-embedding: identical text embeds
        # identically (what the semantic cache relies on), and texts sharing
        # words land close together (what RAG retrieval relies on).
        dims = 256
        out = [0.0] * dims
        for word in re.findall(r"[a-z0-9]+", str(text).lower()):
            out[int(hashlib.sha256(word.encode()).hexdigest(), 16) % dims] += 1.0
        norm = sum(value * value for value in out) ** 0.5 or 1.0
        return [value / norm for value in out]

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vector(item)}
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4100)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
