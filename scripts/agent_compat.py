"""Validate the gateway against the official OpenAI and Anthropic SDKs.

Coding agents (opencode, Claude Code, Cursor) drive the gateway through these
SDKs rather than raw HTTP, so SDK-level acceptance is what actually decides
whether an agent works. The SDKs are strict about response shapes in ways a
hand-written ``curl`` check is not: they validate types, reassemble streaming
deltas, and accumulate tool-call arguments across chunks.

Usage::

    uv run python scripts/agent_compat.py --base-url http://localhost:4030 \\
        --api-key sk-eval-master --model eval-chat

Exit code is 0 when every check passes and 1 otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@dataclass
class Results:
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    def check(self, name: str, fn: Any) -> None:
        try:
            detail = fn()
        except Exception as exc:  # report, never abort the suite
            self.failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            return
        self.passed.append(name)
        print(f"  ok    {name}{f' — {detail}' if detail else ''}")


def openai_checks(results: Results, base_url: str, api_key: str, model: str) -> None:
    client = openai.OpenAI(base_url=f"{base_url}/v1", api_key=api_key, max_retries=0)
    print("\nOpenAI SDK")

    def models() -> str:
        listed = list(client.models.list())
        if not listed:
            raise AssertionError("no models returned")
        if not any(m.id == model for m in listed):
            raise AssertionError(f"{model} missing from /v1/models")
        return f"{len(listed)} models, {model} present"

    def unary() -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello."}],
        )
        content = response.choices[0].message.content
        if not content:
            raise AssertionError("empty content")
        if response.usage is None or response.usage.total_tokens <= 0:
            raise AssertionError("usage not reported")
        return f"{response.usage.total_tokens} tokens"

    def streaming() -> str:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count to five."}],
            stream=True,
        )
        chunks = 0
        text = ""
        for chunk in stream:
            chunks += 1
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                text += delta
        if chunks < 2:
            raise AssertionError(f"expected multiple chunks, got {chunks}")
        if not text.strip():
            raise AssertionError("stream produced no text")
        return f"{chunks} chunks, {len(text)} chars"

    def tool_call() -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "__tool__ weather in Berlin"}],
            tools=[WEATHER_TOOL],  # type: ignore[list-item]
        )
        calls = response.choices[0].message.tool_calls
        if not calls:
            raise AssertionError("no tool_calls returned")
        if calls[0].function.name != "get_weather":
            raise AssertionError(f"wrong tool: {calls[0].function.name}")
        if response.choices[0].finish_reason != "tool_calls":
            raise AssertionError(f"finish_reason={response.choices[0].finish_reason}")
        return f"{calls[0].function.name} via {response.choices[0].finish_reason}"

    def tool_call_streaming() -> str:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "__tool__ weather in Berlin"}],
            tools=[WEATHER_TOOL],  # type: ignore[list-item]
            stream=True,
        )
        names: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            for call in chunk.choices[0].delta.tool_calls or []:
                if call.function and call.function.name:
                    names.append(call.function.name)
        if "get_weather" not in names:
            raise AssertionError(f"tool name never streamed (saw {names})")
        return "get_weather reassembled from deltas"

    def multi_turn_tool_result() -> str:
        """The turn that agents actually depend on: send the tool result back.

        The assertion is that the gateway accepts an assistant message carrying
        ``tool_calls`` followed by a ``role: tool`` message and still produces a
        valid assistant turn. It deliberately does not require text content: the
        trigger prompt is still the last user message, so a stub upstream may
        legitimately call the tool again, and that loop is what a real agent
        does too.
        """
        first = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "__tool__ weather in Berlin"}],
            tools=[WEATHER_TOOL],  # type: ignore[list-item]
        )
        call = (first.choices[0].message.tool_calls or [None])[0]
        if call is None:
            raise AssertionError("no tool call to respond to")
        second = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "__tool__ weather in Berlin"},
                first.choices[0].message,  # type: ignore[list-item]
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"temp_c": 12}',
                },
            ],
            tools=[WEATHER_TOOL],  # type: ignore[list-item]
        )
        message = second.choices[0].message
        if message.role != "assistant":
            raise AssertionError(f"unexpected role {message.role}")
        if not message.content and not message.tool_calls:
            raise AssertionError("neither content nor tool_calls after tool result")
        kind = "content" if message.content else "another tool call"
        return f"tool result accepted, replied with {kind}"

    def embeddings() -> str:
        response = client.embeddings.create(model="text-embedding-3-small", input="hello world")
        vector = response.data[0].embedding
        if len(vector) < 8:
            raise AssertionError(f"suspicious embedding length {len(vector)}")
        return f"dim={len(vector)}"

    results.check("models.list", models)
    results.check("chat.completions (unary)", unary)
    results.check("chat.completions (streaming)", streaming)
    results.check("tool call (unary)", tool_call)
    results.check("tool call (streaming)", tool_call_streaming)
    results.check("tool result round-trip", multi_turn_tool_result)
    results.check("embeddings", embeddings)


def anthropic_checks(results: Results, base_url: str, api_key: str, model: str) -> None:
    client = anthropic.Anthropic(base_url=base_url, api_key=api_key, max_retries=0)
    print("\nAnthropic SDK")

    def unary() -> str:
        message = client.messages.create(
            model=model,
            max_tokens=128,
            messages=[{"role": "user", "content": "Say hello."}],
        )
        if not message.content:
            raise AssertionError("empty content blocks")
        block = message.content[0]
        text = getattr(block, "text", "")
        if not text:
            raise AssertionError("first block carries no text")
        if message.usage.input_tokens <= 0:
            raise AssertionError("usage not reported")
        return f"in={message.usage.input_tokens} out={message.usage.output_tokens}"

    def system_prompt() -> str:
        message = client.messages.create(
            model=model,
            max_tokens=128,
            system="You are terse.",
            messages=[{"role": "user", "content": "Hello."}],
        )
        if not message.content:
            raise AssertionError("empty content with system prompt")
        return "system prompt accepted"

    def streaming() -> str:
        text = ""
        events = 0
        with client.messages.stream(
            model=model,
            max_tokens=128,
            messages=[{"role": "user", "content": "Count to five."}],
        ) as stream:
            for delta in stream.text_stream:
                events += 1
                text += delta
            final = stream.get_final_message()
        if not text.strip():
            raise AssertionError("stream produced no text")
        if final.usage.output_tokens <= 0:
            raise AssertionError("final message missing usage")
        return f"{events} deltas, {len(text)} chars"

    results.check("messages (unary)", unary)
    results.check("messages (system prompt)", system_prompt)
    results.check("messages (streaming)", streaming)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:4030")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="eval-chat")
    parser.add_argument(
        "--skip-anthropic",
        action="store_true",
        help="Skip the Anthropic dialect checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable summary on stdout; the progress log goes to stderr.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    results = Results()

    log = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
    with log:
        print(f"Agent compatibility — {base_url}, model={args.model}")
        openai_checks(results, base_url, args.api_key, args.model)
        if not args.skip_anthropic:
            anthropic_checks(results, base_url, args.api_key, args.model)

    total = len(results.passed) + len(results.failed)
    if args.json:
        print(
            json.dumps(
                {
                    "checks": total,
                    "passed": len(results.passed),
                    "failed": [{"name": name, "detail": detail} for name, detail in results.failed],
                    "pass_rate": round(len(results.passed) / total, 4) if total else 0.0,
                },
                indent=2,
            )
        )
        return 1 if results.failed else 0
    print(f"\n{len(results.passed)}/{total} checks passed")
    if results.failed:
        print("\nFailures:")
        for name, detail in results.failed:
            print(f"  - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
