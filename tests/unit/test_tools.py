from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.pipeline import RequestContext
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    Choice,
    FinishReason,
    FunctionDef,
    Message,
    Role,
    ToolCall,
    ToolDef,
)
from app.tools.executor import ToolExecutor, run_agentic_loop
from app.tools.validation import validate_arguments


@pytest.fixture
def sample_tool() -> ToolDef:
    return ToolDef(
        function=FunctionDef(
            name="test__lookup",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    )


def test_validate_accepts_valid_arguments(sample_tool: ToolDef) -> None:
    assert validate_arguments(sample_tool, '{"query":"hello"}') == {"query": "hello"}


def test_validate_repairs_fences_unquoted_keys_values_and_trailing_comma(
    sample_tool: ToolDef,
) -> None:
    parsed = validate_arguments(sample_tool, "```json\n{query: hello,}\n```")
    assert parsed == {"query": "hello"}


def test_validate_returns_structured_missing_required_error(sample_tool: ToolDef) -> None:
    result = validate_arguments(sample_tool, "{}")
    assert result["error"] == "arguments.query is required"
    assert result["expected_schema"] == sample_tool.function.parameters


class _Registry:
    def __init__(self, *, failing: set[str] | None = None) -> None:
        self.failing = failing or set()
        self.active = 0
        self.max_active = 0

    async def tools_for(self) -> list[ToolDef]:
        return [
            ToolDef(function=FunctionDef(name=f"demo__{name}", parameters={"type": "object"}))
            for name in ("ok", "bad")
        ]

    def server_name(self, server_id: str) -> str:
        return server_id

    def resolve(self, name: str) -> tuple[str, str]:
        return "server", name.split("__", 1)[1]

    async def call_tool(
        self, server_id: str, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.03)
            if name in self.failing:
                raise RuntimeError("tool failed")
            return {"content": [{"type": "text", "text": name}]}
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_execute_all_is_concurrent_and_isolates_failure() -> None:
    registry = _Registry(failing={"bad"})
    executor = ToolExecutor(registry)  # type: ignore[arg-type]
    results = await executor.execute_all(
        [
            ToolCall(id="a", name="demo__ok", arguments="{}"),
            ToolCall(id="b", name="demo__bad", arguments="{}"),
        ]
    )
    assert len(results) == 2
    assert registry.max_active == 2
    assert results[0].tool_call_id == "a"
    assert results[0].text() == "ok"
    assert results[1].tool_call_id == "b"
    assert "tool failed" in results[1].text()


def _context(registry: _Registry) -> RequestContext:
    request = ChatRequest(model="test", messages=[Message(role=Role.USER, content="go")])
    state = SimpleNamespace(components={"mcp_registry": registry})
    return RequestContext(request=request, state=state)  # type: ignore[arg-type]


def _response(*, tool_call: bool = False) -> ChatResponse:
    message = Message(
        role=Role.ASSISTANT,
        content=None if tool_call else "done",
        tool_calls=[ToolCall(id="call-1", name="demo__ok", arguments="{}")] if tool_call else [],
    )
    return ChatResponse(
        model="test",
        choices=[
            Choice(
                message=message,
                finish_reason=FinishReason.TOOL_CALLS if tool_call else FinishReason.STOP,
            )
        ],
    )


@pytest.mark.asyncio
async def test_agentic_loop_terminates_on_normal_stop() -> None:
    ctx = _context(_Registry())
    calls = 0

    async def pipeline(_: RequestContext) -> ChatResponse:
        nonlocal calls
        calls += 1
        return _response()

    result = await run_agentic_loop(ctx, pipeline)
    assert calls == 1
    assert result.stop_reason == "completed"


@pytest.mark.asyncio
async def test_agentic_loop_stops_at_iteration_limit() -> None:
    ctx = _context(_Registry())
    calls = 0

    async def pipeline(_: RequestContext) -> ChatResponse:
        nonlocal calls
        calls += 1
        return _response(tool_call=True)

    result = await run_agentic_loop(ctx, pipeline, max_iterations=2)
    assert calls == 2
    assert result.stop_reason == "max_iterations"


@pytest.mark.asyncio
async def test_agentic_loop_detects_identical_tool_call() -> None:
    ctx = _context(_Registry())
    calls = 0

    async def pipeline(_: RequestContext) -> ChatResponse:
        nonlocal calls
        calls += 1
        return _response(tool_call=calls < 3)

    await run_agentic_loop(ctx, pipeline, max_iterations=4)
    assert any(
        message.role == Role.SYSTEM and "identical arguments" in message.text()
        for message in ctx.request.messages
    )
    assert calls == 3
