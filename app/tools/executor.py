from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from app.core.errors import InvalidRequestError
from app.core.pipeline import RequestContext
from app.core.schemas import ChatResponse, FinishReason, Message, Role, ToolCall, Usage
from app.mcp.registry import McpRegistry
from app.mcp.translate import tool_result_to_message
from app.tools.validation import validate_arguments


class ToolExecutor:
    def __init__(self, mcp_registry: McpRegistry, *, max_concurrency: int = 8) -> None:
        self.mcp_registry = mcp_registry
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def execute(self, tool_call: ToolCall) -> Message:
        message, _ = await self.execute_detailed(tool_call)
        return message

    async def execute_detailed(self, tool_call: ToolCall) -> tuple[Message, bool]:
        """Run one call; returns the tool message and whether it is an error."""
        tool_defs = await self.mcp_registry.tools_for()
        tool = next((item for item in tool_defs if item.function.name == tool_call.name), None)
        if tool is None:
            return _tool_error(tool_call.id, f"Tool {tool_call.name!r} is unavailable"), True
        arguments = validate_arguments(tool, tool_call.arguments)
        if _is_validation_error(arguments):
            return _tool_error(tool_call.id, json.dumps(arguments, ensure_ascii=False)), True
        try:
            server_id, original_name = self.mcp_registry.resolve(tool_call.name)
            async with self._semaphore:
                result = await self.mcp_registry.call_tool(server_id, original_name, arguments)
            return tool_result_to_message(tool_call.id, result), bool(result.get("isError"))
        except Exception as exc:
            return _tool_error(tool_call.id, f"Tool execution failed: {exc}"), True

    async def execute_all(self, tool_calls: Sequence[ToolCall]) -> list[Message]:
        return [message for message, _ in await self.execute_all_detailed(tool_calls)]

    async def execute_all_detailed(
        self, tool_calls: Sequence[ToolCall]
    ) -> list[tuple[Message, dict[str, Any]]]:
        """Run calls concurrently; each result carries an audit record."""

        async def timed(call: ToolCall) -> tuple[Message, dict[str, Any]]:
            started = time.perf_counter()
            try:
                message, is_error = await self.execute_detailed(call)
            except BaseException as exc:  # isolate one call's failure from the rest
                message, is_error = _tool_error(call.id, f"Tool execution failed: {exc}"), True
            return message, {
                "name": call.name,
                "ms": round((time.perf_counter() - started) * 1000, 3),
                "is_error": is_error,
            }

        return list(await asyncio.gather(*(timed(call) for call in tool_calls)))


async def run_agentic_loop(
    ctx: RequestContext,
    pipeline_run: Callable[[RequestContext], Awaitable[ChatResponse]],
    *,
    max_iterations: int = 10,
) -> ChatResponse:
    """Run bounded model/tool hops, executing every tool call through MCP.

    The gateway's own agent mode uses `app.tools.agent.AgenticExecutor`, which
    calls `run_tool_loop` inside the pipeline so pre- and post-stages run once.
    """
    executor = ToolExecutor(ctx.state.components["mcp_registry"])
    response = await run_tool_loop(ctx, pipeline_run, executor, max_iterations=max_iterations)
    response.stop_reason = ctx.stop_reason
    return response


async def run_tool_loop(
    ctx: RequestContext,
    call_model: Callable[[RequestContext], Awaitable[ChatResponse]],
    executor: ToolExecutor,
    *,
    max_iterations: int,
    mcp_names: set[str] | None = None,
) -> ChatResponse:
    """Alternate model calls and tool execution until the model stops asking.

    Ends with `ctx.stop_reason` set to "completed", "max_iterations", or
    "client_tool_call" (a call the gateway cannot run, which is returned to the
    caller unexecuted). `mcp_names` limits which calls the gateway may execute;
    None means every call goes to MCP. Usage is summed across hops so the
    request is costed and budgeted for all of them.

    Messages and tool output are appended without compaction, so context grows
    with each hop; callers should choose a cap appropriate to model limits.
    """
    if max_iterations < 1:
        raise InvalidRequestError("max_iterations must be at least 1")
    seen_calls: set[tuple[str, str]] = set()
    total = Usage()
    response: ChatResponse | None = None

    # Every path through the body breaks on the last iteration at the latest.
    for iteration in range(max_iterations):  # pragma: no branch
        response = await call_model(ctx)
        total = total + response.usage
        choice = response.choices[0] if response.choices else None
        if choice is None or choice.finish_reason != FinishReason.TOOL_CALLS:
            ctx.stop_reason = "completed"
            break
        calls = choice.message.tool_calls
        if mcp_names is not None and any(call.name not in mcp_names for call in calls):
            ctx.stop_reason = "client_tool_call"
            break
        if iteration == max_iterations - 1:
            ctx.stop_reason = "max_iterations"
            break

        ctx.request.messages.append(choice.message)
        repeated: list[ToolCall] = []
        for call in calls:
            signature = (call.name, _normalized_arguments(call.arguments))
            if signature in seen_calls:
                repeated.append(call)
            seen_calls.add(signature)
        for message, record in await executor.execute_all_detailed(calls):
            ctx.request.messages.append(message)
            ctx.tool_executions.append({**record, "iteration": iteration + 1})
        if repeated:
            names = ", ".join(sorted({call.name for call in repeated}))
            ctx.request.messages.append(
                Message(
                    role=Role.SYSTEM,
                    content=(
                        f"You repeated the same tool call ({names}) with identical arguments. "
                        "Use the previous result or choose a different action."
                    ),
                )
            )

    assert response is not None  # max_iterations >= 1
    response.usage = total
    return response


def _tool_error(tool_call_id: str, message: str) -> Message:
    return Message(role=Role.TOOL, content=message, tool_call_id=tool_call_id)


def _is_validation_error(arguments: dict[str, Any]) -> bool:
    return set(arguments) == {"error", "expected_schema"}


def _normalized_arguments(raw_arguments: str) -> str:
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return raw_arguments.strip()
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
