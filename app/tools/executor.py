from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from app.core.errors import InvalidRequestError
from app.core.pipeline import RequestContext
from app.core.schemas import ChatResponse, FinishReason, Message, Role, ToolCall
from app.mcp.registry import McpRegistry
from app.mcp.translate import tool_result_to_message
from app.tools.validation import validate_arguments


class ToolExecutor:
    def __init__(self, mcp_registry: McpRegistry, *, max_concurrency: int = 8) -> None:
        self.mcp_registry = mcp_registry
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def execute(self, tool_call: ToolCall) -> Message:
        tool_defs = await self.mcp_registry.tools_for()
        tool = next((item for item in tool_defs if item.function.name == tool_call.name), None)
        if tool is None:
            return _tool_error(tool_call.id, f"Tool {tool_call.name!r} is unavailable")
        arguments = validate_arguments(tool, tool_call.arguments)
        if _is_validation_error(arguments):
            return _tool_error(tool_call.id, json.dumps(arguments, ensure_ascii=False))
        try:
            server_id, original_name = self.mcp_registry.resolve(tool_call.name)
            async with self._semaphore:
                result = await self.mcp_registry.call_tool(server_id, original_name, arguments)
            return tool_result_to_message(tool_call.id, result)
        except Exception as exc:
            return _tool_error(tool_call.id, f"Tool execution failed: {exc}")

    async def execute_all(self, tool_calls: Sequence[ToolCall]) -> list[Message]:
        results = await asyncio.gather(
            *(self.execute(tool_call) for tool_call in tool_calls),
            return_exceptions=True,
        )
        messages: list[Message] = []
        for tool_call, result in zip(tool_calls, results, strict=True):
            if isinstance(result, BaseException):
                messages.append(_tool_error(tool_call.id, f"Tool execution failed: {result}"))
            else:
                messages.append(result)
        return messages


async def run_agentic_loop(
    ctx: RequestContext,
    pipeline_run: Callable[[RequestContext], Awaitable[ChatResponse]],
    *,
    max_iterations: int = 10,
) -> ChatResponse:
    """Run bounded model/tool hops.

    Messages and tool output are appended without compaction, so context can grow
    with each iteration; callers should choose a cap appropriate to model limits.
    """
    if max_iterations < 1:
        raise InvalidRequestError("max_iterations must be at least 1")
    executor = ToolExecutor(ctx.state.components["mcp_registry"])
    seen_calls: set[tuple[str, str]] = set()

    for iteration in range(max_iterations):
        response = await pipeline_run(ctx)
        choice = response.choices[0] if response.choices else None
        if choice is None or choice.finish_reason != FinishReason.TOOL_CALLS:
            object.__setattr__(response, "stop_reason", "completed")
            return response
        if iteration == max_iterations - 1:
            object.__setattr__(response, "stop_reason", "max_iterations")
            return response

        ctx.request.messages.append(choice.message)
        repeated: list[ToolCall] = []
        for call in choice.message.tool_calls:
            signature = (call.name, _normalized_arguments(call.arguments))
            if signature in seen_calls:
                repeated.append(call)
            seen_calls.add(signature)
        ctx.request.messages.extend(await executor.execute_all(choice.message.tool_calls))
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

    object.__setattr__(response, "stop_reason", "max_iterations")
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
