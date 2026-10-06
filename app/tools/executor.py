from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.core.errors import InvalidRequestError
from app.core.pipeline import RequestContext
from app.core.schemas import ChatResponse, FinishReason, Message, Role, ToolCall, Usage
from app.db.models import ToolCallLog
from app.guardrails.base import Action, GuardrailResult, Phase
from app.guardrails.registry import GuardrailRegistry
from app.guardrails.stage import persist_violations
from app.mcp.registry import McpRegistry
from app.mcp.translate import tool_result_to_message
from app.observability.metrics import MCP_TOOL_CALL_SECONDS, MCP_TOOL_CALLS
from app.tools.governance import UNRESTRICTED_SCOPE, McpScope, ToolCallContext
from app.tools.validation import validate_arguments

logger = logging.getLogger(__name__)

#: Used when there are no settings (tests, direct construction).
_DEFAULT_MAX_RESULT_CHARS = 20000


@dataclass
class _Audit:
    tool: str
    arguments: str
    status: str = "ok"
    server_id: str | None = None
    result_chars: int = 0
    truncated: bool = False
    guardrail: str | None = None
    error: str | None = None


class ToolExecutor:
    """Runs MCP tool calls under the caller's scope, guardrails, and audit log.

    Every call is checked, in order: the tool exists, the caller's key may use
    its server and name, the arguments match the schema, input guardrails pass
    (for policies with ``apply_to_tools``), then the server is called, output
    guardrails screen the result, and oversized results are truncated. One
    ``ToolCallLog`` row records the outcome. Failures become tool messages, so
    the model can react instead of the request failing.
    """

    def __init__(
        self,
        mcp_registry: McpRegistry,
        *,
        max_concurrency: int = 8,
        scope: McpScope = UNRESTRICTED_SCOPE,
        call_context: ToolCallContext | None = None,
    ) -> None:
        self.mcp_registry = mcp_registry
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._scope = scope
        self._context = call_context

    async def execute(self, tool_call: ToolCall) -> Message:
        message, _ = await self.execute_detailed(tool_call)
        return message

    async def execute_detailed(self, tool_call: ToolCall) -> tuple[Message, bool]:
        """Run one call; returns the tool message and whether it is an error."""
        audit = _Audit(tool=tool_call.name, arguments=tool_call.arguments)
        started = time.perf_counter()
        message, is_error = await self._run(tool_call, audit)
        await self._record(audit, (time.perf_counter() - started) * 1000)
        return message, is_error

    async def _run(self, tool_call: ToolCall, audit: _Audit) -> tuple[Message, bool]:
        def refuse(status: str, text: str) -> tuple[Message, bool]:
            audit.status = status
            audit.error = text
            return _tool_error(tool_call.id, text), True

        tool_defs = await self.mcp_registry.tools_for()
        tool = next((item for item in tool_defs if item.function.name == tool_call.name), None)
        if tool is None:
            return refuse("unavailable", f"Tool {tool_call.name!r} is unavailable")
        server_id, original_name = self.mcp_registry.resolve(tool_call.name)
        audit.server_id = server_id
        if not self._scope.allows(self.mcp_registry, tool_call.name):
            return refuse("denied", f"Tool {tool_call.name!r} is not allowed for this API key")
        arguments = validate_arguments(tool, tool_call.arguments)
        if _is_validation_error(arguments):
            return refuse("invalid", json.dumps(arguments, ensure_ascii=False))

        guard, policy = self._guardrails()
        if guard is not None:
            checked = await self._screen(
                guard, policy, json.dumps(arguments, ensure_ascii=False), Phase.INPUT, audit
            )
            if checked.blocked:
                return refuse("blocked", f"Tool call blocked by guardrail rule {audit.guardrail!r}")
            if checked.redacted:
                arguments = _map_strings(
                    arguments, lambda value: guard.redact_text(policy, value, Phase.INPUT)
                )

        try:
            async with self._semaphore:
                result = await self.mcp_registry.call_tool(server_id, original_name, arguments)
        except Exception as exc:
            return refuse("failed", f"Tool execution failed: {exc}")
        message = tool_result_to_message(tool_call.id, result)
        is_error = bool(result.get("isError"))
        audit.status = "tool_error" if is_error else "ok"
        content = message.text()
        audit.result_chars = len(content)

        if guard is not None:
            checked = await self._screen(guard, policy, content, Phase.OUTPUT, audit)
            if checked.blocked:
                audit.status = "blocked"
                content = f"[Tool result withheld by guardrail rule {audit.guardrail!r}]"
                is_error = True
            elif checked.redacted:
                content = checked.text

        limit = self._max_result_chars()
        if len(content) > limit:
            audit.truncated = True
            dropped = len(content) - limit
            content = f"{content[:limit]}\n[Tool result truncated: {dropped} more characters]"
        message.content = content
        return message, is_error

    def _guardrails(self) -> tuple[GuardrailRegistry | None, str]:
        """The registry and policy to screen tool traffic with, if the policy asks for it."""
        if self._context is None:
            return None, "default"
        registry: GuardrailRegistry | None = self._context.state.components.get("guardrails")
        policy = self._context.policy
        if registry is None or not registry.get_policy(policy).apply_to_tools:
            return None, policy
        return registry, policy

    async def _screen(
        self, guard: GuardrailRegistry, policy: str, text: str, phase: Phase, audit: _Audit
    ) -> GuardrailResult:
        assert self._context is not None  # _guardrails only returns a registry with a context
        result = await guard.evaluate(policy, text, phase)
        if result.matches:
            audit.guardrail = next(
                (m.rule_name for m in result.matches if m.action == Action.BLOCK),
                result.matches[0].rule_name,
            )
            await persist_violations(
                self._context.state.db,
                result,
                request_id=self._context.request_id,
                key_id=self._context.key_id,
                details={
                    "tool": audit.tool,
                    "stage": "tool_arguments" if phase == Phase.INPUT else "tool_result",
                },
            )
        return result

    def _max_result_chars(self) -> int:
        settings = self._context.state.settings if self._context is not None else None
        return int(getattr(settings, "mcp_max_result_chars", _DEFAULT_MAX_RESULT_CHARS))

    async def _record(self, audit: _Audit, duration_ms: float) -> None:
        server = (
            self.mcp_registry.server_name(audit.server_id) if audit.server_id else None
        ) or "unknown"
        MCP_TOOL_CALLS.labels(server, audit.tool, audit.status).inc()
        MCP_TOOL_CALL_SECONDS.labels(server).observe(duration_ms / 1000)
        if self._context is None:
            return
        try:
            async with self._context.state.db.session() as session:
                session.add(
                    ToolCallLog(
                        request_id=self._context.request_id,
                        virtual_key_id=self._context.key_id,
                        team_id=self._context.team_id,
                        source=self._context.source,
                        server_id=audit.server_id,
                        tool=audit.tool,
                        status=audit.status,
                        duration_ms=round(duration_ms, 3),
                        arguments_hash=hashlib.sha256(
                            _normalized_arguments(audit.arguments).encode()
                        ).hexdigest(),
                        result_chars=audit.result_chars,
                        truncated=audit.truncated,
                        guardrail=audit.guardrail,
                        error=(audit.error or "")[:2000] or None,
                    )
                )
        except Exception:
            logger.exception("Could not record MCP tool call %s", audit.tool)

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


def _map_strings(value: Any, fn: Callable[[str], str]) -> Any:
    """Apply ``fn`` to every string inside a JSON-like value (redacting arguments)."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {key: _map_strings(item, fn) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_strings(item, fn) for item in value]
    return value


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
