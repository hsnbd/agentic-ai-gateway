"""Server-side agent mode: the gateway runs MCP tools for the caller (`aigw.mcp`)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from app.core.pipeline import Executor, RequestContext, _replay_as_stream
from app.core.schemas import ChatResponse, StreamChunk
from app.tools.executor import ToolExecutor, run_tool_loop

if TYPE_CHECKING:
    from app.mcp.registry import McpRegistry


class AgenticExecutor:
    """Wraps the resilient executor with an MCP tool loop.

    Requests without `aigw.mcp` pass straight through. With it, the MCP tools
    are offered to the model alongside any the client sent, and every MCP call
    the model makes is executed and fed back until it answers, asks for a
    client-side tool, or hits `max_iterations`. Because the loop lives in the
    executor, auth, guardrails, cache and accounting run once per request, and
    each hop still gets routing, retries and fallback.
    """

    def __init__(self, inner: Executor) -> None:
        self._inner = inner

    async def execute(self, ctx: RequestContext) -> ChatResponse:
        options = ctx.request.mcp
        if options is None:
            return await self._inner.execute(ctx)

        registry: McpRegistry = ctx.state.components["mcp_registry"]
        mcp_tools = await registry.tools_for(options.servers)
        client_names = {tool.function.name for tool in ctx.request.tools}
        offered = [tool for tool in mcp_tools if tool.function.name not in client_names]
        streaming = ctx.request.stream
        # Hops are unary even for a streamed request; the answer is replayed.
        ctx.request = ctx.request.model_copy(
            update={"tools": [*ctx.request.tools, *offered], "stream": False}
        )
        try:
            return await run_tool_loop(
                ctx,
                self._inner.execute,
                ToolExecutor(registry),
                max_iterations=options.max_iterations,
                mcp_names={tool.function.name for tool in offered},
            )
        finally:
            ctx.request.stream = streaming

    def execute_stream(self, ctx: RequestContext) -> AsyncIterator[StreamChunk]:
        if ctx.request.mcp is None:
            return self._inner.execute_stream(ctx)
        return self._agent_stream(ctx)

    async def _agent_stream(self, ctx: RequestContext) -> AsyncIterator[StreamChunk]:
        response = await self.execute(ctx)
        async for chunk in _replay_as_stream(response):
            chunk.deployment_id = response.deployment_id
            yield chunk
