"""Who may call which MCP tools, and the identity a tool call is audited under.

``McpScope`` comes from the caller's virtual key (``allowed_mcp_servers`` and
``allowed_tools``); the master key and console admins are unrestricted.
``ToolCallContext`` carries the request, key, team, and guardrail policy a call
is checked and logged against, for both the agent loop and direct calls.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.pipeline import RequestContext
    from app.mcp.registry import McpRegistry


@dataclass(frozen=True)
class McpScope:
    #: Server ids or names; empty means every server.
    servers: tuple[str, ...] = ()
    #: Namespaced tool names (``alias__tool``) with ``*`` wildcards; empty means every tool.
    tools: tuple[str, ...] = ()

    @classmethod
    def for_key(cls, key: Any) -> McpScope:
        if key is None:
            return UNRESTRICTED_SCOPE
        return cls(
            servers=tuple(getattr(key, "allowed_mcp_servers", None) or ()),
            tools=tuple(getattr(key, "allowed_tools", None) or ()),
        )

    def permits_server(self, server_id: str, name: str | None) -> bool:
        return not self.servers or server_id in self.servers or name in self.servers

    def permits_tool(self, namespaced_name: str) -> bool:
        return not self.tools or any(
            fnmatch.fnmatchcase(namespaced_name, pattern) for pattern in self.tools
        )

    def allows(self, registry: McpRegistry, namespaced_name: str) -> bool:
        """Whether a discovered tool (``alias__tool``) is within this scope."""
        server_id, _ = registry.resolve(namespaced_name)
        return self.permits_server(server_id, registry.server_name(server_id)) and (
            self.permits_tool(namespaced_name)
        )


UNRESTRICTED_SCOPE = McpScope()


@dataclass(frozen=True)
class ToolCallContext:
    state: Any
    #: "agent" for the aigw.mcp loop, "direct" for POST /v1/mcp/tools/call.
    source: str
    request_id: str | None = None
    key_id: str | None = None
    team_id: str | None = None
    policy: str = "default"

    @classmethod
    def for_request(cls, ctx: RequestContext) -> ToolCallContext:
        return cls(
            state=ctx.state,
            source="agent",
            request_id=ctx.request_id,
            key_id=None if ctx.key_id == "master" else ctx.key_id,
            team_id=ctx.team_id,
            policy=ctx.request.guardrail_policy or "default",
        )
