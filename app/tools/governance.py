"""Who may call which MCP tools, and the identity a tool call is audited under.

``McpScope`` comes from the caller's virtual key (``allowed_mcp_servers`` and
``allowed_tools``); the master key and console admins are unrestricted.
``ToolCallContext`` carries the request, key, team, and guardrail policy a call
is checked and logged against, for both the agent loop and direct calls.

``ToolBindings`` comes from the key's ``tool_arg_bindings``: arguments the
gateway fills in from the caller's identity instead of letting the model choose
them, so an agent acting for one end user cannot act for another. Bound
arguments are also hidden from the tool schemas the model is shown.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.core.schemas import FunctionDef, ToolDef

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
class ToolBindings:
    """Sources are the request's end user, the calling key or team, or "none"
    (the argument is always removed, so the model can never supply it)."""

    #: ``(tool pattern, {argument: source})`` pairs; later patterns win.
    rules: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()

    @classmethod
    def for_key(cls, key: Any) -> ToolBindings:
        raw = getattr(key, "tool_arg_bindings", None) or {}
        return cls(
            rules=tuple(
                (pattern, tuple(sorted(arguments.items())))
                for pattern, arguments in raw.items()
                if isinstance(arguments, dict)
            )
        )

    def for_tool(self, namespaced_name: str) -> dict[str, str]:
        """``{argument: source}`` for every rule matching this tool."""
        bound: dict[str, str] = {}
        for pattern, arguments in self.rules:
            if fnmatch.fnmatchcase(namespaced_name, pattern):
                bound.update(arguments)
        return bound

    def hide(self, tool: ToolDef) -> ToolDef:
        """The tool as the model should see it: without its bound arguments."""
        parameters = tool.function.parameters
        declared = declared_arguments(parameters)
        bound = {name for name in self.for_tool(tool.function.name) if name in declared}
        if not bound:
            return tool
        properties = {
            name: value for name, value in parameters["properties"].items() if name not in bound
        }
        visible = {**parameters, "properties": properties}
        if isinstance(parameters.get("required"), list):
            visible["required"] = [name for name in parameters["required"] if name not in bound]
        return ToolDef(
            function=FunctionDef(
                name=tool.function.name,
                description=tool.function.description,
                parameters=visible,
            )
        )


NO_BINDINGS = ToolBindings()


def declared_arguments(parameters: dict[str, Any]) -> dict[str, Any]:
    properties = parameters.get("properties")
    return properties if isinstance(properties, dict) else {}


@dataclass(frozen=True)
class ToolCallContext:
    state: Any
    #: "agent" for the aigw.mcp loop, "direct" for POST /v1/mcp/tools/call.
    source: str
    request_id: str | None = None
    key_id: str | None = None
    team_id: str | None = None
    policy: str = "default"
    #: Who the agent acts for: the request's ``user`` or a direct call's ``end_user``.
    end_user: str | None = None
    bindings: ToolBindings = NO_BINDINGS

    @classmethod
    def for_request(cls, ctx: RequestContext) -> ToolCallContext:
        return cls(
            state=ctx.state,
            source="agent",
            request_id=ctx.request_id,
            key_id=None if ctx.key_id == "master" else ctx.key_id,
            team_id=ctx.team_id,
            policy=ctx.request.guardrail_policy or "default",
            end_user=ctx.end_user,
            bindings=ToolBindings.for_key(ctx.virtual_key),
        )

    def binding_value(self, source: str) -> str | None:
        """The identity value a binding source resolves to, if the caller has one."""
        values = {"end_user": self.end_user, "key_id": self.key_id, "team_id": self.team_id}
        return values.get(source)
