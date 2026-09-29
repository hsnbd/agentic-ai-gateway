from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, model_validator

from app.api.deps import require_gateway_admin, require_gateway_writer
from app.core.errors import GatewayError, InvalidRequestError
from app.core.schemas import Message, ToolCall, ToolDef
from app.db.models import McpServer
from app.mcp.registry import McpRegistry
from app.tools.executor import ToolExecutor

router = APIRouter(tags=["mcp"])

#: Registering or changing MCP servers can launch host processes (stdio
#: transport), so it is reserved for operators. Calling tools is data-plane.
_ADMIN = [Depends(require_gateway_admin)]
_WRITE = [Depends(require_gateway_writer)]


class McpServerCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    transport: Literal["stdio", "http", "streamable-http"] = "http"
    url: str | None = Field(default=None, min_length=1, max_length=1024)
    command: str | None = Field(default=None, min_length=1, max_length=1024)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    description: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    tool_prefix: str | None = Field(default=None, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_transport_configuration(self) -> McpServerCreate:
        _validate_transport_fields(self.transport, self.url, self.command)
        return self


class McpServerUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    transport: Literal["stdio", "http", "streamable-http"] | None = None
    url: str | None = Field(default=None, min_length=1, max_length=1024)
    command: str | None = Field(default=None, min_length=1, max_length=1024)
    args: list[str] | None = None
    env: dict[str, str] | None = None
    description: str | None = None
    headers: dict[str, str] | None = None
    tool_prefix: str | None = Field(default=None, max_length=64)
    metadata: dict[str, Any] | None = None
    is_active: bool | None = None


class McpServerResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    transport: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    is_active: bool
    health_status: str
    last_health_check_at: datetime | None = None
    discovered_tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_prefix: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_server(cls, server: McpServer) -> McpServerResponse:
        return cls(
            id=server.id,
            name=server.name,
            description=server.description,
            transport=server.transport,
            command=server.command,
            args=list(server.args or []),
            env=dict.fromkeys(server.env or {}, "***"),
            url=server.url,
            headers=dict(server.headers or {}),
            is_active=server.is_active,
            health_status=server.health_status,
            last_health_check_at=server.last_health_check_at,
            discovered_tools=list(server.discovered_tools or []),
            tool_prefix=server.tool_prefix,
            metadata=dict(server.metadata_ or {}),
        )


class McpRefreshResponse(BaseModel):
    server_id: str
    healthy: bool
    tools: list[ToolDef] = Field(default_factory=list)
    error: str | None = None
    failure_kind: str | None = None


class McpToolCallRequest(BaseModel):
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class McpToolCallResponse(BaseModel):
    message: Message


def _validate_transport_fields(transport: str, url: str | None, command: str | None) -> None:
    if transport == "stdio":
        if not command:
            raise ValueError("stdio transport requires command")
        if url is not None:
            raise ValueError("stdio transport must not include url")
    else:
        if not url:
            raise ValueError("HTTP transport requires url")
        if command is not None:
            raise ValueError("HTTP transport must not include command")


def _registry(request: Request) -> McpRegistry:
    # Created by GatewayState.startup, shared with the agent executor.
    registry: McpRegistry = request.app.state.gateway.components["mcp_registry"]
    return registry


def _raise_http(error: GatewayError) -> NoReturn:
    raise HTTPException(status_code=error.status_code, detail=error.to_dict()) from error


def _validated_merged_config(server: McpServer, updates: dict[str, Any]) -> None:
    transport = updates.get("transport", server.transport)
    url = updates.get("url", server.url)
    command = updates.get("command", server.command)
    _validate_transport_fields(transport, url, command)


@router.get("/v1/mcp/servers", response_model=list[McpServerResponse])
async def list_servers(request: Request) -> list[McpServerResponse]:
    try:
        rows = await _registry(request).list_servers()
    except GatewayError as exc:
        _raise_http(exc)
    return [McpServerResponse.from_server(row) for row in rows]


@router.post(
    "/v1/mcp/servers", dependencies=_ADMIN, response_model=McpServerResponse, status_code=201
)
async def create_server(payload: McpServerCreate, request: Request) -> McpServerResponse:
    registry = _registry(request)
    try:
        row = await registry.add_server(
            name=payload.name,
            transport=payload.transport,
            url=payload.url,
            command=payload.command,
            args=payload.args,
            env=payload.env,
            headers=payload.headers,
            description=payload.description,
            tool_prefix=payload.tool_prefix,
            metadata=payload.metadata,
        )
    except GatewayError as exc:
        _raise_http(exc)
    return McpServerResponse.from_server(row)


@router.get("/v1/mcp/servers/{server_id}", response_model=McpServerResponse)
async def get_server(server_id: str, request: Request) -> McpServerResponse:
    try:
        row = await _registry(request).get_server(server_id)
    except GatewayError as exc:
        _raise_http(exc)
    return McpServerResponse.from_server(row)


@router.patch("/v1/mcp/servers/{server_id}", dependencies=_ADMIN, response_model=McpServerResponse)
async def update_server(
    server_id: str, payload: McpServerUpdate, request: Request
) -> McpServerResponse:
    registry = _registry(request)
    try:
        existing = await registry.get_server(server_id)
        updates = payload.model_dump(exclude_unset=True)
        _validated_merged_config(existing, updates)
        row = await registry.update_server(server_id, **updates)
    except ValueError as exc:
        _raise_http(InvalidRequestError(str(exc)))
    except GatewayError as exc:
        _raise_http(exc)
    return McpServerResponse.from_server(row)


@router.delete("/v1/mcp/servers/{server_id}", dependencies=_ADMIN, status_code=204)
async def delete_server(server_id: str, request: Request) -> Response:
    try:
        await _registry(request).remove_server(server_id)
    except GatewayError as exc:
        _raise_http(exc)
    return Response(status_code=204)


@router.post(
    "/v1/mcp/servers/{server_id}/refresh", dependencies=_ADMIN, response_model=McpRefreshResponse
)
async def refresh_server(server_id: str, request: Request) -> McpRefreshResponse:
    try:
        registry = _registry(request)
        health = await registry.refresh(server_id)
        healthy = health.get(server_id, False)
        tools = await registry.tools_for([server_id]) if healthy else []
        diagnostic = await registry.diagnostics(server_id)
    except GatewayError as exc:
        _raise_http(exc)
    return McpRefreshResponse(
        server_id=server_id,
        healthy=healthy,
        tools=tools,
        error=diagnostic.get("message") if diagnostic else None,
        failure_kind=diagnostic.get("failure_kind") if diagnostic else None,
    )


@router.get("/v1/mcp/tools", response_model=list[ToolDef])
async def list_tools(
    request: Request,
    server_id: str | None = Query(default=None),
) -> list[ToolDef]:
    try:
        return await _registry(request).tools_for([server_id] if server_id else None)
    except GatewayError as exc:
        _raise_http(exc)


@router.post("/v1/mcp/tools/call", dependencies=_WRITE, response_model=McpToolCallResponse)
async def call_tool(payload: McpToolCallRequest, request: Request) -> McpToolCallResponse:
    call = ToolCall(name=payload.name, arguments=json.dumps(payload.arguments))
    message = await ToolExecutor(_registry(request)).execute(call)
    return McpToolCallResponse(message=message)


@router.get("/v1/mcp/health", response_model=dict[str, bool])
async def mcp_health(request: Request) -> dict[str, bool]:
    return await _registry(request).health()
