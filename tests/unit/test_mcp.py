from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.core.errors import GatewayError
from app.core.schemas import Role
from app.mcp.client import McpClient
from app.mcp.registry import McpRegistry, _ServerRecord
from app.mcp.translate import mcp_tool_to_tooldef, sanitize_schema, tool_result_to_message


def _transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_initialize_handshake_and_session_header() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        if payload["method"] == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"protocolVersion": "2025-03-26", "capabilities": {}},
                },
                headers={"Mcp-Session-Id": "session-42"},
            )
        assert request.headers["mcp-session-id"] == "session-42"
        return httpx.Response(202)

    async with McpClient("https://mcp.example", transport=_transport(handler)) as client:
        result = await client.initialize()
        assert result["protocolVersion"] == "2025-03-26"
        assert client.session_id == "session-42"
        assert seen[0]["method"] == "initialize"
        assert seen[0]["params"]["protocolVersion"] == "2025-03-26"
        assert seen[0]["params"]["capabilities"] == {}
        assert seen[0]["params"]["clientInfo"]["name"] == "aigateway"
        assert seen[1]["method"] == "notifications/initialized"


@pytest.mark.asyncio
async def test_tools_list_paginates_until_cursor_exhausted() -> None:
    cursors: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        cursor = payload["params"].get("cursor")
        cursors.append(cursor)
        result = {"tools": [{"name": "one"}], "nextCursor": "next"}
        if cursor:
            result = {"tools": [{"name": "two"}]}
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
        )

    async with McpClient("https://mcp.example", transport=_transport(handler)) as client:
        assert [tool["name"] for tool in await client.list_tools()] == ["one", "two"]
    assert cursors == [None, "next"]


@pytest.mark.asyncio
async def test_jsonrpc_error_is_gateway_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": payload["id"], "error": {"code": -1, "message": "bad"}},
        )

    async with McpClient("https://mcp.example", transport=_transport(handler)) as client:
        with pytest.raises(GatewayError, match="bad"):
            await client.list_resources()


def test_namespaces_tool_names_and_resolves_original_name() -> None:
    tool = {"name": "lookup", "inputSchema": {"type": "object"}}
    canonical = mcp_tool_to_tooldef("inventory", tool)
    registry = McpRegistry(db=None, settings=None)
    registry._records["server-1"] = _ServerRecord(
        id="server-1",
        name="Inventory server",
        transport="http",
        url="https://mcp.example",
        command=None,
        args=[],
        env={},
        headers={},
        is_active=True,
        tool_prefix="inventory",
        health_status="healthy",
        discovered_tools=[tool],
    )
    assert canonical.function.name == "inventory__lookup"
    assert registry.resolve(canonical.function.name) == ("server-1", "lookup")


def test_schema_sanitizing_removes_gemini_unsupported_fields_recursively() -> None:
    result = sanitize_schema(
        {
            "$schema": "draft",
            "$ref": "#/definitions/value",
            "definitions": {"ignored": {}},
            "properties": {
                "value": {
                    "type": "string",
                    "additionalProperties": False,
                    "$defs": {"nested": {}},
                }
            },
        }
    )
    assert result == {"properties": {"value": {"type": "string"}}}


def test_mcp_result_content_flattens_to_linked_tool_message() -> None:
    message = tool_result_to_message(
        "call-1",
        {
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "mimeType": "image/png", "data": "YWJj"},
                {"type": "resource", "resource": {"uri": "file://x", "text": "resource text"}},
            ],
            "isError": True,
        },
    )
    assert message.role == Role.TOOL
    assert message.tool_call_id == "call-1"
    assert "hello" in message.text()
    assert "Image content" in message.text()
    assert "resource text" in message.text()
    assert "Tool error" in message.text()


@pytest.mark.asyncio
async def test_tool_result_is_error_is_returned_not_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"content": [{"type": "text", "text": "failure detail"}], "isError": True},
            },
        )

    async with McpClient("https://mcp.example", transport=_transport(handler)) as client:
        result = await client.call_tool("fail", {})
    message = tool_result_to_message("call-2", result)
    assert message.tool_call_id == "call-2"
    assert "failure detail" in message.text()
    assert "Tool error" in message.text()


@pytest.mark.asyncio
async def test_streamable_http_parses_sse_jsonrpc_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        event = json.dumps({"jsonrpc": "2.0", "id": payload["id"], "result": {"resources": []}})
        return httpx.Response(
            200,
            text=f"event: message\ndata: {event}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    async with McpClient("https://mcp.example", transport=_transport(handler)) as client:
        assert await client.list_resources() == []


@pytest.mark.asyncio
async def test_stdio_client_launches_command_and_speaks_jsonrpc() -> None:
    import sys

    script = (
        "import json,sys\n"
        "for line in sys.stdin:\n"
        " request=json.loads(line)\n"
        " if 'id' in request:\n"
        "  method=request['method']\n"
        "  result={'protocolVersion':'2025-03-26','capabilities':{}} if method=='initialize' "
        "else {'tools':[{'name':'stdio_tool'}]}\n"
        "  print(json.dumps({'jsonrpc':'2.0','id':request['id'],'result':result}),flush=True)\n"
    )
    client = McpClient(command=sys.executable, args=["-u", "-c", script], env={"MCP_TEST": "1"})
    try:
        handshake = await client.initialize()
        assert handshake["protocolVersion"] == "2025-03-26"
        assert [tool["name"] for tool in await client.list_tools()] == ["stdio_tool"]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_stdio_start_failure_has_connection_diagnostic() -> None:
    client = McpClient(command="/definitely/not/a/real/mcp-server")
    with pytest.raises(GatewayError) as caught:
        await client.initialize()
    assert caught.value.details["failure_kind"] == "connection"
    await client.aclose()


class _ApiRegistry:
    def __init__(self, server: Any | None = None) -> None:
        from app.db.models import McpServer

        self.server = server or McpServer(
            id="server-1",
            name="local",
            transport="http",
            url="https://mcp.example",
            command=None,
            args=[],
            env={},
            headers={},
            is_active=True,
            health_status="healthy",
            discovered_tools=[],
            tool_prefix=None,
            metadata_={},
        )
        self.added: dict[str, Any] | None = None
        self.updated: dict[str, Any] | None = None

    async def add_server(self, **values: Any) -> Any:
        from app.db.models import McpServer

        self.added = values
        values["id"] = "server-1"
        values["health_status"] = "unknown"
        values["is_active"] = True
        values["discovered_tools"] = []
        values["last_health_check_at"] = None
        values["metadata_"] = values.pop("metadata", {})
        self.server = McpServer(**values)
        return self.server

    async def get_server(self, server_id: str) -> Any:
        assert server_id == self.server.id
        return self.server

    async def update_server(self, server_id: str, **values: Any) -> Any:
        self.updated = values
        for key, value in values.items():
            setattr(self.server, "metadata_" if key == "metadata" else key, value)
        return self.server

    async def list_servers(self) -> list[Any]:
        return [self.server]

    async def refresh(self, server_id: str) -> dict[str, bool]:
        return {server_id: False}

    async def tools_for(self, server_ids: list[str]) -> list[Any]:
        return []

    async def diagnostics(self, server_id: str) -> dict[str, str] | None:
        return {"message": "Connection refused", "failure_kind": "connection"}


async def _api_request(
    monkeypatch: pytest.MonkeyPatch,
    registry: _ApiRegistry,
    method: str,
    path: str,
    **kwargs: Any,
) -> httpx.Response:
    from fastapi import FastAPI

    import app.api.mcp as api

    app = FastAPI()
    app.include_router(api.router)
    monkeypatch.setattr(api, "_registry", lambda request: registry)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.request(method, path, **kwargs)


@pytest.mark.asyncio
async def test_api_accepts_stdio_config_and_redacts_environment_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _ApiRegistry()
    response = await _api_request(
        monkeypatch,
        registry,
        "POST",
        "/v1/mcp/servers",
        json={
            "name": "local",
            "transport": "stdio",
            "command": "python",
            "args": ["server.py"],
            "env": {"API_TOKEN": "secret-value", "MODE": "production"},
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["command"] == "python"
    assert body["args"] == ["server.py"]
    assert body["env"] == {"API_TOKEN": "***", "MODE": "***"}
    assert "secret-value" not in response.text
    assert registry.added is not None
    assert registry.added["transport"] == "stdio"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"name": "bad-http", "transport": "http", "command": "python"},
        {
            "name": "bad-stdio",
            "transport": "stdio",
            "command": "python",
            "url": "https://mcp.example",
        },
    ],
)
async def test_api_rejects_mismatched_transport_configuration(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    response = await _api_request(
        monkeypatch, _ApiRegistry(), "POST", "/v1/mcp/servers", json=payload
    )
    assert response.status_code == 422
    assert "transport" in response.text.lower()


@pytest.mark.asyncio
async def test_patch_server_endpoint_and_refresh_failure_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _ApiRegistry()
    updated = await _api_request(
        monkeypatch,
        registry,
        "PATCH",
        "/v1/mcp/servers/server-1",
        json={"url": "https://new-mcp.example"},
    )
    assert updated.status_code == 200
    assert updated.json()["url"] == "https://new-mcp.example"
    assert registry.updated == {"url": "https://new-mcp.example"}

    refreshed = await _api_request(
        monkeypatch, registry, "POST", "/v1/mcp/servers/server-1/refresh"
    )
    assert refreshed.status_code == 200
    assert refreshed.json()["healthy"] is False
    assert refreshed.json()["error"] == "Connection refused"
    assert refreshed.json()["failure_kind"] == "connection"


@pytest.mark.asyncio
async def test_registry_update_reconnects_and_rediscovers_for_endpoint_change() -> None:
    from unittest.mock import AsyncMock

    from app.db.models import McpServer

    row = McpServer(
        id="server-update",
        name="old",
        transport="http",
        url="https://old.example",
        command=None,
        args=[],
        env={},
        headers={},
        is_active=True,
        health_status="healthy",
        discovered_tools=[],
        tool_prefix=None,
        metadata_={},
    )

    class _Session:
        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def get(self, model: Any, server_id: str) -> Any:
            assert server_id == row.id
            return row

        async def flush(self) -> None:
            return None

    class _Database:
        def session(self) -> _Session:
            return _Session()

    class _Client:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    registry = McpRegistry(_Database(), settings=None)
    client = _Client()
    registry._clients[row.id] = client  # type: ignore[assignment]
    registry._load_records = AsyncMock()  # type: ignore[method-assign]
    registry.refresh = AsyncMock(return_value={row.id: True})  # type: ignore[method-assign]
    updated = await registry.update_server(row.id, url="https://new.example")
    assert updated.url == "https://new.example"
    assert client.closed is True
    registry.refresh.assert_awaited_once_with(row.id)
