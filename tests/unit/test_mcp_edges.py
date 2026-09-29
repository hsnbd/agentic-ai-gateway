"""Edge cases for the MCP client, tool translation, and the server registry."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import pytest

from app.config.settings import Settings
from app.core.errors import ErrorCode, GatewayError, NotFoundError
from app.db.session import Database
from app.mcp import registry as registry_module
from app.mcp.client import McpClient, _failure_kind, _parse_sse, _timeout_seconds
from app.mcp.registry import McpRegistry, _describe_failure
from app.mcp.translate import mcp_tool_to_tooldef, sanitize_schema, tool_result_to_message

# -- Translation ------------------------------------------------------------


def test_tool_translation_defaults() -> None:
    with pytest.raises(ValueError, match="valid name"):
        mcp_tool_to_tooldef("srv", {"name": ""})
    tool = mcp_tool_to_tooldef("srv", {"name": "t", "inputSchema": "nope", "description": 5})
    assert tool.function.name == "srv__t"
    assert tool.function.description == ""
    assert tool.function.parameters == {"type": "object", "properties": {}}


def test_sanitize_schema_cleans_lists() -> None:
    schema = {"anyOf": [{"type": "string", "$schema": "x"}], "additionalProperties": False}
    assert sanitize_schema(schema) == {"anyOf": [{"type": "string"}]}


def test_tool_results_flatten_every_block_kind() -> None:
    message = tool_result_to_message(
        "call_1",
        {
            "content": [
                "not-a-block",
                {"type": "text", "text": 5},
                {"type": "image", "mimeType": "image/png", "data": "AAAA"},
                {"type": "image"},
                {"type": "resource", "resource": {"text": "inline"}},
                {"type": "resource", "resource": {"uri": "file://a", "blob": "QUJD"}},
                {"type": "resource", "resource": "nope"},
                {"type": "audio"},
            ]
        },
    )
    lines = message.text().split("\n")
    assert lines == [
        "[Image content (image/png, 4 encoded characters) omitted]",
        "[Image content (unknown MIME type, 0 encoded characters) omitted]",
        "inline",
        "[Resource file://a (unknown MIME type; 4 encoded characters; non-text data)]",
    ]
    assert tool_result_to_message("c", {"content": "nope"}).text() == ""


# -- Client helpers ---------------------------------------------------------


def test_timeout_resolution() -> None:
    assert _timeout_seconds(5, 30.0) == 5.0
    assert _timeout_seconds(httpx.Timeout(3.0), 30.0) == 3.0
    assert _timeout_seconds(httpx.Timeout(None, connect=4.0), 30.0) == 4.0
    assert _timeout_seconds(httpx.Timeout(None), 30.0) == 30.0
    assert _timeout_seconds(None, httpx.Timeout(7.0)) == 7.0
    assert _timeout_seconds(None, httpx.Timeout(None)) == 30.0
    assert _timeout_seconds(None, 9) == 9.0


def test_failure_kind_classification() -> None:
    assert _failure_kind("Connection refused") == "connection"
    assert _failure_kind("read timed out") == "timeout"
    assert _failure_kind("something else") == "connection"


def test_parse_sse_ignores_non_data_lines_and_non_objects() -> None:
    body = 'event: message\ndata: [1]\n\n: comment\ndata: {"id": 1}\n'
    assert _parse_sse(body) == [{"id": 1}]


# -- HTTP client ------------------------------------------------------------


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> McpClient:
    return McpClient("https://mcp.example/mcp", transport=httpx.MockTransport(handler))


def _reply(request: httpx.Request, **fields: Any) -> httpx.Response:
    payload = json.loads(request.content)
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], **fields})


async def test_http_client_rejects_invalid_results() -> None:
    replies = {
        "tools/list": {"result": {"tools": "nope"}},
        "tools/call": {"result": {"content": "nope"}},
        "resources/list": {"result": {"resources": "nope"}},
        "resources/read": {"result": [1]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return _reply(request, **replies[json.loads(request.content)["method"]])

    async with _client(handler) as client:
        with pytest.raises(GatewayError, match="invalid tools"):
            await client.list_tools()
        with pytest.raises(GatewayError, match="invalid content"):
            await client.call_tool("t", {})
        with pytest.raises(GatewayError, match="invalid resources"):
            await client.list_resources()
        with pytest.raises(GatewayError, match="non-object result"):
            await client.read_resource("file://a")


async def test_http_client_resources_and_request_timeouts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        method = json.loads(request.content)["method"]
        if method == "resources/list":
            return _reply(request, result={"resources": [{"uri": "a"}, "bad"]})
        return _reply(request, result={"contents": []})

    async with _client(handler) as client:
        assert await client.list_resources(request_timeout=1.0) == [{"uri": "a"}]
        assert await client.read_resource("a") == {"contents": []}


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (httpx.ReadTimeout("slow"), "provider_timeout"),
        (httpx.ConnectError("refused"), "provider_unavailable"),
        (httpx.DecodingError("bad gzip"), "provider_unavailable"),
        (
            httpx.HTTPStatusError(
                "bad", request=httpx.Request("POST", "https://x"), response=httpx.Response(500)
            ),
            "provider_error",
        ),
    ],
)
async def test_http_transport_errors_are_mapped(error: Exception, code: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    async with _client(handler) as client:
        with pytest.raises(GatewayError) as raised:
            await client.list_tools()
    assert raised.value.code.value == code


@pytest.mark.parametrize(
    ("response", "message", "kind"),
    [
        (httpx.Response(401), "HTTP 401", "authentication"),
        (httpx.Response(500), "HTTP 500", "protocol_error"),
        (httpx.Response(200), "empty response", None),
        (httpx.Response(200, content=b"{not json"), "invalid JSON-RPC", "protocol_error"),
        (httpx.Response(200, json=[1]), "did not match request id", None),
        (httpx.Response(200, json={"id": 999}), "did not match request id", None),
    ],
)
async def test_http_response_failures(
    response: httpx.Response, message: str, kind: str | None
) -> None:
    async with _client(lambda request: response) as client:
        with pytest.raises(GatewayError, match=message) as raised:
            await client.list_tools()
    assert raised.value.details.get("failure_kind") == kind


async def test_notifications_tolerate_bodies() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if "id" not in payload:
            return httpx.Response(200, json={"jsonrpc": "2.0"})
        return _reply(request, result={"protocolVersion": "x"})

    async with _client(handler) as client:
        assert await client.initialize() == {"protocolVersion": "x"}


# -- stdio client -----------------------------------------------------------


def _stdio(script: str) -> McpClient:
    return McpClient(command=sys.executable, args=["-u", "-c", script], timeout=5.0)


async def test_stdio_skips_unrelated_lines_and_reuses_process() -> None:
    script = (
        "import json,sys\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line)\n"
        " if 'id' in r:\n"
        "  print(json.dumps({'jsonrpc':'2.0','method':'log'}),flush=True)\n"
        "  print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':{'tools':[]}}),flush=True)\n"
    )
    client = _stdio(script)
    try:
        assert await client.list_tools() == []
        process = client._process
        assert await client.list_tools(request_timeout=httpx.Timeout(5.0)) == []
        assert client._process is process
    finally:
        await client.aclose()


async def test_stdio_invalid_json_and_exit_are_reported() -> None:
    garbage = _stdio("import sys\nsys.stdin.readline()\nprint('not json', flush=True)\n")
    try:
        with pytest.raises(GatewayError, match="invalid JSON-RPC") as raised:
            await garbage.list_tools()
        assert raised.value.details["failure_kind"] == "protocol_error"
    finally:
        await garbage.aclose()

    exits = _stdio("import sys\nsys.stdin.readline()\n")
    try:
        with pytest.raises(GatewayError, match="exited") as raised:
            await exits.list_tools()
        assert raised.value.details["failure_kind"] == "connection"
    finally:
        await exits.aclose()


async def test_stdio_timeout() -> None:
    client = _stdio("import sys,time\nsys.stdin.readline()\ntime.sleep(30)\n")
    try:
        with pytest.raises(GatewayError, match="timed out") as raised:
            await client.list_tools(request_timeout=0.2)
        assert raised.value.details["failure_kind"] == "timeout"
    finally:
        await client.aclose()


async def test_stdio_close_kills_processes_that_ignore_terminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[bool] = []

    class Stubborn:
        returncode = None

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            killed.append(True)
            self.returncode = -9

        async def wait(self) -> int:
            if not killed:
                await asyncio.sleep(10)
            return -9

    real_timeout = asyncio.timeout
    monkeypatch.setattr("app.mcp.client.asyncio.timeout", lambda _: real_timeout(0.05))
    client = McpClient(command="unused")
    client._process = Stubborn()  # type: ignore[assignment]
    await client.close()
    assert killed == [True]


class _Pipe:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def write(self, data: bytes) -> None:
        pass

    async def drain(self) -> None:
        if self.error is not None:
            raise self.error


async def test_stdio_without_pipes_or_with_broken_pipe() -> None:
    client = McpClient(command="unused")
    client._process = SimpleNamespace(returncode=None, stdin=None, stdout=None)  # type: ignore[assignment]
    with pytest.raises(GatewayError, match="no usable stdin"):
        await client.list_tools()

    client._process = SimpleNamespace(  # type: ignore[assignment]
        returncode=None, stdin=_Pipe(BrokenPipeError("gone")), stdout=object()
    )
    with pytest.raises(GatewayError, match="transport failed") as raised:
        await client.list_tools()
    assert raised.value.details["failure_kind"] == "connection"
    client._process = None
    await client.aclose()


async def test_stdio_notifications_do_not_wait_for_replies() -> None:
    client = McpClient(command="unused")
    client._process = SimpleNamespace(returncode=None, stdin=_Pipe(), stdout=object())  # type: ignore[assignment]
    await client._notify("notifications/initialized")
    client._process = None
    await client.aclose()


async def test_ensure_process_requires_a_command() -> None:
    client = McpClient()
    with pytest.raises(GatewayError, match="requires a command"):
        await client._ensure_process()
    await client.aclose()


# -- Registry ---------------------------------------------------------------


class FakeClient:
    """Stands in for McpClient inside the registry; behaviour is set per URL."""

    behaviours: ClassVar[dict[str, Any]] = {}
    instances: ClassVar[list[FakeClient]] = []

    def __init__(self, url: str | None, headers: Any, **kwargs: Any) -> None:
        self.url = url or kwargs.get("command") or ""
        if FakeClient.behaviours.get(self.url) == "unconstructable":
            raise ValueError("bad client configuration")
        self.closed = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        FakeClient.instances.append(self)

    async def initialize(self) -> dict[str, Any]:
        behaviour = FakeClient.behaviours.get(self.url)
        if isinstance(behaviour, Exception):
            raise behaviour
        return {}

    async def list_tools(self) -> list[dict[str, Any]]:
        behaviour = FakeClient.behaviours.get(self.url, [])
        if isinstance(behaviour, Exception):
            raise behaviour
        return list(behaviour)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"content": [{"type": "text", "text": "ok"}]}

    async def aclose(self) -> None:
        self.closed = True
        if FakeClient.behaviours.get("close") == "explode":
            raise RuntimeError("close failed")


@pytest.fixture
async def registry(monkeypatch: pytest.MonkeyPatch) -> Any:
    FakeClient.behaviours = {}
    FakeClient.instances = []
    monkeypatch.setattr(registry_module, "McpClient", FakeClient)
    database = Database(Settings(database_url="sqlite+aiosqlite:///:memory:"))
    await database.startup()
    await database.create_all()
    yield McpRegistry(database, Settings(mcp_tool_cache_ttl_seconds=300))
    await database.shutdown()


async def test_add_discover_resolve_and_call(registry: McpRegistry) -> None:
    FakeClient.behaviours["http://a"] = [{"name": "search"}]
    server = await registry.add_server(name="alpha", url="http://a", tool_prefix="a")
    assert server.health_status == "healthy"
    tools = await registry.tools_for()
    assert [tool.function.name for tool in tools] == ["a__search"]
    assert registry.resolve("a__search") == (server.id, "search")
    with pytest.raises(NotFoundError):
        registry.resolve("a__missing")
    with pytest.raises(NotFoundError):
        registry.resolve("zzz__search")
    result = await registry.call_tool(server.id, "search", {"q": "x"})
    assert result["content"][0]["text"] == "ok"
    assert await registry.health() == {server.id: True}
    assert await registry.diagnostics(server.id) is None
    assert [row.name for row in await registry.list_servers()] == ["alpha"]


async def test_unhealthy_servers_are_diagnosed_and_excluded(registry: McpRegistry) -> None:
    FakeClient.behaviours["http://down"] = GatewayError(
        ErrorCode.PROVIDER_UNAVAILABLE,
        "refused",
        details={"failure_kind": "connection"},
    )
    down = await registry.add_server(name="down", url="http://down")
    stdio = await registry.add_server(name="stdio", transport="stdio")
    no_url = await registry.add_server(name="nourl", url="  ")
    assert await registry.tools_for() == []
    assert (await registry.diagnostics(down.id)) == {
        "message": "refused",
        "failure_kind": "connection",
    }
    assert (await registry.diagnostics(stdio.id))["failure_kind"] == "configuration"  # type: ignore[index]
    assert "missing a URL" in (await registry.diagnostics(no_url.id))["message"]  # type: ignore[index]
    with pytest.raises(NotFoundError, match="unavailable"):
        await registry.call_tool(down.id, "x", {})
    assert FakeClient.instances[0].closed


async def test_refresh_inactive_and_unknown_servers(registry: McpRegistry) -> None:
    server = await registry.add_server(name="alpha", url="http://a")
    await registry.update_server(server.id, is_active=False)
    assert await registry.refresh() == {server.id: False}
    assert registry._records[server.id].health_status == "inactive"
    with pytest.raises(NotFoundError):
        await registry.refresh("missing")
    with pytest.raises(NotFoundError):
        await registry.tools_for(["missing"])


async def test_tools_for_refreshes_expired_records(registry: McpRegistry) -> None:
    FakeClient.behaviours["http://a"] = [{"name": "one"}]
    server = await registry.add_server(name="alpha", url="http://a")
    FakeClient.behaviours["http://a"] = [{"name": "one"}, {"name": "two"}]
    registry._records[server.id].expires_at = 0
    assert len(await registry.tools_for([server.id])) == 2
    registry._records[server.id].expires_at = 0
    assert len(await registry.tools_for()) == 2


async def test_update_without_connection_change_does_not_reconnect(registry: McpRegistry) -> None:
    server = await registry.add_server(name="alpha", url="http://a")
    before = len(FakeClient.instances)
    updated = await registry.update_server(server.id, description="docs", metadata={"k": 1})
    assert updated.description == "docs"
    assert updated.metadata_ == {"k": 1}
    assert len(FakeClient.instances) == before
    with pytest.raises(NotFoundError):
        await registry.update_server("missing", description="x")


async def test_connection_change_without_existing_client(registry: McpRegistry) -> None:
    FakeClient.behaviours["http://a"] = RuntimeError("boom")
    server = await registry.add_server(name="alpha", url="http://a")
    assert server.id not in registry._clients
    FakeClient.behaviours["http://b"] = [{"name": "t"}]
    updated = await registry.update_server(server.id, url="http://b")
    assert updated.health_status == "healthy"


async def test_remove_server_and_stale_records(registry: McpRegistry) -> None:
    server = await registry.add_server(name="alpha", url="http://a")
    client = registry._clients[server.id]
    await registry.remove_server(server.id)
    assert client.closed  # type: ignore[attr-defined]
    assert server.id not in registry._records
    with pytest.raises(NotFoundError):
        await registry.remove_server(server.id)
    with pytest.raises(NotFoundError):
        await registry.get_server(server.id)

    # A server with no live client can still be removed.
    FakeClient.behaviours["http://b"] = RuntimeError("down")
    other = await registry.add_server(name="beta", url="http://b")
    await registry.remove_server(other.id)

    # Rows deleted behind the registry's back drop out on reload.
    third = await registry.add_server(name="gamma", url="http://c")
    async with registry.db.session() as session:
        await session.delete(await session.get(registry_module.McpServer, third.id))
    await registry._load_records(force=True)
    assert third.id not in registry._records


async def test_saving_discovery_for_a_deleted_row_is_a_no_op(registry: McpRegistry) -> None:
    server = await registry.add_server(name="alpha", url="http://a")
    record = registry._records[server.id]
    async with registry.db.session() as session:
        await session.delete(await session.get(registry_module.McpServer, server.id))
    await registry._save_discovery(record)


async def test_close_tolerates_client_errors(registry: McpRegistry) -> None:
    await registry.add_server(name="alpha", url="http://a")
    FakeClient.behaviours["close"] = "explode"
    await registry.close()
    assert registry._clients == {}


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("HTTP 401 unauthorized", "authentication"),
        ("request timed out", "timeout"),
        ("invalid JSON-RPC data", "protocol_error"),
        ("Connection refused", "connection"),
        ("", "unknown"),
    ],
)
def test_describe_failure_classifies_messages(message: str, kind: str) -> None:
    described = _describe_failure(RuntimeError(message))
    assert described["failure_kind"] == kind
    if not message:
        assert described["message"] == "RuntimeError"


def test_describe_failure_prefers_explicit_kind_and_tolerates_odd_details() -> None:
    error = RuntimeError("timeout")
    error.details = {"failure_kind": "authentication"}  # type: ignore[attr-defined]
    assert _describe_failure(error)["failure_kind"] == "authentication"
    odd = RuntimeError("refused")
    odd.details = "nope"  # type: ignore[attr-defined]
    assert _describe_failure(odd)["failure_kind"] == "connection"


async def test_client_construction_failure_marks_server_unhealthy(registry: McpRegistry) -> None:
    FakeClient.behaviours["http://bad"] = "unconstructable"
    server = await registry.add_server(name="bad", url="http://bad")
    assert server.health_status == "unhealthy"
    assert (await registry.diagnostics(server.id))["message"] == "bad client configuration"  # type: ignore[index]


# -- HTTP API error paths ---------------------------------------------------


class _FailingApiRegistry:
    """Every registry call fails the way an unknown server does."""

    def __getattr__(self, name: str) -> Any:
        async def fail(*args: Any, **kwargs: Any) -> Any:
            raise NotFoundError("MCP server 'x' was not found")

        return fail


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/v1/mcp/servers", None),
        ("POST", "/v1/mcp/servers", {"name": "n", "transport": "http", "url": "http://x"}),
        ("GET", "/v1/mcp/servers/x", None),
        ("PATCH", "/v1/mcp/servers/x", {"description": "d"}),
        ("DELETE", "/v1/mcp/servers/x", None),
        ("POST", "/v1/mcp/servers/x/refresh", None),
        ("GET", "/v1/mcp/tools?server_id=x", None),
    ],
)
async def test_api_maps_registry_errors_to_http(
    monkeypatch: pytest.MonkeyPatch, method: str, path: str, body: Any
) -> None:
    from tests.unit.test_mcp import _api_request

    kwargs = {"json": body} if body is not None else {}
    response = await _api_request(monkeypatch, _FailingApiRegistry(), method, path, **kwargs)  # type: ignore[arg-type]
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    "patch",
    [
        {"transport": "http", "url": "http://x", "command": "python"},
        {"transport": "stdio", "command": None},
    ],
)
async def test_api_rejects_invalid_merged_updates(
    monkeypatch: pytest.MonkeyPatch, patch: dict[str, Any]
) -> None:
    from tests.unit.test_mcp import _api_request, _ApiRegistry

    response = await _api_request(
        monkeypatch, _ApiRegistry(), "PATCH", "/v1/mcp/servers/server-1", json=patch
    )
    assert response.status_code == 400
    assert "transport" in response.text.lower()


def test_registry_is_read_from_gateway_state() -> None:
    from app.api.mcp import _registry

    registry = object()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(gateway=SimpleNamespace(components={"mcp_registry": registry}))
        )
    )
    assert _registry(request) is registry  # type: ignore[arg-type]
