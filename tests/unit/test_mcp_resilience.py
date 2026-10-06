"""MCP resilience: expired sessions, crashed stdio servers, per-server circuit
breakers, discovery retries, and per-server timeouts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from app.config.settings import Settings
from app.core.errors import ErrorCode, GatewayError
from app.db.session import Database
from app.mcp import registry as registry_module
from app.mcp.client import McpClient
from app.mcp.registry import McpRegistry

FAKE_MCP = str(Path(__file__).resolve().parents[2] / "scripts" / "fake_mcp_server.py")


# -- Expired HTTP sessions -----------------------------------------------------


def _session_server(expire_on: set[str]) -> tuple[httpx.MockTransport, list[str]]:
    """A streamable-HTTP MCP server that forgets the session once per method in ``expire_on``."""
    seen: list[str] = []
    sessions = iter(f"session-{n}" for n in range(1, 10))
    current: dict[str, str] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method = body["method"]
        seen.append(method)
        if method == "initialize":
            current["id"] = next(sessions)
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": {}},
                headers={"mcp-session-id": current["id"]},
            )
        if "id" not in body:
            return httpx.Response(202)
        if method in expire_on:
            expire_on.discard(method)
            return httpx.Response(404, json={"error": "unknown session"})
        assert request.headers.get("mcp-session-id") == current["id"]
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": body["id"], "result": {"content": [], "tools": []}},
        )

    return httpx.MockTransport(handle), seen


async def test_an_expired_session_is_reinitialised_and_the_call_replayed() -> None:
    transport, seen = _session_server({"tools/call"})
    client = McpClient("http://mcp.test/mcp", transport=transport)
    await client.initialize()
    assert client.session_id == "session-1"
    result = await client.call_tool("echo", {"text": "hi"})
    assert result == {"content": [], "tools": []}
    assert client.session_id == "session-2"
    assert seen == [
        "initialize",
        "notifications/initialized",
        "tools/call",
        "initialize",
        "notifications/initialized",
        "tools/call",
    ]
    await client.aclose()


async def test_a_404_without_a_session_is_an_ordinary_error() -> None:
    transport, _ = _session_server({"initialize"})

    def no_session(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "gone"})

    client = McpClient("http://mcp.test/mcp", transport=httpx.MockTransport(no_session))
    with pytest.raises(GatewayError, match="HTTP 404"):
        await client.list_tools()
    await client.aclose()
    del transport


async def test_a_second_404_after_reinitialising_is_raised() -> None:
    def always_expired(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "initialize":
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": {}},
                headers={"mcp-session-id": "s"},
            )
        if "id" not in body:
            return httpx.Response(202)
        return httpx.Response(404, json={"error": "unknown session"})

    client = McpClient("http://mcp.test/mcp", transport=httpx.MockTransport(always_expired))
    await client.initialize()
    with pytest.raises(GatewayError, match="HTTP 404"):
        await client.call_tool("echo", {})
    await client.aclose()


# -- stdio crashes and stderr --------------------------------------------------


async def test_a_crashed_stdio_server_is_restarted_with_a_fresh_handshake() -> None:
    client = McpClient(command=sys.executable, args=[FAKE_MCP, "--stdio"])
    await client.initialize()
    first = client._process
    assert first is not None
    first.kill()
    await first.wait()

    result = await client.call_tool("echo", {"text": "after crash"})
    assert "after crash" in json.dumps(result)
    assert client._process is not first
    await client.aclose()


async def test_stdio_stderr_is_kept_for_diagnostics() -> None:
    script = "import sys; sys.stderr.write('starting\\nconfig missing: API_TOKEN\\n'); sys.exit(3)"
    client = McpClient(command=sys.executable, args=["-c", script], timeout=5)
    with pytest.raises(GatewayError, match="exited"):
        await client.initialize()
    assert client._stderr_task is not None
    await client._stderr_task
    assert list(client.stderr_tail) == ["starting", "config missing: API_TOKEN"]
    await client.aclose()


# -- Registry: breaker, discovery retry, timeouts --------------------------------


class _Client:
    """Programmable stand-in for McpClient inside the registry."""

    init_failures: ClassVar[list[Exception]] = []
    call_results: ClassVar[list[Any]] = []
    instances: ClassVar[list[_Client]] = []

    def __init__(self, url: str | None, headers: Any, **kwargs: Any) -> None:
        self.timeout = kwargs["timeout"]
        self.stderr_tail = ["line"]
        _Client.instances.append(self)

    async def initialize(self) -> dict[str, Any]:
        if _Client.init_failures:
            raise _Client.init_failures.pop(0)
        return {}

    async def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": "t"}]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        outcome = _Client.call_results.pop(0) if _Client.call_results else {"content": []}
        if isinstance(outcome, Exception):
            raise outcome
        result: dict[str, Any] = outcome
        return result

    async def aclose(self) -> None:
        return None


@pytest.fixture
async def registry(monkeypatch: pytest.MonkeyPatch) -> Any:
    _Client.init_failures, _Client.call_results, _Client.instances = [], [], []
    monkeypatch.setattr(registry_module, "McpClient", _Client)
    monkeypatch.setattr(registry_module, "_DISCOVERY_RETRY_DELAY", 0)
    database = Database(Settings(database_url="sqlite+aiosqlite:///:memory:"))
    await database.startup()
    await database.create_all()
    settings = Settings(
        circuit_breaker_threshold=2, circuit_breaker_cooldown_seconds=60, mcp_timeout_seconds=7
    )
    yield McpRegistry(database, settings)
    await database.shutdown()


def _outage() -> GatewayError:
    return GatewayError(ErrorCode.PROVIDER_UNAVAILABLE, "refused")


async def test_the_breaker_opens_on_outages_and_fails_fast(registry: McpRegistry) -> None:
    server = await registry.add_server(name="flaky", url="http://flaky")
    _Client.call_results = [_outage(), _outage()]
    for _ in range(2):
        with pytest.raises(GatewayError, match="refused"):
            await registry.call_tool(server.id, "t", {})
    assert registry.runtime(server.id) == {"breaker": "open", "stderr": ["line"]}
    with pytest.raises(GatewayError, match="calls are paused") as paused:
        await registry.call_tool(server.id, "t", {})
    assert paused.value.details["failure_kind"] == "circuit_open"


async def test_tool_errors_do_not_trip_the_breaker(registry: McpRegistry) -> None:
    server = await registry.add_server(name="strict", url="http://strict")
    _Client.call_results = [
        GatewayError(ErrorCode.PROVIDER_ERROR, "invalid params"),
        GatewayError(ErrorCode.PROVIDER_ERROR, "invalid params"),
        {"content": [{"type": "text", "text": "fine"}]},
    ]
    for _ in range(2):
        with pytest.raises(GatewayError, match="invalid params"):
            await registry.call_tool(server.id, "t", {})
    assert (await registry.call_tool(server.id, "t", {}))["content"][0]["text"] == "fine"
    assert registry.runtime(server.id)["breaker"] == "closed"


async def test_reconfiguring_or_removing_a_server_resets_its_breaker(registry: McpRegistry) -> None:
    server = await registry.add_server(name="reset", url="http://reset")
    _Client.call_results = [_outage(), _outage()]
    for _ in range(2):
        with pytest.raises(GatewayError):
            await registry.call_tool(server.id, "t", {})
    await registry.update_server(server.id, url="http://reset-2")
    assert registry.runtime(server.id)["breaker"] == "closed"
    await registry.remove_server(server.id)
    assert registry.runtime(server.id) == {"breaker": "closed", "stderr": []}


async def test_discovery_retries_one_outage_but_not_other_errors(registry: McpRegistry) -> None:
    _Client.init_failures = [_outage()]
    healthy = await registry.add_server(name="blip", url="http://blip")
    assert healthy.health_status == "healthy"
    assert len(_Client.instances) == 2

    _Client.init_failures = [GatewayError(ErrorCode.PROVIDER_ERROR, "bad handshake")]
    broken = await registry.add_server(name="bad", url="http://bad")
    assert broken.health_status == "unhealthy"

    _Client.init_failures = [_outage(), _outage()]
    down = await registry.add_server(name="down", url="http://down")
    assert down.health_status == "unhealthy"


async def test_a_per_server_timeout_overrides_the_default(registry: McpRegistry) -> None:
    await registry.add_server(name="default", url="http://d")
    await registry.add_server(name="slow", url="http://s", timeout_seconds=45)
    assert [client.timeout for client in _Client.instances] == [7.0, 45]


async def test_background_health_checks_refresh_and_stop(
    registry: McpRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    registry.start_health_checks(0)
    assert registry._health_task is None  # disabled

    calls: list[str] = []

    async def refresh(server_id: str | None = None) -> dict[str, bool]:
        calls.append("refresh")
        if len(calls) == 1:
            raise RuntimeError("database blip")  # logged, the loop keeps going
        return {}

    monkeypatch.setattr(registry, "refresh", refresh)
    registry.start_health_checks(0.01)
    task = registry._health_task
    registry.start_health_checks(0.01)  # idempotent
    assert registry._health_task is task
    for _ in range(100):
        if len(calls) >= 2:
            break
        await asyncio.sleep(0.01)
    assert len(calls) >= 2
    await registry.close()
    assert registry._health_task is None and task is not None and task.cancelled()
