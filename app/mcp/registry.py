from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from app.core.errors import ErrorCode, GatewayError, NotFoundError
from app.core.schemas import ToolDef
from app.core.secrets import SecretBox
from app.db.models import McpServer
from app.mcp.client import McpClient
from app.mcp.translate import mcp_tool_to_tooldef
from app.observability.metrics import MCP_BREAKER_OPEN
from app.routing.breaker import BreakerState, CircuitBreaker

logger = logging.getLogger(__name__)

#: Pause before retrying a failed discovery.
_DISCOVERY_RETRY_DELAY = 0.2


@dataclass
class _ServerRecord:
    id: str
    name: str
    transport: str
    url: str | None
    command: str | None
    args: list[str]
    env: dict[str, str]
    headers: dict[str, str]
    is_active: bool
    tool_prefix: str | None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    health_status: str = "unknown"
    discovered_tools: list[dict[str, Any]] = field(default_factory=list)
    expires_at: float = 0.0
    last_error: dict[str, str] | None = None
    timeout_seconds: float | None = None

    @property
    def alias(self) -> str:
        return self.tool_prefix or self.name


class McpRegistry:
    """Loads configured MCP servers and caches their discovered tools."""

    def __init__(self, db: Any, settings: Any) -> None:
        self.db = db
        self.settings = settings
        self._records: dict[str, _ServerRecord] = {}
        self._clients: dict[str, McpClient] = {}
        self._last_errors: dict[str, dict[str, str]] = {}
        self._refresh_lock = asyncio.Lock()
        self._cache_ttl = float(getattr(settings, "mcp_tool_cache_ttl_seconds", 300))
        self._default_timeout = float(getattr(settings, "mcp_timeout_seconds", 10.0))
        self._health_task: asyncio.Task[None] | None = None
        # Per-server breaker: a failing server fails fast instead of making every
        # agent loop wait for its timeout.
        self._breaker = CircuitBreaker(
            int(getattr(settings, "circuit_breaker_threshold", 5)),
            float(getattr(settings, "circuit_breaker_cooldown_seconds", 30.0)),
        )
        key = getattr(settings, "secrets_encryption_key", None)
        self._secrets = SecretBox(key.get_secret_value() if key is not None else None)

    def start_health_checks(self, interval: float) -> None:
        """Re-discover every server every ``interval`` seconds, so health and tool
        lists stay current without waiting for a request to notice a change."""
        if interval > 0 and self._health_task is None:
            self._health_task = asyncio.create_task(self._health_loop(interval))

    async def _health_loop(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                await self._load_records(force=True)
                await self.refresh()
            except Exception:
                logger.warning("Background MCP health check failed", exc_info=True)

    async def close(self) -> None:
        """Stop health checks and close every client, terminating stdio processes."""
        if self._health_task is not None:
            self._health_task.cancel()
            await asyncio.gather(self._health_task, return_exceptions=True)
            self._health_task = None
        clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            try:
                await client.aclose()
            except Exception:
                logger.warning("MCP client did not close cleanly", exc_info=True)

    async def refresh(self, server_id: str | None = None) -> dict[str, bool]:
        async with self._refresh_lock:
            await self._load_records(force=True)
            selected = list(self._records.values())
            if server_id is not None:
                record = self._records.get(server_id)
                if record is None:
                    raise NotFoundError(f"MCP server {server_id!r} was not found")
                selected = [record]
            pending: list[_ServerRecord] = []
            for record in selected:
                if not record.is_active:
                    record.health_status = "inactive"
                    record.discovered_tools = []
                    record.expires_at = time.monotonic() + self._cache_ttl
                    await self._save_discovery(record)
                else:
                    pending.append(record)
            await asyncio.gather(*(self._discover(record) for record in pending))
            return {record.id: record.health_status == "healthy" for record in selected}

    async def tools_for(self, server_ids: list[str] | None = None) -> list[ToolDef]:
        await self._load_records()
        if server_ids is None:
            selected_ids = [record.id for record in self._records.values() if record.is_active]
        else:
            selected_ids = server_ids
        for server_id in selected_ids:
            record = self._records.get(server_id)
            if record is None:
                raise NotFoundError(f"MCP server {server_id!r} was not found")
            if record.expires_at <= time.monotonic():
                if server_ids is None:
                    await self.refresh()
                else:
                    await self.refresh(server_id)
        result: list[ToolDef] = []
        for server_id in selected_ids:
            record = self._records[server_id]
            if record.health_status != "healthy":
                continue
            result.extend(
                mcp_tool_to_tooldef(record.alias, tool) for tool in record.discovered_tools
            )
        return result

    def resolve(self, namespaced_name: str) -> tuple[str, str]:
        for record in self._records.values():
            prefix = f"{record.alias}__"
            if namespaced_name.startswith(prefix):
                original_name = namespaced_name[len(prefix) :]
                if record.health_status == "healthy" and any(
                    tool.get("name") == original_name for tool in record.discovered_tools
                ):
                    return record.id, original_name
        raise NotFoundError(f"MCP tool {namespaced_name!r} was not found or is unavailable")

    async def call_tool(
        self, server_id: str, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        client = self._clients.get(server_id)
        if client is None:
            await self.refresh(server_id)
            client = self._clients.get(server_id)
        if client is None:
            raise NotFoundError(f"MCP server {server_id!r} is unavailable")
        if not self._breaker.is_available(server_id):
            raise GatewayError(
                ErrorCode.PROVIDER_UNAVAILABLE,
                f"MCP server {server_id!r} is failing; calls are paused until it recovers",
                details={"failure_kind": "circuit_open"},
            )
        started = time.monotonic()
        try:
            result = await client.call_tool(name, arguments)
        except GatewayError as exc:
            if _is_outage(exc):
                self._breaker.record_failure(server_id, exc.message)
            else:
                # The server answered (with an error), so it is up.
                self._breaker.record_success(server_id)
            raise
        else:
            self._breaker.record_success(server_id, (time.monotonic() - started) * 1000)
        finally:
            MCP_BREAKER_OPEN.labels(self.server_name(server_id) or server_id).set(
                1 if self._breaker.health(server_id).state is BreakerState.OPEN else 0
            )
        return result

    async def health(self) -> dict[str, bool]:
        await self._load_records()
        return {
            server_id: record.health_status == "healthy"
            for server_id, record in self._records.items()
        }

    async def add_server(
        self,
        *,
        name: str,
        transport: str = "http",
        url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        description: str | None = None,
        tool_prefix: str | None = None,
        metadata: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> McpServer:
        row = McpServer(
            timeout_seconds=timeout_seconds,
            name=name,
            description=description,
            transport=transport,
            url=url,
            command=command,
            args=args or [],
            env=self._secrets.encrypt_map(env),
            headers=self._secrets.encrypt_map(headers),
            is_active=True,
            tool_prefix=tool_prefix,
            metadata_=metadata or {},
        )
        async with self.db.session() as session:
            session.add(row)
            await session.flush()
        await self._load_records(force=True)
        await self.refresh(row.id)
        return await self.get_server(row.id)

    async def remove_server(self, server_id: str) -> None:
        async with self.db.session() as session:
            row = await session.get(McpServer, server_id)
            if not isinstance(row, McpServer):
                raise NotFoundError(f"MCP server {server_id!r} was not found")
            await session.delete(row)
        client = self._clients.pop(server_id, None)
        if client is not None:
            await client.aclose()
        self._records.pop(server_id, None)
        self._last_errors.pop(server_id, None)
        self._breaker.reset(server_id)

    async def update_server(self, server_id: str, **updates: Any) -> McpServer:
        reconnect_fields = {
            "transport",
            "url",
            "command",
            "args",
            "env",
            "headers",
            "is_active",
            "timeout_seconds",
        }
        changed_connection = bool(reconnect_fields.intersection(updates))
        async with self.db.session() as session:
            row = await session.get(McpServer, server_id)
            if not isinstance(row, McpServer):
                raise NotFoundError(f"MCP server {server_id!r} was not found")
            for field_name, value in updates.items():
                if field_name in ("env", "headers"):
                    value = self._secrets.encrypt_map(value)
                setattr(row, "metadata_" if field_name == "metadata" else field_name, value)
            await session.flush()
        if changed_connection:
            client = self._clients.pop(server_id, None)
            if client is not None:
                await client.aclose()
            self._last_errors.pop(server_id, None)
            self._breaker.reset(server_id)
            await self._load_records(force=True)
            await self.refresh(server_id)
        else:
            await self._load_records(force=True)
        return await self.get_server(server_id)

    async def diagnostics(self, server_id: str) -> dict[str, str] | None:
        await self._load_records()
        return self._last_errors.get(server_id)

    def server_name(self, server_id: str) -> str | None:
        record = self._records.get(server_id)
        return record.name if record is not None else None

    def runtime(self, server_id: str) -> dict[str, Any]:
        """Live state for operators: breaker state and a stdio server's recent stderr."""
        client = self._clients.get(server_id)
        return {
            "breaker": self._breaker.health(server_id).state.value,
            "stderr": list(client.stderr_tail) if client is not None else [],
        }

    async def get_server(self, server_id: str) -> McpServer:
        async with self.db.session() as session:
            row = await session.get(McpServer, server_id)
            if not isinstance(row, McpServer):
                raise NotFoundError(f"MCP server {server_id!r} was not found")
            return row

    async def list_servers(self) -> list[McpServer]:
        async with self.db.session() as session:
            result = await session.execute(select(McpServer).order_by(McpServer.name))
            return list(result.scalars().all())

    async def _load_records(self, *, force: bool = False) -> None:
        if self._records and not force:
            return
        async with self.db.session() as session:
            result = await session.execute(select(McpServer))
            rows = list(result.scalars().all())
            # Rows written before a key was configured are encrypted on first load.
            for row in rows:
                if self._secrets.needs_encryption(row.env):
                    row.env = self._secrets.encrypt_map(row.env)
                if self._secrets.needs_encryption(row.headers):
                    row.headers = self._secrets.encrypt_map(row.headers)
        for row in rows:
            existing = self._records.get(row.id)
            self._records[row.id] = _ServerRecord(
                id=row.id,
                name=row.name,
                transport=row.transport,
                url=row.url,
                command=row.command,
                args=list(row.args or []),
                env=self._secrets.decrypt_map(row.env),
                headers=self._secrets.decrypt_map(row.headers),
                is_active=row.is_active,
                tool_prefix=row.tool_prefix,
                description=row.description,
                metadata=dict(row.metadata_ or {}),
                timeout_seconds=row.timeout_seconds,
                health_status=row.health_status or "unknown",
                discovered_tools=list(row.discovered_tools or []),
                expires_at=existing.expires_at if existing else 0.0,
                last_error=existing.last_error if existing else self._last_errors.get(row.id),
            )
        current_ids = {row.id for row in rows}
        for stale_id in set(self._records) - current_ids:
            self._records.pop(stale_id, None)

    async def _discover(self, record: _ServerRecord) -> None:
        if record.transport == "stdio" and not record.command:
            error = {
                "message": "MCP stdio server is missing a command",
                "failure_kind": "configuration",
            }
            await self._mark_unhealthy(record, error)
            return
        if record.transport != "stdio" and (record.url is None or not record.url.strip()):
            error = {"message": "MCP HTTP server is missing a URL", "failure_kind": "configuration"}
            await self._mark_unhealthy(record, error)
            return
        try:
            client, tools = await self._connect(record)
        except Exception as exc:
            logger.warning(
                "MCP server discovery failed",
                extra={"mcp_server_id": record.id},
                exc_info=True,
            )
            error = _describe_failure(exc)
            await self._mark_unhealthy(record, error)
            return
        self._clients[record.id] = client
        record.health_status = "healthy"
        record.discovered_tools = tools
        record.last_error = None
        self._last_errors.pop(record.id, None)
        record.expires_at = time.monotonic() + self._cache_ttl
        await self._save_discovery(record)

    async def _connect(self, record: _ServerRecord) -> tuple[McpClient, list[dict[str, Any]]]:
        """Reuse or open the server's client and list its tools.

        Discovery has no side effects, so a transport failure or timeout is
        retried once with a fresh connection. A failed client is closed and
        forgotten, so the next discovery starts clean.
        """
        client = self._clients.get(record.id)
        for attempt in (1, 2):
            fresh = client is None
            if client is None:
                client = McpClient(
                    record.url,
                    record.headers,
                    timeout=record.timeout_seconds or self._default_timeout,
                    command=record.command if record.transport == "stdio" else None,
                    args=record.args,
                    env=record.env,
                )
            try:
                if fresh:
                    await client.initialize()
                return client, await client.list_tools()
            except Exception as exc:
                await client.aclose()
                self._clients.pop(record.id, None)
                client = None
                retryable = isinstance(exc, GatewayError) and _is_outage(exc)
                if attempt == 2 or not retryable:
                    raise
                await asyncio.sleep(_DISCOVERY_RETRY_DELAY)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _mark_unhealthy(self, record: _ServerRecord, error: dict[str, str]) -> None:
        record.health_status = "unhealthy"
        record.discovered_tools = []
        record.last_error = error
        self._last_errors[record.id] = error
        record.expires_at = time.monotonic() + self._cache_ttl
        await self._save_discovery(record)

    async def _save_discovery(self, record: _ServerRecord) -> None:
        async with self.db.session() as session:
            row = await session.get(McpServer, record.id)
            if row is not None:
                row.health_status = record.health_status
                row.last_health_check_at = datetime.now(UTC)
                row.discovered_tools = record.discovered_tools


def _is_outage(exc: GatewayError) -> bool:
    """Transport failures and timeouts; not JSON-RPC or tool-level errors."""
    return exc.code in (ErrorCode.PROVIDER_UNAVAILABLE, ErrorCode.PROVIDER_TIMEOUT)


def _describe_failure(exc: Exception) -> dict[str, str]:
    message = str(exc) or type(exc).__name__
    details = exc.details if hasattr(exc, "details") else {}
    failure_kind = details.get("failure_kind") if isinstance(details, dict) else None
    if failure_kind is None:
        text = message.lower()
        if "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text:
            failure_kind = "authentication"
        elif "timeout" in text or "timed out" in text:
            failure_kind = "timeout"
        elif "protocol" in text or "json-rpc" in text or "invalid json" in text:
            failure_kind = "protocol_error"
        elif "refused" in text or "connect" in text or "start mcp stdio" in text:
            failure_kind = "connection"
        else:
            failure_kind = "unknown"
    return {"message": message, "failure_kind": str(failure_kind)}
