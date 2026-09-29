from __future__ import annotations

import asyncio
import itertools
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from app.core.errors import ErrorCode, GatewayError


class McpClient(httpx.AsyncClient):
    """JSON-RPC client supporting MCP streamable HTTP and stdio transports."""

    def __init__(
        self,
        base_url: str | None = None,
        headers: Mapping[str, str] | None = None,
        *,
        timeout: float | httpx.Timeout = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        command: str | None = None,
        args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url or "http://localhost",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                **dict(headers or {}),
            },
            timeout=timeout,
            transport=transport,
        )
        self._request_ids = itertools.count(1)
        self._id_lock = asyncio.Lock()
        self._stdio_lock = asyncio.Lock()
        self.session_id: str | None = None
        self._command = command
        self._args = list(args)
        self._env = dict(env or {})
        self._process: asyncio.subprocess.Process | None = None

    async def initialize(
        self, *, request_timeout: float | httpx.Timeout | None = None
    ) -> dict[str, Any]:
        result = await self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "aigateway", "version": "0.1.0"},
            },
            request_timeout=request_timeout,
        )
        await self._notify("notifications/initialized")
        return result

    async def list_tools(
        self, *, request_timeout: float | httpx.Timeout | None = None
    ) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            result = await self._rpc(
                "tools/list", {"cursor": cursor} if cursor else {}, request_timeout=request_timeout
            )
            page = result.get("tools", [])
            if not isinstance(page, list):
                raise GatewayError(
                    ErrorCode.PROVIDER_ERROR,
                    "MCP tools/list returned an invalid tools value",
                )
            tools.extend(tool for tool in page if isinstance(tool, dict))
            cursor_value = result.get("nextCursor")
            cursor = cursor_value if isinstance(cursor_value, str) and cursor_value else None
            if cursor is None:
                return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        request_timeout: float | httpx.Timeout | None = None,
    ) -> dict[str, Any]:
        result = await self._rpc(
            "tools/call",
            {"name": name, "arguments": arguments},
            request_timeout=request_timeout,
        )
        if not isinstance(result.get("content", []), list):
            raise GatewayError(
                ErrorCode.PROVIDER_ERROR,
                "MCP tools/call returned an invalid content value",
            )
        return result

    async def list_resources(
        self, *, request_timeout: float | httpx.Timeout | None = None
    ) -> list[dict[str, Any]]:
        result = await self._rpc("resources/list", {}, request_timeout=request_timeout)
        resources = result.get("resources", [])
        if not isinstance(resources, list):
            raise GatewayError(
                ErrorCode.PROVIDER_ERROR,
                "MCP resources/list returned an invalid resources value",
            )
        return [resource for resource in resources if isinstance(resource, dict)]

    async def read_resource(
        self, uri: str, *, request_timeout: float | httpx.Timeout | None = None
    ) -> dict[str, Any]:
        return await self._rpc("resources/read", {"uri": uri}, request_timeout=request_timeout)

    async def close(self) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                async with asyncio.timeout(2.0):
                    await process.wait()
            except TimeoutError:
                process.kill()
                await process.wait()
        await super().aclose()

    async def _rpc(
        self,
        method: str,
        params: dict[str, Any],
        *,
        request_timeout: float | httpx.Timeout | None = None,
    ) -> dict[str, Any]:
        async with self._id_lock:
            request_id = next(self._request_ids)
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        response = await self._post(payload, request_timeout=request_timeout)
        if response is None or response.get("id") != request_id:
            raise GatewayError(
                ErrorCode.PROVIDER_ERROR,
                f"MCP {method} response did not match request id {request_id}",
            )
        error = response.get("error")
        if isinstance(error, dict):
            raise GatewayError(
                ErrorCode.PROVIDER_ERROR,
                f"MCP {method} failed: {error.get('message', 'JSON-RPC error')}",
                details={"jsonrpc_error": error, "failure_kind": "protocol_error"},
            )
        result = response.get("result", {})
        if not isinstance(result, dict):
            raise GatewayError(
                ErrorCode.PROVIDER_ERROR,
                f"MCP {method} returned a non-object result",
                details={"failure_kind": "protocol_error"},
            )
        return result

    async def _notify(self, method: str) -> None:
        await self._post({"jsonrpc": "2.0", "method": method}, allow_empty=True)

    async def _post(
        self,
        payload: dict[str, Any],
        *,
        request_timeout: float | httpx.Timeout | None = None,
        allow_empty: bool = False,
    ) -> dict[str, Any] | None:
        if self._command is not None:
            return await self._stdio_post(
                payload,
                request_timeout=request_timeout,
                allow_empty=allow_empty,
            )
        try:
            if request_timeout is None:
                response = await self.post("", json=payload)
            else:
                response = await self.post("", json=payload, timeout=request_timeout)
        except httpx.TimeoutException as exc:
            raise GatewayError(
                ErrorCode.PROVIDER_TIMEOUT,
                "MCP server request timed out",
                cause=exc,
                details={"failure_kind": "timeout"},
            ) from exc
        except httpx.RequestError as exc:
            raise GatewayError(
                ErrorCode.PROVIDER_UNAVAILABLE,
                f"MCP server transport failed: {exc}",
                cause=exc,
                details={"failure_kind": _failure_kind(str(exc))},
            ) from exc
        except httpx.HTTPError as exc:
            raise GatewayError(
                ErrorCode.PROVIDER_ERROR,
                f"MCP HTTP client failed: {exc}",
                cause=exc,
                details={"failure_kind": "protocol_error"},
            ) from exc

        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self.session_id = session_id
            self.headers["Mcp-Session-Id"] = session_id
        if response.is_error:
            failure_kind = (
                "authentication" if response.status_code in {401, 403} else "protocol_error"
            )
            raise GatewayError(
                ErrorCode.PROVIDER_ERROR,
                f"MCP server returned HTTP {response.status_code}",
                status_code=response.status_code,
                details={"failure_kind": failure_kind, "http_status": response.status_code},
            )
        if not response.content:
            if allow_empty:
                return None
            raise GatewayError(ErrorCode.PROVIDER_ERROR, "MCP server returned an empty response")
        try:
            if "text/event-stream" in response.headers.get("content-type", ""):
                messages = _parse_sse(response.text)
            else:
                decoded = response.json()
                messages = [decoded] if isinstance(decoded, dict) else []
        except (json.JSONDecodeError, ValueError) as exc:
            raise GatewayError(
                ErrorCode.PROVIDER_ERROR,
                "MCP server returned an invalid JSON-RPC response",
                cause=exc,
                details={"failure_kind": "protocol_error"},
            ) from exc
        if allow_empty:
            return None
        request_id = payload.get("id")
        return next(
            (
                message
                for message in messages
                if isinstance(message, dict) and message.get("id") == request_id
            ),
            None,
        )

    async def _stdio_post(
        self,
        payload: dict[str, Any],
        *,
        request_timeout: float | httpx.Timeout | None,
        allow_empty: bool,
    ) -> dict[str, Any] | None:
        async with self._stdio_lock:
            process = await self._ensure_process()
            if process.stdin is None or process.stdout is None:
                raise GatewayError(
                    ErrorCode.PROVIDER_UNAVAILABLE,
                    "MCP stdio process has no usable stdin/stdout",
                    details={"failure_kind": "connection"},
                )
            process.stdin.write(json.dumps(payload).encode() + b"\n")
            try:
                await process.stdin.drain()
                if allow_empty:
                    return None
                limit = _timeout_seconds(request_timeout, self.timeout)
                async with asyncio.timeout(limit):
                    while True:
                        line = await process.stdout.readline()
                        if not line:
                            raise GatewayError(
                                ErrorCode.PROVIDER_UNAVAILABLE,
                                f"MCP stdio process exited with status {process.returncode}",
                                details={"failure_kind": "connection"},
                            )
                        try:
                            response = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise GatewayError(
                                ErrorCode.PROVIDER_ERROR,
                                "MCP stdio server emitted invalid JSON-RPC data",
                                cause=exc,
                                details={"failure_kind": "protocol_error"},
                            ) from exc
                        if isinstance(response, dict) and response.get("id") == payload.get("id"):
                            return response
            except TimeoutError as exc:
                raise GatewayError(
                    ErrorCode.PROVIDER_TIMEOUT,
                    "MCP stdio server request timed out",
                    cause=exc,
                    details={"failure_kind": "timeout"},
                ) from exc
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise GatewayError(
                    ErrorCode.PROVIDER_UNAVAILABLE,
                    f"MCP stdio process transport failed: {exc}",
                    cause=exc,
                    details={"failure_kind": "connection"},
                ) from exc

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        if self._process is not None and self._process.returncode is None:
            return self._process
        if not self._command:
            raise GatewayError(
                ErrorCode.PROVIDER_ERROR,
                "MCP stdio transport requires a command",
                details={"failure_kind": "configuration"},
            )
        environment = os.environ.copy()
        environment.update(self._env)
        try:
            self._process = await asyncio.create_subprocess_exec(
                self._command,
                *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=environment,
            )
        except OSError as exc:
            raise GatewayError(
                ErrorCode.PROVIDER_UNAVAILABLE,
                f"Unable to start MCP stdio command {self._command!r}: {exc}",
                cause=exc,
                details={"failure_kind": "connection"},
            ) from exc
        return self._process


def _timeout_seconds(timeout: float | httpx.Timeout | None, default: Any) -> float:
    if isinstance(timeout, (float, int)):
        return float(timeout)
    if isinstance(timeout, httpx.Timeout):
        return float(timeout.read or timeout.connect or 30.0)
    if isinstance(default, httpx.Timeout):
        return float(default.read or default.connect or 30.0)
    return float(default)


def _failure_kind(message: str) -> str:
    lowered = message.lower()
    if "refused" in lowered or "connect" in lowered:
        return "connection"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    return "connection"


def _parse_sse(body: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    data_lines: list[str] = []
    for line in [*body.splitlines(), ""]:
        if not line:
            if data_lines:
                parsed = json.loads("\n".join(data_lines))
                if isinstance(parsed, dict):
                    messages.append(parsed)
                data_lines.clear()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    return messages
