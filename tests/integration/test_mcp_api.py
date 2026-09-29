"""MCP over HTTP: server registration, discovery, tool calls, and authorization.

The gateway talks to a real MCP server (scripts/fake_mcp_server.py) over both
transports: streamable HTTP on a local port, and stdio as a subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from tests.integration.conftest import create_virtual_key

FAKE_MCP = str(Path(__file__).resolve().parents[2] / "scripts" / "fake_mcp_server.py")


def _register(
    client: TestClient, headers: dict[str, str], url: str, **fields: Any
) -> dict[str, Any]:
    response = client.post(
        "/v1/mcp/servers",
        json={"name": "tools", "transport": "http", "url": url, **fields},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def _call(client: TestClient, headers: dict[str, str], name: str, arguments: dict[str, Any]) -> Any:
    return client.post(
        "/v1/mcp/tools/call", json={"name": name, "arguments": arguments}, headers=headers
    )


class TestServers:
    def test_register_discovers_tools(
        self, client: TestClient, auth_headers: dict[str, str], fake_mcp_url: str
    ) -> None:
        server = _register(client, auth_headers, fake_mcp_url)
        assert server["health_status"] in {"healthy", "unknown"}

        tools = client.get("/v1/mcp/tools", headers=auth_headers).json()
        names = {tool["function"]["name"] for tool in tools}
        assert {"tools__echo", "tools__add", "tools__fail"} <= names

    def test_get_update_refresh_delete(
        self, client: TestClient, auth_headers: dict[str, str], fake_mcp_url: str
    ) -> None:
        server = _register(client, auth_headers, fake_mcp_url)
        path = f"/v1/mcp/servers/{server['id']}"

        assert client.get(path, headers=auth_headers).json()["name"] == "tools"

        patched = client.patch(path, json={"description": "demo"}, headers=auth_headers)
        assert patched.status_code == 200, patched.text
        assert patched.json()["description"] == "demo"

        refreshed = client.post(f"{path}/refresh", headers=auth_headers)
        assert refreshed.status_code == 200, refreshed.text

        assert client.get("/v1/mcp/servers", headers=auth_headers).json()[0]["id"] == server["id"]
        assert client.delete(path, headers=auth_headers).status_code == 204
        assert client.get(path, headers=auth_headers).status_code == 404

    def test_tool_prefix_namespaces_tools(
        self, client: TestClient, auth_headers: dict[str, str], fake_mcp_url: str
    ) -> None:
        _register(client, auth_headers, fake_mcp_url, tool_prefix="calc")
        names = {
            tool["function"]["name"]
            for tool in client.get("/v1/mcp/tools", headers=auth_headers).json()
        }
        assert "calc__add" in names

    def test_health_reports_each_server(
        self, client: TestClient, auth_headers: dict[str, str], fake_mcp_url: str
    ) -> None:
        server = _register(client, auth_headers, fake_mcp_url)
        health = client.get("/v1/mcp/health", headers=auth_headers)
        assert health.status_code == 200
        assert health.json().get(server["id"]) is True

    def test_unreachable_server_is_reported_unhealthy(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        server = _register(client, auth_headers, "http://127.0.0.1:9/mcp", name="dead")
        health = client.get("/v1/mcp/health", headers=auth_headers).json()
        assert health.get(server["id"]) is False

    def test_mismatched_transport_is_rejected(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/v1/mcp/servers",
            json={"name": "bad", "transport": "stdio", "url": "http://x"},
            headers=auth_headers,
        )
        assert response.status_code == 422


class TestToolCalls:
    def test_call_tool_over_http(
        self, client: TestClient, auth_headers: dict[str, str], fake_mcp_url: str
    ) -> None:
        _register(client, auth_headers, fake_mcp_url)
        response = _call(client, auth_headers, "tools__add", {"a": 2, "b": 3})
        assert response.status_code == 200, response.text
        message = response.json()["message"]
        assert message["role"] == "tool"
        assert "5" in str(message["content"])

    def test_invalid_arguments_are_reported_not_sent(
        self, client: TestClient, auth_headers: dict[str, str], fake_mcp_url: str
    ) -> None:
        _register(client, auth_headers, fake_mcp_url)
        response = _call(client, auth_headers, "tools__add", {"a": "two"})
        assert response.status_code == 200
        assert "error" in str(response.json()["message"]["content"]).lower()

    def test_tool_error_is_surfaced(
        self, client: TestClient, auth_headers: dict[str, str], fake_mcp_url: str
    ) -> None:
        _register(client, auth_headers, fake_mcp_url)
        response = _call(client, auth_headers, "tools__fail", {})
        assert response.status_code == 200
        assert "failed on purpose" in str(response.json()["message"]["content"])

    def test_call_tool_over_stdio(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.post(
            "/v1/mcp/servers",
            json={
                "name": "local",
                "transport": "stdio",
                "command": sys.executable,
                "args": [FAKE_MCP, "--stdio"],
            },
            headers=auth_headers,
        )
        assert response.status_code == 201, response.text
        called = _call(client, auth_headers, "local__echo", {"text": "hi from stdio"})
        assert called.status_code == 200, called.text
        assert "hi from stdio" in str(called.json()["message"]["content"])

    def test_virtual_key_can_call_tools(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
        fake_mcp_url: str,
    ) -> None:
        _register(client, auth_headers, fake_mcp_url)
        secret, _ = create_virtual_key(client, admin_headers)
        response = _call(
            client, {"Authorization": f"Bearer {secret}"}, "tools__echo", {"text": "hello"}
        )
        assert response.status_code == 200, response.text


class TestAuthorization:
    def test_virtual_key_cannot_register_servers(
        self, client: TestClient, admin_headers: dict[str, str], fake_mcp_url: str
    ) -> None:
        """A stdio server runs a host command; applications must not be able to add one."""
        secret, _ = create_virtual_key(client, admin_headers)
        headers = {"Authorization": f"Bearer {secret}"}
        stdio = client.post(
            "/v1/mcp/servers",
            json={"name": "evil", "transport": "stdio", "command": "/bin/sh", "args": ["-c", "id"]},
            headers=headers,
        )
        assert stdio.status_code == 403
        http = client.post(
            "/v1/mcp/servers",
            json={"name": "x", "transport": "http", "url": fake_mcp_url},
            headers=headers,
        )
        assert http.status_code == 403

    def test_viewer_is_read_only(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        viewer_headers: dict[str, str],
        fake_mcp_url: str,
    ) -> None:
        server = _register(client, auth_headers, fake_mcp_url)
        assert client.get("/v1/mcp/servers", headers=viewer_headers).status_code == 200
        assert client.get("/v1/mcp/tools", headers=viewer_headers).status_code == 200
        path = f"/v1/mcp/servers/{server['id']}"
        assert client.delete(path, headers=viewer_headers).status_code == 403
        assert client.patch(path, json={"name": "x"}, headers=viewer_headers).status_code == 403
        assert _call(client, viewer_headers, "tools__echo", {"text": "x"}).status_code == 403

    def test_console_admin_can_register(
        self, client: TestClient, admin_headers: dict[str, str], fake_mcp_url: str
    ) -> None:
        _register(client, admin_headers, fake_mcp_url)
