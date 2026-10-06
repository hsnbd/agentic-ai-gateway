"""MCP server credentials are encrypted at rest and never returned by the API."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config.settings import Settings
from app.core.errors import GatewayError
from app.db.models import McpServer
from app.db.session import Database
from app.mcp.registry import McpRegistry

KEY = "integration-secrets-key-that-is-long-enough"


@pytest.fixture
def extra_env() -> dict[str, str]:
    return {"SECRETS_ENCRYPTION_KEY": KEY}


def _with_db[T](fn: Callable[[Database], Awaitable[T]]) -> T:
    """Run ``fn`` against its own engine: the app's engine lives on the TestClient's loop."""

    async def run() -> T:
        db = Database(Settings())
        await db.startup()
        try:
            return await fn(db)
        finally:
            await db.shutdown()

    return asyncio.run(run())


def _row(server_id: str) -> McpServer:
    async def load(db: Database) -> McpServer:
        async with db.session() as session:
            row = await session.get(McpServer, server_id)
            assert row is not None
            return row

    return _with_db(load)


def _register(client: TestClient, headers: dict[str, str], url: str) -> dict[str, Any]:
    response = client.post(
        "/v1/mcp/servers",
        json={
            "name": "secured",
            "transport": "http",
            "url": url,
            "headers": {"Authorization": "Bearer upstream-token"},
            "env": {"API_TOKEN": "s3cret"},
        },
        headers=headers,
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def test_values_are_encrypted_in_the_database_and_redacted_by_the_api(
    client: TestClient, auth_headers: dict[str, str], fake_mcp_url: str
) -> None:
    server = _register(client, auth_headers, fake_mcp_url)
    assert server["health_status"] == "healthy"  # decrypted headers still reach the server
    assert server["headers"] == {"Authorization": "***"}
    assert server["env"] == {"API_TOKEN": "***"}

    row = _row(server["id"])
    assert row.headers["Authorization"].startswith("enc:v1:")
    assert row.env["API_TOKEN"].startswith("enc:v1:")
    assert "upstream-token" not in str(row.headers)

    listed = client.get("/v1/mcp/servers", headers=auth_headers).json()
    assert "upstream-token" not in str(listed)

    patched = client.patch(
        f"/v1/mcp/servers/{server['id']}",
        json={"headers": {"Authorization": "Bearer rotated"}},
        headers=auth_headers,
    )
    assert patched.status_code == 200
    assert _row(server["id"]).headers["Authorization"].startswith("enc:v1:")


def test_a_restarted_registry_decrypts_and_plaintext_rows_are_encrypted_on_load(
    client: TestClient, auth_headers: dict[str, str], fake_mcp_url: str
) -> None:
    server = _register(client, auth_headers, fake_mcp_url)

    async def scenario(db: Database) -> None:
        # A row written before encryption was configured.
        async with db.session() as session:
            legacy = McpServer(
                name="legacy",
                transport="http",
                url=fake_mcp_url,
                env={"TOKEN": "plain"},
                headers={"X-Key": "plain-header"},
            )
            session.add(legacy)
            await session.flush()
            legacy_id = legacy.id

        registry = McpRegistry(db, Settings())
        await registry._load_records(force=True)
        assert registry._records[server["id"]].headers == {"Authorization": "Bearer upstream-token"}
        assert registry._records[legacy_id].env == {"TOKEN": "plain"}

        async with db.session() as session:
            stored = await session.scalar(select(McpServer).where(McpServer.id == legacy_id))
            assert stored is not None
            assert stored.env["TOKEN"].startswith("enc:v1:")
            assert stored.headers["X-Key"].startswith("enc:v1:")

    _with_db(scenario)


def test_encrypted_rows_without_the_key_fail_loudly(
    client: TestClient,
    auth_headers: dict[str, str],
    fake_mcp_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register(client, auth_headers, fake_mcp_url)

    for configured in ("", "a-different-key-entirely"):
        monkeypatch.setenv("SECRETS_ENCRYPTION_KEY", configured)

        async def load(db: Database) -> None:
            await McpRegistry(db, Settings())._load_records(force=True)

        with pytest.raises(GatewayError, match="SECRETS_ENCRYPTION_KEY"):
            _with_db(load)
