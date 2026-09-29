"""The operator CLI, run end to end against a throwaway SQLite database."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select

from app import cli
from app.config.settings import Settings, get_settings
from app.db.models import AdminUser, VirtualKey
from app.db.session import Database


@pytest.fixture
def database_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    url = f"sqlite+aiosqlite:///{tmp_path / 'cli.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    yield url
    get_settings.cache_clear()


def _rows(url: str, model: type[Any]) -> list[Any]:
    async def fetch() -> list[Any]:
        db = Database(Settings(database_url=url))
        await db.startup()
        try:
            async with db.session() as session:
                return list((await session.scalars(select(model))).all())
        finally:
            await db.shutdown()

    return asyncio.run(fetch())


def test_init_db_create_admin_and_create_key(
    database_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init-db"]) == 0
    assert "Schema created." in capsys.readouterr().out

    assert cli.main(["create-admin", "--email", "Ops@Example.com", "--password", "pw"]) == 0
    assert "Created admin ops@example.com" in capsys.readouterr().out
    [admin] = _rows(database_url, AdminUser)
    assert admin.role == "admin"

    assert (
        cli.main(
            [
                "create-key",
                "--name",
                "smoke",
                "--budget",
                "2.5",
                "--rpm",
                "10",
                "--model",
                "gpt-4o",
                "--model",
                "claude",
            ]
        )
        == 0
    )
    raw_key = capsys.readouterr().out.strip()
    assert raw_key.startswith("sk-aigw-")
    [key] = _rows(database_url, VirtualKey)
    assert key.max_budget_usd == 2.5
    assert key.rpm_limit == 10
    assert key.allowed_models == ["gpt-4o", "claude"]

    assert cli.main(["create-key", "--name", "unlimited"]) == 0
    unlimited = next(k for k in _rows(database_url, VirtualKey) if k.name == "unlimited")
    assert unlimited.max_budget_usd is None
    assert unlimited.allowed_models == []


def test_create_admin_rejects_unknown_roles() -> None:
    with pytest.raises(SystemExit):
        cli.main(["create-admin", "--email", "a@b.c", "--password", "x", "--role", "owner"])


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


@pytest.mark.parametrize(
    ("command", "function", "message"),
    [
        ("migrate", "upgrade", "Database migrated to head."),
        ("db-stamp", "stamp", "Database stamped at head"),
    ],
)
def test_migration_commands(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    function: str,
    message: str,
) -> None:
    from app.db import migrate

    called: list[str] = []
    monkeypatch.setattr(migrate, function, lambda revision: called.append(revision))
    assert cli.main([command]) == 0
    assert called == ["head"]
    assert message in capsys.readouterr().out


def test_serve_passes_arguments_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda target, **kwargs: seen.update(kwargs, target=target))
    assert cli.main(["serve", "--host", "127.0.0.1", "--port", "9999", "--reload"]) == 0
    assert seen == {
        "target": "app.main:app",
        "host": "127.0.0.1",
        "port": 9999,
        "reload": True,
        "factory": False,
    }
    get_settings.cache_clear()
    assert cli.main(["serve"]) == 0
    assert seen["host"] == get_settings().host and seen["port"] == get_settings().port


def test_walk_routes_descends_into_included_routers() -> None:
    inner = SimpleNamespace(routes=[SimpleNamespace(path="/inner", methods={"GET"})])
    routes = [
        SimpleNamespace(path="/a", methods={"POST", "GET"}),
        SimpleNamespace(original_router=inner),
        SimpleNamespace(),  # a mount without a path
        SimpleNamespace(path="/ws", methods=None),
    ]
    assert cli._walk_routes(routes) == [
        {"path": "/a", "methods": ["GET", "POST"]},
        {"path": "/inner", "methods": ["GET"]},
        {"path": "/ws", "methods": []},
    ]


@pytest.mark.parametrize("as_json", [True, False])
def test_routes_lists_the_mounted_api(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], as_json: bool
) -> None:
    fake_app = SimpleNamespace(
        routes=[
            SimpleNamespace(path="/v1/chat/completions", methods={"POST"}),
            SimpleNamespace(path="/ws", methods=None),
        ]
    )
    monkeypatch.setattr("app.main.create_app", lambda: fake_app)
    assert cli.main(["routes", "--json"] if as_json else ["routes"]) == 0
    captured = capsys.readouterr()
    if as_json:
        assert json.loads(captured.out)[0] == {"path": "/v1/chat/completions", "methods": ["POST"]}
    else:
        assert "POST" in captured.out and "-" in captured.out
    assert "2 routes" in captured.err
