"""The application factory: routers, error envelopes, CORS, and the console mount.

The app is built but its lifespan is never entered, so no datastore is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import main
from app.config.settings import Settings
from app.core.errors import NotFoundError, RateLimitExceededError


def _settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]


def test_error_handlers_render_envelopes_and_retry_after() -> None:
    app = main.create_app(_settings(ui_enabled=False))

    @app.get("/limited")
    async def limited() -> None:
        raise RateLimitExceededError("slow down", retry_after=2.2)

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("secret internals")

    @app.get("/missing")
    async def missing() -> None:
        raise NotFoundError("nothing here")

    client = TestClient(app, raise_server_exceptions=False)
    not_found = client.get("/missing")
    assert not_found.status_code == 404 and "retry-after" not in not_found.headers
    limited_response = client.get("/limited")
    assert limited_response.status_code == 429
    assert limited_response.headers["retry-after"] == "3"
    assert limited_response.json()["error"]["code"] == "rate_limit_exceeded"

    boom_response = client.get("/boom")
    assert boom_response.status_code == 500
    assert boom_response.json()["error"]["code"] == "internal_error"
    assert "secret internals" not in boom_response.text


def test_cors_is_only_added_when_origins_are_configured() -> None:
    def has_cors(app: FastAPI) -> bool:
        return any(m.cls.__name__ == "CORSMiddleware" for m in app.user_middleware)  # type: ignore[attr-defined]

    assert has_cors(main.create_app(_settings(cors_origins=["http://x"], ui_enabled=False)))
    assert not has_cors(main.create_app(_settings(cors_origins=[], ui_enabled=False)))


def test_optional_routers_that_fail_to_import_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main,
        "_ROUTER_MODULES",
        (*main._ROUTER_MODULES, ("app.api.does_not_exist", "ghost", False)),
    )
    app = main.create_app(_settings(ui_enabled=False))
    assert "ghost" not in app.state.mounted_routers
    assert "admin" in app.state.mounted_routers


@pytest.fixture
def console_dir(tmp_path: Path) -> Path:
    (tmp_path / "index.html").write_text("<html>console</html>", encoding="utf-8")
    (tmp_path / "app.js").write_text("console.log(1)", encoding="utf-8")
    return tmp_path


def _console_app(static_dir: Path | None, **settings: object) -> TestClient:
    app = FastAPI()
    main._mount_console(app, _settings(**settings), str(static_dir) if static_dir else None)
    return TestClient(app)


def test_console_serves_assets_and_falls_back_to_index(console_dir: Path) -> None:
    client = _console_app(console_dir)
    assert client.get("/ui/app.js").text == "console.log(1)"
    # Client-side routes get the SPA shell; missing assets stay 404.
    assert client.get("/ui/logs/req-1").text == "<html>console</html>"
    assert client.get("/ui/missing.js").status_code == 404
    # Other HTTP errors from the static handler are not swallowed.
    assert client.post("/ui/app.js").status_code == 405


def test_console_is_skipped_when_disabled_or_not_built(tmp_path: Path) -> None:
    disabled = _console_app(tmp_path, ui_enabled=False)
    assert disabled.get("/ui/").status_code == 404
    unbuilt = _console_app(tmp_path)  # no index.html
    assert unbuilt.get("/ui/").status_code == 404


def test_default_console_directory_is_inside_the_package() -> None:
    assert Path(main.CONSOLE_DIR).parent == Path(main.__file__).parent
