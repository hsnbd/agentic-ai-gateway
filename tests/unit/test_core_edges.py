"""Error envelopes, message flattening, and gateway state lifecycle."""

from __future__ import annotations

from typing import Any

import pytest

from app.config.settings import Settings
from app.core import state as state_module
from app.core.errors import (
    AllProvidersFailedError,
    ConfigurationError,
    ErrorCode,
    GatewayError,
    InvalidRequestError,
    NotFoundError,
    retry_after_header,
)
from app.core.schemas import ImagePart, Message, Role, TextPart
from app.core.state import GatewayState, get_state


def test_error_envelope_includes_optional_fields() -> None:
    error = GatewayError(
        ErrorCode.PROVIDER_ERROR,
        "failed",
        provider="openai",
        model="gpt-4o",
        details={"attempts": 2},
    )
    body = error.to_dict()["error"]
    assert body == {
        "message": "failed",
        "type": "provider_error",
        "code": "provider_error",
        "provider": "openai",
        "model": "gpt-4o",
        "details": {"attempts": 2},
    }
    assert repr(error) == "GatewayError(code='provider_error', message='failed')"
    assert "provider" not in InvalidRequestError("bad").to_dict()["error"]


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (NotFoundError("x"), 404),
        (AllProvidersFailedError("x"), 502),
        (ConfigurationError("x"), 500),
    ],
)
def test_error_subclasses_map_to_http_status(error: GatewayError, status: int) -> None:
    assert error.status_code == status


def test_retry_after_header_rounds_up_and_floors_at_one() -> None:
    assert retry_after_header(0.2) == "1"
    assert retry_after_header(2.1) == "3"


def test_message_text_flattens_parts_and_handles_none() -> None:
    assert Message(role=Role.ASSISTANT, content=None).text() == ""
    parts = Message(
        role=Role.USER,
        content=[
            TextPart(text="look at "),
            ImagePart(url="https://example.com/a.png"),
            TextPart(text="this"),
        ],
    )
    assert parts.text() == "look at this"


class _Recorder:
    def __init__(self) -> None:
        self.events: list[str] = []


class _Registry:
    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder

    async def startup(self) -> None:
        self.recorder.events.append("registry.startup")

    async def shutdown(self) -> None:
        self.recorder.events.append("registry.shutdown")


class _Db:
    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder

    async def startup(self) -> None:
        self.recorder.events.append("db.startup")

    async def create_all(self) -> None:
        self.recorder.events.append("db.create_all")

    async def shutdown(self) -> None:
        self.recorder.events.append("db.shutdown")


class _Redis:
    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy
        self.closed = False

    async def ping(self) -> None:
        if not self.healthy:
            raise ConnectionError("down")

    async def aclose(self) -> None:
        self.closed = True


class _McpRegistry:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _state(recorder: _Recorder, **settings: Any) -> GatewayState:
    state = GatewayState(Settings(**settings))
    state.registry = _Registry(recorder)  # type: ignore[assignment]
    state.db = _Db(recorder)  # type: ignore[assignment]
    return state


async def test_startup_builds_components_and_bootstraps_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    state = _state(recorder, auto_create_schema=True)
    redis = _Redis()
    monkeypatch.setattr(state_module.aioredis, "from_url", lambda *a, **k: redis)
    monkeypatch.setattr("app.core.builder.build_pipeline", lambda s: "pipeline")

    bootstrapped: list[bool] = []

    class _Console:
        def __init__(self, db: Any) -> None:
            pass

        async def ensure_bootstrap_admin(self) -> None:
            bootstrapped.append(True)

    monkeypatch.setattr("app.auth.console.ConsoleAuthService", _Console)
    await state.startup()
    assert recorder.events == ["registry.startup", "db.startup", "db.create_all"]
    assert state.redis is redis
    assert state.pipeline == "pipeline"
    assert state.require_pipeline() == "pipeline"
    assert {"rag_service", "mcp_registry"} <= set(state.components)
    assert bootstrapped == [True]


async def test_startup_can_skip_schema_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    state = _state(recorder, auto_create_schema=False)
    monkeypatch.setattr(state_module.aioredis, "from_url", lambda *a, **k: _Redis())
    monkeypatch.setattr("app.core.builder.build_pipeline", lambda s: "pipeline")

    async def no_admin(self: Any) -> None:
        return None

    monkeypatch.setattr(GatewayState, "_bootstrap_console_admin", no_admin)
    await state.startup()
    assert "db.create_all" not in recorder.events


def test_require_pipeline_before_startup_fails() -> None:
    with pytest.raises(RuntimeError, match="not initialised"):
        GatewayState(Settings()).require_pipeline()


async def test_shutdown_closes_everything_once() -> None:
    recorder = _Recorder()
    state = _state(recorder)
    redis, mcp = _Redis(), _McpRegistry()
    state.redis = redis  # type: ignore[assignment]
    state.components["mcp_registry"] = mcp
    await state.shutdown()
    assert redis.closed and mcp.closed
    assert state.redis is None
    assert recorder.events == ["db.shutdown", "registry.shutdown"]

    # Nothing to close the second time round.
    state.components.clear()
    await state.shutdown()


async def test_redis_health() -> None:
    state = GatewayState(Settings())
    assert not await state.redis_healthy()
    state.redis = _Redis()  # type: ignore[assignment]
    assert await state.redis_healthy()
    state.redis = _Redis(healthy=False)  # type: ignore[assignment]
    assert not await state.redis_healthy()


def test_get_state_reads_app_state() -> None:
    from types import SimpleNamespace

    gateway = GatewayState(Settings())
    app = SimpleNamespace(state=SimpleNamespace(gateway=gateway))
    assert get_state(app) is gateway  # type: ignore[arg-type]
