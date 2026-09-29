"""Admin API edge cases: validation, not-found paths, fallbacks, and failure modes.

Runs the real admin router on SQLite with stubbed registry and Redis (see the
``admin_api`` fixture in ``test_admin_api``).
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import admin as admin_module
from app.api.deps import get_current_admin, require_role
from app.auth.console import create_refresh_token, hash_password
from app.core.errors import ErrorCode, NotFoundError, ProviderError
from app.db.models import AdminUser, GuardrailViolation, RequestLog, UsageRollup
from app.providers.base import Deployment
from app.routing.breaker import CircuitBreaker
from tests.unit.test_admin_api import StubRedis, token

ADMIN = ("admin@example.test", "correct-horse-battery")
VIEWER = ("viewer@example.test", "viewer-password-long")


def _headers(client: TestClient, who: tuple[str, str] = ADMIN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token(client, *who)}"}


def _seed(db: Any, *rows: Any) -> None:
    async def run() -> None:
        async with db.session() as session:
            session.add_all(list(rows))

    asyncio.run(run())


class RevocationRedis(StubRedis):
    """StubRedis plus the string commands token revocation needs."""

    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value

    async def mget(self, *keys: str) -> list[str | None]:
        return [self.values.get(key) for key in keys]


# -- Dependencies -----------------------------------------------------------


def test_require_role_rejects_unknown_roles_at_definition_time() -> None:
    with pytest.raises(ValueError, match="Unsupported console role"):
        require_role("owner")


def test_missing_gateway_state_is_503() -> None:
    app = FastAPI()
    app.include_router(admin_module.router)
    with TestClient(app) as client:
        response = client.get("/admin/api/auth/me", headers={"Authorization": "Bearer x"})
    assert response.status_code == 503


def test_tokens_without_subject_or_for_inactive_users_are_rejected(admin_api: Any) -> None:
    import jwt

    client, _, db = admin_api
    settings = client.app.state.gateway.settings
    missing_sub = jwt.encode(
        {"type": "access", "exp": int(time.time()) + 60},
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    response = client.get("/admin/api/auth/me", headers={"Authorization": f"Bearer {missing_sub}"})
    assert response.status_code == 401
    assert "subject" in response.json()["detail"]

    headers = _headers(client, VIEWER)

    async def deactivate() -> None:
        from sqlalchemy import update

        async with db.session() as session:
            await session.execute(
                update(AdminUser).where(AdminUser.email == VIEWER[0]).values(is_active=False)
            )

    asyncio.run(deactivate())
    assert client.get("/admin/api/auth/me", headers=headers).status_code == 401
    login = client.post("/admin/api/auth/login", json={"email": VIEWER[0], "password": VIEWER[1]})
    assert login.status_code == 401


# -- Sessions ---------------------------------------------------------------


def test_refresh_rotation_logout_and_revocation(admin_api: Any) -> None:
    client, _, _ = admin_api
    client.app.state.gateway.redis = RevocationRedis()
    login = client.post("/admin/api/auth/login", json={"email": ADMIN[0], "password": ADMIN[1]})
    refresh = login.json()["refresh_token"]

    assert client.post("/admin/api/auth/refresh", json={"refresh_token": "junk"}).status_code == 401
    rotated = client.post("/admin/api/auth/refresh", json={"refresh_token": refresh})
    assert rotated.status_code == 200
    reused = client.post("/admin/api/auth/refresh", json={"refresh_token": refresh})
    assert reused.status_code == 401
    assert "revoked" in reused.json()["detail"]

    ghost = create_refresh_token("no-such-user")
    assert client.post("/admin/api/auth/refresh", json={"refresh_token": ghost}).status_code == 401

    access = rotated.json()["access_token"]
    new_refresh = rotated.json()["refresh_token"]
    headers = {"Authorization": f"Bearer {access}"}
    logout = client.post(
        "/admin/api/auth/logout", json={"refresh_token": new_refresh}, headers=headers
    )
    assert logout.status_code == 200
    assert client.get("/admin/api/auth/me", headers=headers).status_code == 401
    assert (
        client.post("/admin/api/auth/refresh", json={"refresh_token": new_refresh}).status_code
        == 401
    )

    # Bad tokens in a logout body are ignored, not errors.
    fresh = _headers(client)
    assert (
        client.post(
            "/admin/api/auth/logout", json={"refresh_token": "junk"}, headers=fresh
        ).status_code
        == 200
    )
    assert client.post("/admin/api/auth/logout", headers=_headers(client)).status_code == 200


def test_change_password_endpoints(admin_api: Any) -> None:
    client, _, _ = admin_api
    headers = _headers(client)
    wrong = client.post(
        "/admin/api/auth/change-password",
        json={"current_password": "nope", "new_password": "a-new-long-password"},
        headers=headers,
    )
    assert wrong.status_code == 400
    ok = client.post(
        "/admin/api/auth/change-password",
        json={"current_password": ADMIN[1], "new_password": "a-new-long-password"},
        headers=headers,
    )
    assert ok.status_code == 200
    login = client.post(
        "/admin/api/auth/login", json={"email": ADMIN[0], "password": "a-new-long-password"}
    )
    assert login.status_code == 200

    viewer = _headers(client, VIEWER)
    wrong_own = client.post(
        "/admin/api/users/me/change-password",
        json={"current_password": "nope", "new_password": "another-long-password"},
        headers=viewer,
    )
    assert wrong_own.status_code == 400


# -- Dashboard --------------------------------------------------------------


@pytest.mark.parametrize("window", ["abc", "h", "0h", "5y", "99999d"])
def test_invalid_windows_are_rejected(admin_api: Any, window: str) -> None:
    client, _, _ = admin_api
    response = client.get(f"/admin/api/dashboard/summary?window={window}", headers=_headers(client))
    assert response.status_code == 422


def _rollup(model: str = "model-a", **fields: Any) -> UsageRollup:
    bucket = fields.pop("bucket", datetime.now(UTC).replace(minute=0, second=0, microsecond=0))
    defaults: dict[str, Any] = {
        "request_count": 2,
        "success_count": 1,
        "error_count": 1,
        "cache_hit_count": 1,
        "total_tokens": 20,
        "cost_usd": 0.5,
    }
    return UsageRollup(bucket=bucket, model=model, provider="stub", **{**defaults, **fields})


def _log(request_id: str, **fields: Any) -> RequestLog:
    defaults: dict[str, Any] = {"model": "model-a", "provider": "stub", "status": "success"}
    return RequestLog(request_id=request_id, **{**defaults, **fields})


@pytest.mark.parametrize("interval", ["hour", "day"])
def test_timeseries_from_rollups(admin_api: Any, interval: str) -> None:
    client, _, db = admin_api
    _seed(db, _rollup(), _rollup("model-b"))
    response = client.get(
        f"/admin/api/dashboard/timeseries?interval={interval}&metric=cost",
        headers=_headers(client),
    )
    assert response.status_code == 200, response.text
    [point] = response.json()["points"]
    assert point["value"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [("requests", 2), ("successes", 1), ("errors", 1), ("tokens", 15), ("cache_hits", 1)],
)
def test_timeseries_falls_back_to_logs(admin_api: Any, metric: str, expected: float) -> None:
    client, _, db = admin_api
    _seed(
        db,
        _log("a", status="success", total_tokens=10, cache_hit=True, cost_usd=0.1),
        _log("b", status="error", total_tokens=5, cache_hit=False, cost_usd=0.0),
    )
    response = client.get(
        f"/admin/api/dashboard/timeseries?interval=day&metric={metric}", headers=_headers(client)
    )
    assert response.status_code == 200, response.text
    assert sum(point["value"] for point in response.json()["points"]) == expected


# -- Keys -------------------------------------------------------------------


def test_key_lifecycle_and_not_found_paths(admin_api: Any) -> None:
    client, _, _ = admin_api
    headers = _headers(client)
    assert (
        client.post(
            "/admin/api/keys", json={"name": "k", "team_id": "ghost"}, headers=headers
        ).status_code
        == 404
    )
    created = client.post("/admin/api/keys", json={"name": "k"}, headers=headers).json()
    key_id = created["id"]
    for method, path in [
        ("get", "/admin/api/keys/missing"),
        ("patch", "/admin/api/keys/missing"),
        ("delete", "/admin/api/keys/missing"),
        ("post", "/admin/api/keys/missing/regenerate"),
    ]:
        kwargs: dict[str, Any] = {"json": {"name": "x"}} if method == "patch" else {}
        assert getattr(client, method)(path, headers=headers, **kwargs).status_code == 404
    assert (
        client.patch(
            f"/admin/api/keys/{key_id}", json={"team_id": "ghost"}, headers=headers
        ).status_code
        == 404
    )
    updated = client.patch(
        f"/admin/api/keys/{key_id}",
        json={"enabled": False, "budget_duration": "daily", "metadata": {"a": 1}},
        headers=headers,
    ).json()
    assert updated["enabled"] is False
    regenerated = client.post(f"/admin/api/keys/{key_id}/regenerate", headers=headers).json()
    assert regenerated["key"] != created["key"]
    assert client.get(f"/admin/api/keys/{key_id}", headers=headers).status_code == 200
    assert client.get("/admin/api/keys", headers=headers).json()["total"] == 1
    assert client.delete(f"/admin/api/keys/{key_id}", headers=headers).status_code == 200


# -- Teams ------------------------------------------------------------------


def test_team_crud_conflicts_and_usage(admin_api: Any) -> None:
    client, _, db = admin_api
    headers = _headers(client)
    team = client.post("/admin/api/teams", json={"name": "alpha"}, headers=headers).json()
    other = client.post("/admin/api/teams", json={"name": "beta"}, headers=headers).json()
    assert (
        client.post("/admin/api/teams", json={"name": "alpha"}, headers=headers).status_code == 409
    )
    assert (
        client.patch(
            f"/admin/api/teams/{other['id']}", json={"name": "alpha"}, headers=headers
        ).status_code
        == 409
    )
    renamed = client.patch(
        f"/admin/api/teams/{team['id']}", json={"metadata": {"cc": "eng"}}, headers=headers
    ).json()
    assert renamed["metadata"] == {"cc": "eng"}
    assert client.get(f"/admin/api/teams/{team['id']}", headers=headers).status_code == 200
    assert [t["name"] for t in client.get("/admin/api/teams", headers=headers).json()["items"]] == [
        "alpha",
        "beta",
    ]
    for method in ("get", "patch", "delete"):
        kwargs: dict[str, Any] = {"json": {}} if method == "patch" else {}
        assert (
            getattr(client, method)(
                "/admin/api/teams/missing", headers=headers, **kwargs
            ).status_code
            == 404
        )
    assert client.get("/admin/api/teams/missing/usage", headers=headers).status_code == 404

    # Usage from raw logs, then from rollups once they exist.
    _seed(
        db,
        _log("t1", team_id=team["id"], total_tokens=7, cost_usd=0.2),
        _log("t2", team_id=team["id"], status="error"),
    )
    raw = client.get(f"/admin/api/teams/{team['id']}/usage", headers=headers).json()
    assert (raw["requests"], raw["success_count"], raw["error_count"]) == (2, 1, 1)
    _seed(db, _rollup(team_id=team["id"]))
    rolled = client.get(f"/admin/api/teams/{team['id']}/usage", headers=headers).json()
    assert rolled["requests"] == 2 and rolled["cost_usd"] == pytest.approx(0.5)

    assert client.delete(f"/admin/api/teams/{other['id']}", headers=headers).status_code == 200


# -- Console users ----------------------------------------------------------


def test_user_management_edge_cases(admin_api: Any) -> None:
    client, _, db = admin_api
    headers = _headers(client)
    body = {"email": "new@example.test", "password": "a-long-password", "role": "viewer"}
    created = client.post("/admin/api/users", json=body, headers=headers)
    assert created.status_code == 201
    assert client.post("/admin/api/users", json=body, headers=headers).status_code == 409

    user_id = created.json()["id"]
    assert client.patch(f"/admin/api/users/{user_id}", json={}, headers=headers).status_code == 422
    assert (
        client.patch(
            f"/admin/api/users/{user_id}", json={"role": None}, headers=headers
        ).status_code
        == 422
    )
    assert (
        client.patch(
            "/admin/api/users/missing", json={"role": "admin"}, headers=headers
        ).status_code
        == 404
    )
    promoted = client.patch(f"/admin/api/users/{user_id}", json={"role": "admin"}, headers=headers)
    assert promoted.json()["role"] == "admin"
    # With two admins, one can be demoted and then deleted.
    assert (
        client.patch(
            f"/admin/api/users/{user_id}", json={"is_active": False}, headers=headers
        ).status_code
        == 200
    )
    assert client.delete("/admin/api/users/missing", headers=headers).status_code == 404
    assert client.delete(f"/admin/api/users/{user_id}", headers=headers).status_code == 200

    # A second active admin can be deleted while the first remains.
    _seed(
        db,
        AdminUser(
            email="second@example.test",
            password_hash=hash_password("second-admin-password"),
            role="admin",
            is_active=True,
        ),
    )
    users = client.get("/admin/api/users", headers=headers).json()["items"]
    second = next(u for u in users if u["email"] == "second@example.test")
    assert client.delete(f"/admin/api/users/{second['id']}", headers=headers).status_code == 200
    me = client.get("/admin/api/auth/me", headers=headers).json()
    assert client.delete(f"/admin/api/users/{me['id']}", headers=headers).status_code == 409


def _ghost(role: str, password: str) -> AdminUser:
    """A console user whose row disappears after authentication (concurrent delete)."""
    return AdminUser(
        id="ghost",
        email="ghost@example.test",
        role=role,
        is_active=True,
        password_hash=hash_password(password),
    )


def test_password_changes_for_users_deleted_mid_request(admin_api: Any) -> None:
    client, _, _ = admin_api
    overrides = client.app.dependency_overrides
    overrides[get_current_admin] = lambda: _ghost("viewer", "ghost-password-long")
    try:
        own = client.post(
            "/admin/api/users/me/change-password",
            json={"current_password": "ghost-password-long", "new_password": "another-long-pw"},
        )
        assert own.status_code == 401
        overrides[get_current_admin] = lambda: _ghost("admin", "ghost-password-long")
        admin = client.post(
            "/admin/api/auth/change-password",
            json={"current_password": "ghost-password-long", "new_password": "another-long-pw"},
        )
        assert admin.status_code == 404
    finally:
        overrides.clear()


# -- Providers, deployments, models -----------------------------------------


class FlakyProvider:
    async def health_check(self, deployment: Deployment) -> bool:
        if deployment.id.startswith("boom"):
            raise RuntimeError("health endpoint exploded")
        return True


def test_provider_status_reflects_breaker_states(admin_api: Any) -> None:
    client, registry, _ = admin_api
    gateway = client.app.state.gateway
    registry.provider = FlakyProvider()
    registry.deployments = [
        Deployment(id="open-a", model_name="m", provider="alpha", provider_model="x", api_key="k"),
        Deployment(id="half-b", model_name="m", provider="beta", provider_model="x", enabled=False),
        Deployment(id="half-b2", model_name="m", provider="beta", provider_model="x"),
        Deployment(id="boom-c", model_name="m", provider="gamma", provider_model="x"),
    ]
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=0)
    breaker.record_failure("open-a")
    breaker.record_failure("half-b2")
    breaker.is_available("half-b2")  # cooldown elapsed: moves to half-open
    breaker.record_failure("boom-c")
    gateway.breaker = breaker
    items = {
        row["provider"]: row
        for row in client.get("/admin/api/providers/status", headers=_headers(client)).json()[
            "items"
        ]
    }
    assert items["alpha"]["health_state"] == "open" and items["alpha"]["configured"]
    assert items["beta"]["health_state"] == "half_open" and not items["beta"]["configured"]
    assert items["gamma"]["reachable"] is False

    deployments = client.get("/admin/api/deployments", headers=_headers(client)).json()["items"]
    assert {d["id"]: d["health_state"] for d in deployments}["open-a"] == "open"


def test_models_skip_aliases_without_deployments(admin_api: Any) -> None:
    client, registry, _ = admin_api
    deployment = Deployment(id="d", model_name="real", provider="stub", provider_model="x")
    registry.list_models = lambda: ["alias", "real"]

    def deployments_for(name: str, include_disabled: bool = False) -> list[Deployment]:
        if name == "alias":
            raise NotFoundError("alias target missing")
        return [deployment]

    registry.deployments_for = deployments_for
    body = client.get("/admin/api/models", headers=_headers(client)).json()
    assert [item["name"] for item in body["items"]] == ["real"]
    assert body["total"] == 2


def test_deployment_health_check_not_found(admin_api: Any) -> None:
    client, registry, _ = admin_api

    def get_deployment(deployment_id: str) -> Deployment:
        raise NotFoundError(f"Unknown deployment: {deployment_id}")

    registry.get_deployment = get_deployment
    response = client.post("/admin/api/deployments/missing/health-check", headers=_headers(client))
    assert response.status_code == 404


# -- Logs -------------------------------------------------------------------


def test_log_filters_ordering_and_export(admin_api: Any) -> None:
    client, _, db = admin_api
    now = datetime.now(UTC)
    _seed(
        db,
        _log("fast", latency_ms=5.0, virtual_key_id="vk", team_id="t", cache_hit=False),
        _log(
            "slow",
            latency_ms=500.0,
            virtual_key_id="vk",
            team_id="t",
            cache_hit=True,
            error_message="upstream exploded",
            status="error",
        ),
    )
    headers = _headers(client)
    params = {
        "start": (now - timedelta(hours=1)).isoformat(),
        "end": (now + timedelta(hours=1)).isoformat(),
        "virtual_key_id": "vk",
        "team_id": "t",
        "provider": "stub",
        "status": "error",
        "cache_hit": "true",
        "min_latency_ms": 100,
        "search": "exploded",
        "order_by": "latency_ms",
        "order_dir": "asc",
    }
    listed = client.get("/admin/api/logs", params=params, headers=headers).json()
    assert [row["request_id"] for row in listed["items"]] == ["slow"]
    assert client.get("/admin/api/logs?order_by=secret", headers=headers).status_code == 422

    export = client.get("/admin/api/logs/export", headers=headers)
    assert export.headers["content-type"].startswith("text/csv")
    rows = list(csv.reader(io.StringIO(export.text)))
    assert rows[0][0] == "request_id"
    assert {row[0] for row in rows[1:]} == {"fast", "slow"}


def test_log_detail_not_found_and_viewer_cannot_reveal(admin_api: Any) -> None:
    client, _, db = admin_api
    _seed(db, _log("secret", request_body={"a": 1}))
    assert client.get("/admin/api/logs/missing", headers=_headers(client)).status_code == 404
    viewer = _headers(client, VIEWER)
    assert client.get("/admin/api/logs/secret?reveal=true", headers=viewer).status_code == 403


# -- Usage ------------------------------------------------------------------


@pytest.mark.parametrize("group_by", ["provider", "key", "team", "hour"])
def test_usage_groupings_from_rollups_and_logs(admin_api: Any, group_by: str) -> None:
    client, _, db = admin_api
    headers = _headers(client)
    _seed(db, _log("u1", total_tokens=3), _log("u2", status="error"))
    raw = client.get(f"/admin/api/usage?group_by={group_by}", headers=headers).json()
    assert sum(row["requests"] for row in raw["rows"]) == 2
    assert sum(row["error_count"] for row in raw["rows"]) == 1
    _seed(db, _rollup())
    rolled = client.get(f"/admin/api/usage?group_by={group_by}", headers=headers).json()
    assert sum(row["requests"] for row in rolled["rows"]) == 2


def test_cost_breakdown_from_logs_rollups_and_in_december(
    admin_api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, db = admin_api
    headers = _headers(client)
    _seed(db, _log("c1", cost_usd=0.25), _log("c2", status="error", cost_usd=0.25))
    raw = client.get("/admin/api/usage/costs", headers=headers).json()
    assert raw["total_cost_usd"] == pytest.approx(0.5)
    assert raw["by_model"][0]["error_count"] == 1

    _seed(db, _rollup(cost_usd=2.0))
    rolled = client.get("/admin/api/usage/costs", headers=headers).json()
    assert rolled["total_cost_usd"] == pytest.approx(2.0)

    class December(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
            return datetime(2026, 12, 15, 12, tzinfo=UTC)

    monkeypatch.setattr(admin_module, "datetime", December)
    december = client.get("/admin/api/usage/costs", headers=headers).json()
    assert december["projection"]["period_end"].startswith("2027-01-01")


# -- Guardrails -------------------------------------------------------------


def test_guardrail_violations_and_policies(admin_api: Any) -> None:
    client, _, db = admin_api
    _seed(
        db,
        GuardrailViolation(
            request_id="r",
            policy="default",
            rule="pii",
            phase="input",
            action="redact",
            severity="high",
            match_count=1,
            excerpt="x***y",
            details={},
        ),
    )
    headers = _headers(client)
    violations = client.get("/admin/api/guardrails/violations", headers=headers).json()
    assert violations["total"] == 1 and violations["items"][0]["rule"] == "pii"
    policies = client.get("/admin/api/guardrails/policies", headers=headers).json()
    assert "default" in [policy["name"] for policy in policies["items"]]


# -- Cache ------------------------------------------------------------------


class _Cache:
    available = True

    def __init__(self) -> None:
        self.invalidated: list[str | None] = []

    async def stats(self) -> dict[str, Any]:
        return {"entries": 3, "hits": 2, "misses": 1, "latency_saved_ms": 42.0}

    async def invalidate(self, namespace: str | None = None) -> int:
        self.invalidated.append(namespace)
        return 3


class OddInfoRedis(StubRedis):
    def __init__(self, info: Any) -> None:
        super().__init__()
        self.info = info

    async def execute_command(self, *args: str) -> Any:
        if isinstance(self.info, Exception):
            raise self.info
        return self.info


def test_cache_stats_with_a_live_cache_and_odd_index_info(admin_api: Any) -> None:
    client, _, _ = admin_api
    gateway = client.app.state.gateway
    gateway.components["cache"] = _Cache()
    headers = _headers(client)
    stats = client.get("/admin/api/cache/stats", headers=headers).json()
    assert (stats["hits"], stats["entries"], stats["estimated_latency_saved_ms"]) == (2, 3, 42.0)
    assert stats["index_size_bytes"] == 2 * 1024 * 1024

    gateway.redis = OddInfoRedis({"not": "a list"})
    assert client.get("/admin/api/cache/stats", headers=headers).json()["index_size_bytes"] is None
    gateway.redis = OddInfoRedis([b":num_docs", b"3"])
    assert client.get("/admin/api/cache/stats", headers=headers).json()["index_size_bytes"] is None
    gateway.redis = OddInfoRedis(RuntimeError("no index"))
    assert client.get("/admin/api/cache/stats", headers=headers).json()["index_size_bytes"] is None


def test_cache_entries_decode_edge_cases(admin_api: Any) -> None:
    client, _, _ = admin_api
    redis = client.app.state.gateway.redis
    redis.hashes = {
        "aigw:cache:entry:ns:1": {
            b"namespace": b"ns",
            b"created_at": b"not-a-number",
            b"response": b"{not json",
            b"prompt": b"key sk-1234567890abcdef",
            b"hit_count": b"7",
            b"embedding": b"\x00\x01",
        },
        "aigw:cache:entry:ns:2": {"response": json.dumps(["not", "a", "dict"])},
    }
    items = client.get("/admin/api/cache/entries", headers=_headers(client)).json()["items"]
    first, second = items
    assert first["age_seconds"] is None and first["model"] is None
    assert first["hit_count"] == 7
    assert "sk-1234567890abcdef" not in first["cached_prompt"]
    assert second["model"] is None and second["hit_count"] is None


def test_cache_routes_without_redis(admin_api: Any) -> None:
    client, _, _ = admin_api
    client.app.state.gateway.redis = None
    headers = _headers(client)
    assert client.get("/admin/api/cache/entries", headers=headers).status_code == 503
    assert (
        client.post(
            "/admin/api/cache/invalidate", json={"all_entries": True}, headers=headers
        ).status_code
        == 503
    )


def test_cache_invalidation_options(admin_api: Any) -> None:
    client, _, _ = admin_api
    gateway = client.app.state.gateway
    headers = _headers(client)
    assert client.post("/admin/api/cache/invalidate", json={}, headers=headers).status_code == 422
    both = {"key": "k", "all_entries": True}
    assert client.post("/admin/api/cache/invalidate", json=both, headers=headers).status_code == 422

    gateway.redis.hashes = {"aigw:cache:entry:k": {}}
    by_key = client.post(
        "/admin/api/cache/invalidate", json={"key": "aigw:cache:entry:k"}, headers=headers
    )
    assert by_key.json()["invalidated"] == 1

    # Without a cache component, entries are scanned and deleted in batches of 500.
    gateway.redis.hashes = {f"aigw:cache:entry:ns:{i}": {} for i in range(501)}
    swept = client.post("/admin/api/cache/invalidate", json={"all_entries": True}, headers=headers)
    assert swept.json()["invalidated"] == 501
    gateway.redis.hashes = {"aigw:cache:entry:ns:1": {}, "aigw:cache:entry:other:1": {}}
    scoped = client.post("/admin/api/cache/invalidate", json={"namespace": "ns"}, headers=headers)
    assert scoped.json()["invalidated"] == 1

    cache = _Cache()
    gateway.components["cache"] = cache
    delegated = client.post(
        "/admin/api/cache/invalidate", json={"namespace": "ns"}, headers=headers
    )
    assert delegated.json()["invalidated"] == 3 and cache.invalidated == ["ns"]


# -- System and playground --------------------------------------------------


def test_system_info_survives_failing_health_checks(admin_api: Any) -> None:
    client, _, _ = admin_api
    gateway = client.app.state.gateway

    async def explode() -> bool:
        raise RuntimeError("check failed")

    gateway.db.healthy = explode
    gateway.redis_healthy = explode
    info = client.get("/admin/api/system/info", headers=_headers(client)).json()
    assert info["database_connected"] is False and info["redis_connected"] is False


class StreamingPipeline:
    def __init__(self, *, fail: bool = False, route: bool = True) -> None:
        self.fail = fail
        self.route = route

    async def run(self, ctx: Any) -> Any:
        from app.core.schemas import ChatResponse, Choice, Message, Role

        return ChatResponse(
            model="m",
            choices=[Choice(message=Message(role=Role.ASSISTANT, content="unrouted"))],
            provider="stub",
            deployment_id="direct",
        )

    async def run_stream(self, ctx: Any) -> AsyncIterator[Any]:
        from app.core.pipeline import RoutingDecision
        from app.core.schemas import ChatResponse, StreamChunk

        if self.route:
            ctx.routing = RoutingDecision(
                deployment=Deployment(id="d", model_name="m", provider="stub", provider_model="x"),
                strategy="priority",
                reason="test",
            )
        else:
            ctx.response = ChatResponse(model="m", choices=[], provider="cache")
        yield StreamChunk(model="m", content="partial")
        if self.fail:
            raise ProviderError(ErrorCode.PROVIDER_ERROR, "stream broke")


def _events(body: str) -> list[tuple[str, str]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        name = next((line[7:] for line in lines if line.startswith("event: ")), "message")
        data = next(line[6:] for line in lines if line.startswith("data: "))
        events.append((name, data))
    return events


@pytest.mark.parametrize(("fail", "route"), [(False, True), (True, True), (False, False)])
def test_playground_streaming(admin_api: Any, fail: bool, route: bool) -> None:
    client, _, _ = admin_api
    gateway = client.app.state.gateway
    pipeline = StreamingPipeline(fail=fail, route=route)
    gateway.require_pipeline = lambda: pipeline
    response = client.post(
        "/admin/api/playground/chat",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        headers=_headers(client),
    )
    events = _events(response.text)
    names = [name for name, _ in events]
    assert names[0] == "message" and names[-1] == "message"
    assert ("error" in names) is fail
    metadata = json.loads(next(data for name, data in events if name == "metadata"))
    assert metadata["provider"] == ("stub" if route else "cache")
    assert events[-1][1] == "[DONE]"


def test_playground_unary_without_routing_and_invalid_messages(admin_api: Any) -> None:
    client, _, _ = admin_api
    gateway = client.app.state.gateway
    pipeline = StreamingPipeline()
    gateway.require_pipeline = lambda: pipeline
    headers = _headers(client)
    body = client.post(
        "/admin/api/playground/chat",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    ).json()
    assert body["routing_decision"] is None and body["deployment_id"] == "direct"
    invalid = client.post(
        "/admin/api/playground/chat",
        json={"model": "m", "messages": [{"role": "wizard", "content": "hi"}]},
        headers=headers,
    )
    assert invalid.status_code == 422


def test_dashboard_latency_percentiles_from_logs(admin_api: Any) -> None:
    client, _, db = admin_api
    _seed(db, *[_log(f"lat{i}", latency_ms=float(i * 10)) for i in range(1, 11)])
    summary = client.get("/admin/api/dashboard/summary", headers=_headers(client)).json()
    assert summary["p50_latency_ms"] == pytest.approx(50.0)
    assert summary["p99_latency_ms"] == pytest.approx(100.0)


def test_timeseries_hourly_from_logs(admin_api: Any) -> None:
    client, _, db = admin_api
    _seed(db, _log("h1", cost_usd=0.5))
    points = client.get(
        "/admin/api/dashboard/timeseries?interval=hour&metric=cost", headers=_headers(client)
    ).json()["points"]
    assert [point["value"] for point in points] == [0.5]


def test_key_hash_collision_is_a_conflict(admin_api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = admin_api
    headers = _headers(client)
    monkeypatch.setattr(
        admin_module, "generate_key", lambda: ("sk-aigw-x", "same-hash", "sk-aigw-x")
    )
    assert client.post("/admin/api/keys", json={"name": "a"}, headers=headers).status_code == 201
    assert client.post("/admin/api/keys", json={"name": "b"}, headers=headers).status_code == 409


def test_provider_with_only_disabled_deployments_is_unreachable(admin_api: Any) -> None:
    client, registry, _ = admin_api
    registry.deployments = [
        Deployment(id="off", model_name="m", provider="delta", provider_model="x", enabled=False)
    ]
    [row] = client.get("/admin/api/providers/status", headers=_headers(client)).json()["items"]
    assert row["reachable"] is False and row["health_state"] == "closed"


def test_cache_stats_and_entries_when_unavailable_or_sparse(admin_api: Any) -> None:
    client, _, _ = admin_api
    gateway = client.app.state.gateway
    headers = _headers(client)
    gateway.redis.hashes = {"aigw:cache:entry:ns:1": {"namespace": "ns"}}
    [entry] = client.get("/admin/api/cache/entries", headers=headers).json()["items"]
    assert entry["model"] is None and entry["namespace"] == "ns"
    assert (
        client.post(
            "/admin/api/cache/invalidate", json={"namespace": "none"}, headers=headers
        ).json()["invalidated"]
        == 0
    )
    gateway.redis = None
    stats = client.get("/admin/api/cache/stats", headers=headers).json()
    assert stats["available"] is False and stats["hits"] is None
