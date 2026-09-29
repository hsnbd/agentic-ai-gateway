from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.admin import router
from app.api.deps import get_current_admin
from app.auth.console import hash_password
from app.config.settings import Settings
from app.core.errors import ConfigurationError
from app.core.pipeline import RoutingDecision
from app.core.schemas import ChatResponse, Choice, Message, Role, Usage
from app.db.models import AdminUser, RequestLog, UsageRollup
from app.db.session import Database
from app.providers.base import Deployment


class StubProvider:
    async def health_check(self, deployment: Deployment) -> bool:
        return deployment.id != "unreachable"


class StubRegistry:
    def __init__(self) -> None:
        self.provider_names = ["stub"]
        self.reload_error = False
        self.deployments: list[Deployment] = []
        self.provider = StubProvider()

    def list_models(self) -> list[str]:
        return ["stub-model"]

    def list_deployments(self) -> list[Deployment]:
        return list(self.deployments)

    def provider_for(self, deployment: Deployment) -> StubProvider:
        return self.provider

    def load_config(self, path: str) -> None:
        if self.reload_error:
            raise ConfigurationError("invalid YAML configuration")


class StubRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, Any]] = {}

    async def ping(self) -> bool:
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self.hashes:
                del self.hashes[key]
                removed += 1
        return removed

    async def scan_iter(self, match: str) -> Any:
        from fnmatch import fnmatch

        for key in sorted(self.hashes):
            if fnmatch(key, match):
                yield key

    async def hgetall(self, key: str) -> dict[str, Any]:
        return self.hashes.get(key, {})

    async def ttl(self, key: str) -> int:
        return 45 if key in self.hashes else -2

    async def execute_command(self, *args: str) -> list[str]:
        return ["vector_index_sz_mb", "2.0"]


@pytest.fixture
def admin_api(tmp_path: Path) -> Any:
    async def prepare(db: Database) -> None:
        await db.startup()
        await db.create_all()
        async with db.session() as session:
            session.add_all(
                [
                    AdminUser(
                        email="admin@example.test",
                        password_hash=hash_password("correct-horse-battery"),
                        role="admin",
                        is_active=True,
                    ),
                    AdminUser(
                        email="viewer@example.test",
                        password_hash=hash_password("viewer-password-long"),
                        role="viewer",
                        is_active=True,
                    ),
                ]
            )

    settings = Settings(database_url=f"sqlite+aiosqlite:///{tmp_path / 'admin.db'}")
    db = Database(settings)
    asyncio.run(prepare(db))
    registry = StubRegistry()
    state = SimpleNamespace(
        db=db,
        redis=StubRedis(),
        registry=registry,
        breaker=None,
        settings=settings,
        components={},
        pipeline=None,
        router=None,
        async_redis_healthy=None,
    )

    async def redis_healthy() -> bool:
        return True

    async def database_healthy() -> bool:
        return await db.healthy()

    state.redis_healthy = redis_healthy
    state.db.healthy = database_healthy
    app = FastAPI()
    app.include_router(router)
    app.state.gateway = state
    with TestClient(app) as client:
        yield client, registry, db
    asyncio.run(db.shutdown())


def token(client: TestClient, email: str, password: str) -> str:
    response = client.post("/admin/api/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    access_token = response.json().get("access_token")
    assert isinstance(access_token, str)
    return access_token


def test_login_issues_jwt_and_rejects_bad_password(admin_api: Any) -> None:
    client, _, _ = admin_api
    response = client.post(
        "/admin/api/auth/login",
        json={"email": "admin@example.test", "password": "correct-horse-battery"},
    )
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert response.json()["access_token"]
    invalid = client.post(
        "/admin/api/auth/login",
        json={"email": "admin@example.test", "password": "wrong-password"},
    )
    assert invalid.status_code == 401


def test_protected_route_requires_auth_and_viewer_cannot_write(admin_api: Any) -> None:
    client, _, _ = admin_api
    assert client.get("/admin/api/auth/me").status_code == 401
    viewer = token(client, "viewer@example.test", "viewer-password-long")
    denied = client.post(
        "/admin/api/teams",
        headers={"Authorization": f"Bearer {viewer}"},
        json={"name": "forbidden"},
    )
    assert denied.status_code == 403


def test_key_plaintext_is_returned_once(admin_api: Any) -> None:
    client, _, _ = admin_api
    admin = token(client, "admin@example.test", "correct-horse-battery")
    headers = {"Authorization": f"Bearer {admin}"}
    created = client.post("/admin/api/keys", headers=headers, json={"name": "one-time"})
    assert created.status_code == 201, created.text
    key_id = created.json()["id"]
    raw_key = created.json()["key"]
    assert raw_key.startswith("sk-aigw-")
    regenerated = client.post(f"/admin/api/keys/{key_id}/regenerate", headers=headers)
    assert regenerated.status_code == 200, regenerated.text
    replacement_key = regenerated.json()["key"]
    assert replacement_key.startswith("sk-aigw-")
    assert replacement_key != raw_key
    fetched = client.get(f"/admin/api/keys/{key_id}", headers=headers)
    assert fetched.status_code == 200
    assert "key" not in fetched.json()
    assert replacement_key not in fetched.text
    schema = client.get("/openapi.json").json()["components"]["schemas"]["VirtualKeyResponse"]
    assert "enabled" in schema["properties"]
    assert "budget_duration" in schema["properties"]
    assert "shown only at creation" in schema["properties"]["key"]["description"].lower()


def test_pagination_caps_limit_at_200(admin_api: Any) -> None:
    client, _, _ = admin_api
    admin = token(client, "admin@example.test", "correct-horse-battery")
    response = client.get(
        "/admin/api/teams?limit=1000", headers={"Authorization": f"Bearer {admin}"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["limit"] == 200


def test_log_filtering_narrows_results(admin_api: Any) -> None:
    client, _, db = admin_api

    async def seed() -> None:
        async with db.session() as session:
            session.add_all(
                [
                    RequestLog(
                        request_id="req-a", model="model-a", provider="stub", status="success"
                    ),
                    RequestLog(
                        request_id="req-b", model="model-b", provider="stub", status="success"
                    ),
                ]
            )

    asyncio.run(seed())
    admin = token(client, "admin@example.test", "correct-horse-battery")
    response = client.get(
        "/admin/api/logs?model=model-a", headers={"Authorization": f"Bearer {admin}"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["total"] == 1
    assert [row["model"] for row in response.json()["items"]] == ["model-a"]


def test_invalid_config_reload_returns_error_envelope(admin_api: Any) -> None:
    client, registry, _ = admin_api
    registry.reload_error = True
    admin = token(client, "admin@example.test", "correct-horse-battery")
    response = client.post("/admin/api/config/reload", headers={"Authorization": f"Bearer {admin}"})
    assert response.status_code == 400
    assert response.json()["detail"].startswith("Model configuration is invalid:")
    assert "error" not in response.json()


def test_dashboard_summary_uses_rollup_counts(admin_api: Any) -> None:
    client, _, db = admin_api

    async def seed() -> None:
        async with db.session() as session:
            bucket = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
            session.add(
                UsageRollup(
                    bucket=bucket,
                    model="model-a",
                    provider="stub",
                    request_count=4,
                    success_count=3,
                    error_count=1,
                    cache_hit_count=2,
                    total_tokens=90,
                    cost_usd=0.75,
                )
            )

    asyncio.run(seed())
    admin = token(client, "admin@example.test", "correct-horse-battery")
    response = client.get(
        "/admin/api/dashboard/summary?window=24h",
        headers={"Authorization": f"Bearer {admin}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["requests"] == 4
    assert response.json()["success_rate"] == pytest.approx(0.75)
    assert response.json()["cache_hit_ratio"] == pytest.approx(0.5)
    assert response.json()["total_tokens"] == 90


def test_dashboard_summary_falls_back_to_request_logs(admin_api: Any) -> None:
    client, _, db = admin_api

    async def seed() -> None:
        async with db.session() as session:
            session.add_all(
                [
                    RequestLog(
                        request_id="raw-success",
                        model="model-a",
                        provider="stub",
                        status="success",
                        cache_hit=True,
                        total_tokens=10,
                    ),
                    RequestLog(
                        request_id="raw-error",
                        model="model-b",
                        provider="stub",
                        status="error",
                        cache_hit=False,
                        total_tokens=5,
                    ),
                ]
            )

    asyncio.run(seed())
    admin = token(client, "admin@example.test", "correct-horse-battery")
    response = client.get(
        "/admin/api/dashboard/summary?window=24h",
        headers={"Authorization": f"Bearer {admin}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["success_rate"] == pytest.approx(0.5)
    assert response.json()["cache_hit_ratio"] == pytest.approx(0.5)
    assert response.json()["error_count"] == 1


def test_log_attempt_details_and_summary_fields_are_exposed(admin_api: Any) -> None:
    client, _, db = admin_api
    attempts = [
        {
            "deployment_id": "primary",
            "provider": "provider-a",
            "outcome": "error",
            "latency_ms": 123.4,
            "error": "upstream unavailable",
        },
        {
            "deployment_id": "backup",
            "provider": "provider-b",
            "outcome": "success",
            "latency_ms": 45.6,
            "error": None,
        },
    ]

    async def seed() -> None:
        async with db.session() as session:
            session.add(
                RequestLog(
                    request_id="attempted-request",
                    model="model-a",
                    provider="provider-b",
                    deployment_id="backup",
                    status="success",
                    latency_ms=169.0,
                    attempt_count=2,
                    fallback_used=True,
                    cache_similarity=0.93,
                    stage_timings={"routing": 2.0, "attempts": attempts},
                )
            )

    asyncio.run(seed())
    headers = {"Authorization": f"Bearer {token(client, 'admin@example.test', 'correct-horse-battery')}"}
    listing = client.get("/admin/api/logs", headers=headers)
    assert listing.status_code == 200, listing.text
    item = listing.json()["items"][0]
    assert item["attempt_count"] == 2
    assert item["fallback_count"] == 1
    assert item["cache_similarity"] == pytest.approx(0.93)

    detail = client.get("/admin/api/logs/attempted-request", headers=headers)
    assert detail.status_code == 200, detail.text
    result = detail.json()
    assert result["attempts"] == attempts
    assert "attempts" not in result["stage_timings"]


def test_log_bodies_are_redacted_by_default_and_reveal_is_admin_only(admin_api: Any) -> None:
    client, _, db = admin_api
    stored_request = {"messages": [{"role": "user", "content": "sensitive prompt text"}]}
    stored_response = {"choices": [{"message": {"content": "sensitive completion text"}}]}

    async def seed() -> None:
        async with db.session() as session:
            session.add(
                RequestLog(
                    request_id="sensitive-request",
                    model="model-a",
                    status="success",
                    request_body=stored_request,
                    response_body=stored_response,
                )
            )

    asyncio.run(seed())
    viewer_headers = {
        "Authorization": f"Bearer {token(client, 'viewer@example.test', 'viewer-password-long')}"
    }
    admin_headers = {
        "Authorization": f"Bearer {token(client, 'admin@example.test', 'correct-horse-battery')}"
    }

    viewer_result = client.get("/admin/api/logs/sensitive-request", headers=viewer_headers)
    assert viewer_result.status_code == 200, viewer_result.text
    viewer_data = viewer_result.json()
    assert viewer_data["request_body"] is None
    assert viewer_data["response_body"] is None
    assert viewer_data["body_redacted"] is True
    assert "sensitive prompt text" not in viewer_result.text
    assert "sensitive completion text" not in viewer_result.text

    forbidden_reveal = client.get(
        "/admin/api/logs/sensitive-request?reveal=true", headers=viewer_headers
    )
    assert forbidden_reveal.status_code == 403
    assert "sensitive prompt text" not in forbidden_reveal.text

    default_admin_result = client.get(
        "/admin/api/logs/sensitive-request", headers=admin_headers
    )
    assert default_admin_result.status_code == 200, default_admin_result.text
    assert default_admin_result.json()["body_redacted"] is True
    assert default_admin_result.json()["request_body"] is None

    revealed = client.get(
        "/admin/api/logs/sensitive-request?reveal=true", headers=admin_headers
    )
    assert revealed.status_code == 200, revealed.text
    assert revealed.json()["request_body"] == stored_request
    assert revealed.json()["response_body"] == stored_response
    assert revealed.json()["body_redacted"] is False


def test_log_detail_marks_unstored_bodies_as_not_redacted(admin_api: Any) -> None:
    client, _, db = admin_api

    async def seed() -> None:
        async with db.session() as session:
            session.add(RequestLog(request_id="no-stored-body", model="model-a", status="success"))

    asyncio.run(seed())
    headers = {
        "Authorization": f"Bearer {token(client, 'viewer@example.test', 'viewer-password-long')}"
    }
    response = client.get("/admin/api/logs/no-stored-body", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["request_body"] is None
    assert response.json()["response_body"] is None
    assert response.json()["body_redacted"] is False


def test_usage_can_group_by_day(admin_api: Any) -> None:
    client, _, db = admin_api
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)

    async def seed() -> None:
        async with db.session() as session:
            session.add_all(
                [
                    UsageRollup(
                        bucket=now - timedelta(days=1),
                        model="model-a",
                        provider="stub",
                        request_count=2,
                        success_count=2,
                    ),
                    UsageRollup(
                        bucket=now,
                        model="model-a",
                        provider="stub",
                        request_count=3,
                        success_count=3,
                    ),
                ]
            )

    asyncio.run(seed())
    headers = {"Authorization": f"Bearer {token(client, 'admin@example.test', 'correct-horse-battery')}"}
    response = client.get("/admin/api/usage?group_by=day", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["group_by"] == "day"
    assert [row["requests"] for row in response.json()["rows"]] == [2, 3]


def test_cache_entry_inspector_and_namespace_invalidation(admin_api: Any) -> None:
    client, _, _ = admin_api
    state = client.app.state.gateway
    redis = state.redis
    redis.hashes.update(
        {
            "aigw:cache:entry:namespace-a:item-a": {
                "namespace": "namespace-a",
                "created_at": str(time.time() - 30),
                "embedding": b"\xff\xfe",
                "response": json.dumps({"model": "model-a"}),
                "prompt": "auth token sk-abcdefgh1234",
                "hit_count": "4",
            },
            "aigw:cache:entry:namespace-b:item-b": {
                "namespace": "namespace-b",
                "created_at": str(time.time() - 60),
                "response": json.dumps({"model": "model-b"}),
            },
        }
    )
    headers = {"Authorization": f"Bearer {token(client, 'admin@example.test', 'correct-horse-battery')}"}
    listing = client.get("/admin/api/cache/entries?limit=1", headers=headers)
    assert listing.status_code == 200, listing.text
    assert listing.json()["total"] == 2
    entry = listing.json()["items"][0]
    assert entry["model"] == "model-a"
    assert entry["namespace"] == "namespace-a"
    assert entry["hit_count"] == 4
    assert entry["age_seconds"] >= 30
    assert entry["ttl_remaining_seconds"] == 45
    assert "[REDACTED]" in entry["cached_prompt"]
    assert "sk-abcdefgh1234" not in entry["cached_prompt"]

    invalidated = client.post(
        "/admin/api/cache/invalidate",
        headers=headers,
        json={"namespace": "namespace-a"},
    )
    assert invalidated.status_code == 200, invalidated.text
    assert invalidated.json()["invalidated"] == 1
    assert "aigw:cache:entry:namespace-a:item-a" not in redis.hashes
    assert "aigw:cache:entry:namespace-b:item-b" in redis.hashes


def test_cache_stats_report_threshold_and_explicit_unavailable_metrics(admin_api: Any) -> None:
    client, _, _ = admin_api
    headers = {"Authorization": f"Bearer {token(client, 'admin@example.test', 'correct-horse-battery')}"}
    response = client.get("/admin/api/cache/stats", headers=headers)
    assert response.status_code == 200, response.text
    stats = response.json()
    assert stats["similarity_threshold"] == client.app.state.gateway.settings.cache_similarity_threshold
    assert stats["index_size_bytes"] == 2 * 1024 * 1024
    assert stats["estimated_latency_saved_ms"] is None


def test_dashboard_summary_includes_previous_window_and_failovers(admin_api: Any) -> None:
    client, _, db = admin_api
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)

    async def seed() -> None:
        async with db.session() as session:
            session.add_all(
                [
                    UsageRollup(
                        bucket=now - timedelta(hours=2),
                        model="current-model",
                        provider="stub",
                        request_count=4,
                        success_count=3,
                        error_count=1,
                        cache_hit_count=2,
                        fallback_count=3,
                        total_tokens=40,
                    ),
                    UsageRollup(
                        bucket=now - timedelta(hours=30),
                        model="previous-model",
                        provider="stub",
                        request_count=2,
                        success_count=1,
                        error_count=1,
                        cache_hit_count=1,
                        fallback_count=1,
                        total_tokens=20,
                    ),
                ]
            )

    asyncio.run(seed())
    headers = {"Authorization": f"Bearer {token(client, 'admin@example.test', 'correct-horse-battery')}"}
    response = client.get("/admin/api/dashboard/summary?window=24h", headers=headers)
    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["requests"] == 4
    assert summary["fallback_count"] == 3
    assert summary["previous_requests"] == 2
    assert summary["previous_success_rate"] == pytest.approx(0.5)
    assert summary["previous_cache_hit_ratio"] == pytest.approx(0.5)
    assert summary["active_requests"] >= 0


def test_admin_user_crud_and_password_hash_is_never_returned(admin_api: Any) -> None:
    client, _, _ = admin_api
    admin = token(client, "admin@example.test", "correct-horse-battery")
    headers = {"Authorization": f"Bearer {admin}"}
    created = client.post(
        "/admin/api/users",
        headers=headers,
        json={
            "email": "new.user@example.test",
            "password": "a-very-long-password",
            "role": "viewer",
        },
    )
    assert created.status_code == 201, created.text
    user = created.json()
    assert user["email"] == "new.user@example.test"
    assert user["role"] == "viewer"
    assert "password_hash" not in user

    listing = client.get("/admin/api/users?limit=1", headers=headers)
    assert listing.status_code == 200
    assert listing.json()["total"] == 3
    assert listing.json()["limit"] == 1
    assert all("password_hash" not in row for row in listing.json()["items"])

    updated = client.patch(
        f"/admin/api/users/{user['id']}", headers=headers, json={"role": "admin"}
    )
    assert updated.status_code == 200
    assert updated.json()["role"] == "admin"
    assert client.delete(f"/admin/api/users/{user['id']}", headers=headers).status_code == 200


def test_admin_cannot_demote_or_deactivate_own_account(admin_api: Any) -> None:
    client, _, _ = admin_api
    admin = token(client, "admin@example.test", "correct-horse-battery")
    headers = {"Authorization": f"Bearer {admin}"}
    current = client.get("/admin/api/auth/me", headers=headers).json()

    demote = client.patch(
        f"/admin/api/users/{current['id']}", headers=headers, json={"role": "viewer"}
    )
    assert demote.status_code == 409
    assert "own admin account" in demote.json()["detail"]

    deactivate = client.patch(
        f"/admin/api/users/{current['id']}", headers=headers, json={"is_active": False}
    )
    assert deactivate.status_code == 409
    assert client.delete(f"/admin/api/users/{current['id']}", headers=headers).status_code == 409


def test_last_active_admin_cannot_be_demoted_or_deleted(admin_api: Any) -> None:
    client, _, _ = admin_api
    admins = client.get("/admin/api/users", headers={"Authorization": "Bearer invalid"})
    assert admins.status_code == 401
    real_admin = token(client, "admin@example.test", "correct-horse-battery")
    users = client.get("/admin/api/users", headers={"Authorization": f"Bearer {real_admin}"})
    last_admin = next(user for user in users.json()["items"] if user["role"] == "admin")
    caller = AdminUser(
        id="separate-operator",
        email="operator@example.test",
        password_hash=hash_password("operator-password-long"),
        role="admin",
        is_active=True,
    )
    client.app.dependency_overrides[get_current_admin] = lambda: caller
    headers = {"Authorization": "Bearer test-override"}
    demote = client.patch(
        f"/admin/api/users/{last_admin['id']}", headers=headers, json={"role": "viewer"}
    )
    assert demote.status_code == 409
    assert "last active admin" in demote.json()["detail"]
    delete = client.delete(f"/admin/api/users/{last_admin['id']}", headers=headers)
    assert delete.status_code == 409
    assert "last active admin" in delete.json()["detail"]
    client.app.dependency_overrides.clear()


def test_provider_status_never_exposes_credentials_and_deployments_explain_routing(
    admin_api: Any,
) -> None:
    client, registry, _ = admin_api
    secret = "super-secret-provider-token"
    registry.deployments = [
        Deployment(
            id="stub-deployment",
            model_name="stub-model",
            provider="stub",
            provider_model="vendor-model",
            api_key=secret,
            priority=2,
            weight=7,
            tags=["fallback", "prod"],
        )
    ]
    admin = token(client, "admin@example.test", "correct-horse-battery")
    headers = {"Authorization": f"Bearer {admin}"}
    providers = client.get("/admin/api/providers/status", headers=headers)
    assert providers.status_code == 200, providers.text
    provider = providers.json()["items"][0]
    assert provider == {
        "provider": "stub",
        "configured": True,
        "reachable": True,
        "health_state": "closed",
    }
    assert secret not in providers.text

    deployments = client.get("/admin/api/deployments", headers=headers)
    deployment = deployments.json()["items"][0]
    assert deployment["priority"] == 2
    assert deployment["weight"] == 7
    assert deployment["tags"] == ["fallback", "prod"]
    priority_schema = client.get("/openapi.json").json()["components"]["schemas"][
        "DeploymentResponse"
    ]
    assert "lower is preferred" in priority_schema["properties"]["priority"]["description"].lower()


def test_system_info_exposes_active_routing_strategy(admin_api: Any) -> None:
    client, _, _ = admin_api
    admin = token(client, "admin@example.test", "correct-horse-battery")
    response = client.get("/admin/api/system/info", headers={"Authorization": f"Bearer {admin}"})
    assert response.status_code == 200
    assert response.json()["active_routing_strategy"] == "priority"


def test_usage_response_contains_pagination_metadata(admin_api: Any) -> None:
    client, _, _ = admin_api
    admin = token(client, "admin@example.test", "correct-horse-battery")
    response = client.get(
        "/admin/api/usage?limit=1&offset=2", headers={"Authorization": f"Bearer {admin}"}
    )
    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert response.json()["limit"] == 1
    assert response.json()["offset"] == 2


def test_console_user_can_change_own_password(admin_api: Any) -> None:
    client, _, _ = admin_api
    viewer = token(client, "viewer@example.test", "viewer-password-long")
    response = client.post(
        "/admin/api/users/me/change-password",
        headers={"Authorization": f"Bearer {viewer}"},
        json={
            "current_password": "viewer-password-long",
            "new_password": "updated-viewer-password",
        },
    )
    assert response.status_code == 200, response.text
    assert token(client, "viewer@example.test", "updated-viewer-password")
    bad_current = client.post(
        "/admin/api/users/me/change-password",
        headers={"Authorization": f"Bearer {viewer}"},
        json={
            "current_password": "viewer-password-long",
            "new_password": "another-viewer-password",
        },
    )
    assert bad_current.status_code == 400


class PlaygroundPipeline:
    def __init__(self) -> None:
        self.seen_request: Any = None

    async def run(self, ctx: Any) -> ChatResponse:
        self.seen_request = ctx.request
        deployment = Deployment(
            id="playground-deployment",
            model_name=ctx.request.model,
            provider="stub",
            provider_model="stub-model",
        )
        ctx.routing = RoutingDecision(
            deployment=deployment,
            strategy=ctx.request.routing_strategy or "priority",
            reason="test routing decision",
            candidates_considered=2,
        )
        ctx.attempted = ["first-deployment", "playground-deployment"]
        ctx.attempt_count = 2
        ctx.fallback_used = True
        ctx.errors = [RuntimeError("first attempt failed")]
        ctx.guardrail_flagged = True
        ctx.guardrail_results = {"input": {"verdict": "pass"}, "output": {"verdict": "flag"}}
        ctx.cache_hit = True
        ctx.cache_similarity = 0.97
        ctx.time_to_first_token_ms = 18.5
        ctx.stage_timings = {"auth": 0.1, "execute": 22.0}
        response = ChatResponse(
            model=ctx.request.model,
            choices=[Choice(message=Message(role=Role.ASSISTANT, content="hello"))],
            usage=Usage(prompt_tokens=11, completion_tokens=4, total_tokens=15),
            provider="stub",
            deployment_id=deployment.id,
            cache_hit=True,
            cache_similarity=0.97,
            latency_ms=25.0,
            cost_usd=0.00042,
            attempt_count=2,
            fallback_used=True,
        )
        ctx.response = response
        ctx.cost_usd = 0.00042
        return response


def test_playground_admin_auth_and_full_chat_parameter_passthrough(admin_api: Any) -> None:
    client, _, _ = admin_api
    gateway = client.app.state.gateway
    pipeline = PlaygroundPipeline()
    gateway.pipeline = pipeline
    gateway.require_pipeline = lambda: pipeline
    admin = token(client, "admin@example.test", "correct-horse-battery")
    headers = {"Authorization": f"Bearer {admin}"}
    payload = {
        "model": "stub-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 42,
        "temperature": 0.3,
        "top_p": 0.8,
        "stop": ["END"],
        "seed": 7,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.4,
        "n": 2,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": {"mode": "function", "function_name": "weather"},
        "parallel_tool_calls": True,
        "response_format": {"type": "json_object"},
        "user": "console-user",
        "metadata": {"source": "playground", "api_key": "spoofed-key"},
        "no_cache": True,
        "cache_ttl": 120,
        "fallbacks": ["backup-model"],
        "routing_strategy": "least-cost",
        "guardrail_policy": "strict",
        "tags": ["console", "debug"],
    }
    response = client.post("/admin/api/playground/chat", headers=headers, json=payload)
    assert response.status_code == 200, response.text

    seen = pipeline.seen_request
    assert seen is not None
    for name in (
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "n",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "user",
        "no_cache",
        "cache_ttl",
        "fallbacks",
        "routing_strategy",
        "guardrail_policy",
        "tags",
    ):
        actual = getattr(seen, name)
        if name in {"tools", "tool_choice"}:
            actual = (
                [item.model_dump(mode="json") for item in actual]
                if name == "tools"
                else actual.model_dump(mode="json")
            )
        assert actual == payload[name]
    assert seen.metadata["api_key"] == gateway.settings.master_key.get_secret_value()
    assert seen.metadata["source"] == "playground"

    trailer = response.json()
    assert trailer["request_id"].startswith("req_")
    assert trailer["deployment_id"] == "playground-deployment"
    assert trailer["provider"] == "stub"
    assert trailer["latency_ms"] == 25.0
    assert trailer["time_to_first_token_ms"] == 18.5
    assert trailer["cache_hit"] is True
    assert trailer["cache_similarity"] == 0.97
    assert trailer["retry_count"] == 1
    assert trailer["fallback_count"] == 1
    assert trailer["fallback_used"] is True
    assert trailer["guardrail_flagged"] is True
    assert trailer["guardrail_results"]["output"]["verdict"] == "flag"
    assert trailer["token_usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    assert trailer["estimated_cost_usd"] == 0.00042


def test_playground_rejects_viewer_and_anonymous_callers(admin_api: Any) -> None:
    client, _, _ = admin_api
    payload = {"model": "stub-model", "messages": [{"role": "user", "content": "hello"}]}
    assert client.post("/admin/api/playground/chat", json=payload).status_code == 401
    viewer = token(client, "viewer@example.test", "viewer-password-long")
    rejected = client.post(
        "/admin/api/playground/chat",
        headers={"Authorization": f"Bearer {viewer}"},
        json=payload,
    )
    assert rejected.status_code == 403
