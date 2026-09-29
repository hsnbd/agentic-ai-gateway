"""The console's admin API, end to end through the real app and database."""

from __future__ import annotations

import csv
import io
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from app.core.errors import ErrorCode, ProviderError
from tests.integration.conftest import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    VIEWER_EMAIL,
    FakeProvider,
    chat_body,
    create_virtual_key,
    request_logs,
)


def _traffic(client: TestClient, headers: dict[str, str], n: int = 3) -> None:
    for i in range(n):
        response = client.post("/v1/chat/completions", json=chat_body(f"hi {i}"), headers=headers)
        assert response.status_code == 200, response.text


class TestConsoleAuth:
    def test_login_me_logout(self, client: TestClient, admin_headers: dict[str, str]) -> None:
        me = client.get("/admin/api/auth/me", headers=admin_headers)
        assert me.status_code == 200
        assert me.json()["email"] == ADMIN_EMAIL
        assert me.json()["role"] == "admin"
        assert client.post("/admin/api/auth/logout", headers=admin_headers).json() == {
            "success": True
        }

    @pytest.mark.parametrize(
        "credentials",
        [
            {"email": ADMIN_EMAIL, "password": "wrong-password"},
            {"email": "nobody@example.com", "password": ADMIN_PASSWORD},
        ],
    )
    def test_bad_credentials_are_rejected(
        self, client: TestClient, credentials: dict[str, str]
    ) -> None:
        assert client.post("/admin/api/auth/login", json=credentials).status_code == 401

    def test_email_is_case_insensitive(self, client: TestClient) -> None:
        response = client.post(
            "/admin/api/auth/login",
            json={"email": ADMIN_EMAIL.upper(), "password": ADMIN_PASSWORD},
        )
        assert response.status_code == 200

    def test_admin_api_requires_a_token(self, client: TestClient) -> None:
        assert client.get("/admin/api/auth/me").status_code == 401
        assert (
            client.get("/admin/api/auth/me", headers={"Authorization": "Bearer junk"}).status_code
            == 401
        )

    def test_change_password(self, client: TestClient, admin_headers: dict[str, str]) -> None:
        new_password = "a-brand-new-password"
        changed = client.post(
            "/admin/api/auth/change-password",
            json={"current_password": ADMIN_PASSWORD, "new_password": new_password},
            headers=admin_headers,
        )
        assert changed.status_code == 200, changed.text
        old = client.post(
            "/admin/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
        )
        assert old.status_code == 401
        new = client.post(
            "/admin/api/auth/login", json={"email": ADMIN_EMAIL, "password": new_password}
        )
        assert new.status_code == 200


class TestUsers:
    def test_user_lifecycle(self, client: TestClient, admin_headers: dict[str, str]) -> None:
        created = client.post(
            "/admin/api/users",
            json={"email": "ops@example.com", "password": "ops-password-123", "role": "viewer"},
            headers=admin_headers,
        )
        assert created.status_code == 201, created.text
        user_id = created.json()["id"]

        emails = [
            u["email"]
            for u in client.get("/admin/api/users", headers=admin_headers).json()["items"]
        ]
        assert "ops@example.com" in emails

        promoted = client.patch(
            f"/admin/api/users/{user_id}", json={"role": "admin"}, headers=admin_headers
        )
        assert promoted.json()["role"] == "admin"

        deactivated = client.patch(
            f"/admin/api/users/{user_id}", json={"is_active": False}, headers=admin_headers
        )
        assert deactivated.json()["is_active"] is False
        login = client.post(
            "/admin/api/auth/login",
            json={"email": "ops@example.com", "password": "ops-password-123"},
        )
        assert login.status_code == 401

        assert (
            client.delete(f"/admin/api/users/{user_id}", headers=admin_headers).status_code == 200
        )

    def test_short_passwords_are_rejected(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/admin/api/users",
            json={"email": "x@example.com", "password": "short", "role": "viewer"},
            headers=admin_headers,
        )
        assert response.status_code == 422

    def test_last_admin_cannot_be_removed(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        me = client.get("/admin/api/auth/me", headers=admin_headers).json()
        demote = client.patch(
            f"/admin/api/users/{me['id']}", json={"role": "viewer"}, headers=admin_headers
        )
        assert demote.status_code in {400, 409}
        delete = client.delete(f"/admin/api/users/{me['id']}", headers=admin_headers)
        assert delete.status_code in {400, 409}

    def test_viewer_changes_own_password(
        self, client: TestClient, viewer_headers: dict[str, str]
    ) -> None:
        from tests.integration.conftest import VIEWER_PASSWORD

        response = client.post(
            "/admin/api/users/me/change-password",
            json={"current_password": VIEWER_PASSWORD, "new_password": "viewer-new-password"},
            headers=viewer_headers,
        )
        assert response.status_code == 200, response.text
        login = client.post(
            "/admin/api/auth/login",
            json={"email": VIEWER_EMAIL, "password": "viewer-new-password"},
        )
        assert login.status_code == 200


class TestRoles:
    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("post", "/admin/api/keys", {"name": "x"}),
            ("post", "/admin/api/teams", {"name": "x"}),
            ("get", "/admin/api/users", None),
            ("post", "/admin/api/config/reload", None),
            ("post", "/admin/api/cache/invalidate", {"all_entries": True}),
            ("post", "/admin/api/playground/chat", {"model": "test-model", "messages": []}),
            ("get", "/admin/api/providers/status", None),
        ],
    )
    def test_viewer_cannot_use_admin_routes(
        self,
        client: TestClient,
        viewer_headers: dict[str, str],
        method: str,
        path: str,
        body: dict[str, object] | None,
    ) -> None:
        response = client.request(method, path, json=body, headers=viewer_headers)
        assert response.status_code == 403

    @pytest.mark.parametrize(
        "path",
        [
            "/admin/api/dashboard/summary",
            "/admin/api/dashboard/timeseries",
            "/admin/api/logs",
            "/admin/api/usage",
            "/admin/api/deployments",
            "/admin/api/models",
            "/admin/api/guardrails/policies",
            "/admin/api/guardrails/violations",
            "/admin/api/cache/stats",
            "/admin/api/system/info",
        ],
    )
    def test_viewer_can_read(
        self, client: TestClient, viewer_headers: dict[str, str], path: str
    ) -> None:
        assert client.get(path, headers=viewer_headers).status_code == 200


class TestKeysAndTeams:
    def test_key_lifecycle(self, client: TestClient, admin_headers: dict[str, str]) -> None:
        secret, key = create_virtual_key(client, admin_headers, rpm_limit=100)
        assert secret.startswith("sk-aigw-")

        listed = client.get("/admin/api/keys", headers=admin_headers).json()
        assert listed["total"] == 1
        assert "key" not in listed["items"][0] or listed["items"][0]["key"] is None

        fetched = client.get(f"/admin/api/keys/{key['id']}", headers=admin_headers).json()
        assert "key" not in fetched or fetched["key"] is None, "secret must never be re-shown"

        patched = client.patch(
            f"/admin/api/keys/{key['id']}", json={"name": "renamed"}, headers=admin_headers
        )
        assert patched.json()["name"] == "renamed"

        regenerated = client.post(
            f"/admin/api/keys/{key['id']}/regenerate", headers=admin_headers
        ).json()
        new_secret = regenerated["key"]
        assert new_secret and new_secret != secret
        old = client.post(
            "/v1/chat/completions",
            json=chat_body(),
            headers={"Authorization": f"Bearer {secret}"},
        )
        assert old.status_code == 401
        new = client.post(
            "/v1/chat/completions",
            json=chat_body(),
            headers={"Authorization": f"Bearer {new_secret}"},
        )
        assert new.status_code == 200

        assert (
            client.delete(f"/admin/api/keys/{key['id']}", headers=admin_headers).status_code == 200
        )
        gone = client.post(
            "/v1/chat/completions",
            json=chat_body(),
            headers={"Authorization": f"Bearer {new_secret}"},
        )
        assert gone.status_code == 401

    def test_disabling_a_key_revokes_it(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        secret, key = create_virtual_key(client, admin_headers)
        headers = {"Authorization": f"Bearer {secret}"}
        assert client.post("/v1/chat/completions", json=chat_body(), headers=headers).is_success
        client.patch(f"/admin/api/keys/{key['id']}", json={"enabled": False}, headers=admin_headers)
        assert (
            client.post("/v1/chat/completions", json=chat_body(), headers=headers).status_code
            == 401
        )

    def test_team_lifecycle_and_usage(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        team = client.post(
            "/admin/api/teams", json={"name": "research"}, headers=admin_headers
        ).json()
        secret, _ = create_virtual_key(client, admin_headers, team_id=team["id"])
        _traffic(client, {"Authorization": f"Bearer {secret}"}, n=2)

        usage = client.get(f"/admin/api/teams/{team['id']}/usage", headers=admin_headers).json()
        assert usage["requests"] == 2
        assert usage["cost_usd"] > 0

        assert client.get("/admin/api/teams", headers=admin_headers).json()["total"] == 1
        renamed = client.patch(
            f"/admin/api/teams/{team['id']}", json={"name": "r&d"}, headers=admin_headers
        )
        assert renamed.json()["name"] == "r&d"
        assert (
            client.delete(f"/admin/api/teams/{team['id']}", headers=admin_headers).status_code
            == 200
        )
        assert (
            client.get(f"/admin/api/teams/{team['id']}", headers=admin_headers).status_code == 404
        )


class TestObservabilityViews:
    def test_logs_filter_detail_and_export(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
    ) -> None:
        _traffic(client, auth_headers, n=3)
        client.post(
            "/v1/chat/completions", json=chat_body(model="other-model"), headers=auth_headers
        )

        assert len(request_logs(client, admin_headers)) == 4
        assert len(request_logs(client, admin_headers, model="other-model")) == 1
        assert len(request_logs(client, admin_headers, status="error")) == 0

        first = request_logs(client, admin_headers)[0]
        detail = client.get(f"/admin/api/logs/{first['request_id']}", headers=admin_headers)
        assert detail.status_code == 200
        assert detail.json()["request_id"] == first["request_id"]
        assert client.get("/admin/api/logs/missing", headers=admin_headers).status_code == 404

        export = client.get("/admin/api/logs/export", headers=admin_headers)
        assert export.status_code == 200
        assert export.headers["content-type"].startswith("text/csv")
        rows = list(csv.DictReader(io.StringIO(export.text)))
        assert len(rows) == 4

    def test_dashboard_usage_and_costs(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
    ) -> None:
        _traffic(client, auth_headers, n=3)

        summary = client.get("/admin/api/dashboard/summary", headers=admin_headers).json()
        assert summary["requests"] == 3
        assert summary["success_rate"] == pytest.approx(1.0)

        series = client.get(
            "/admin/api/dashboard/timeseries",
            params={"metric": "requests", "interval": "hour"},
            headers=admin_headers,
        ).json()
        assert sum(point["value"] for point in series["points"]) == 3

        usage = client.get(
            "/admin/api/usage", params={"group_by": "model"}, headers=admin_headers
        ).json()
        assert usage["rows"], usage

        costs = client.get("/admin/api/usage/costs", headers=admin_headers).json()
        assert costs["total_cost_usd"] == pytest.approx(0.06)


class TestOperations:
    def test_deployments_models_and_health_check(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        deployments = client.get("/admin/api/deployments", headers=admin_headers).json()["items"]
        assert {d["model"] for d in deployments} >= {"test-model", "other-model"}

        models = client.get("/admin/api/models", headers=admin_headers).json()["items"]
        assert "test-model" in {m.get("name") or m.get("model") or m.get("id") for m in models}

        # Generated ids look like "fake/fake-1" and "fake/fake-1#2": both the
        # slash and the hash must survive the round trip once URL-encoded.
        for deployment in deployments:
            assert "/" in deployment["id"]
            check = client.post(
                f"/admin/api/deployments/{quote(deployment['id'], safe='')}/health-check",
                headers=admin_headers,
            )
            assert check.status_code == 200, check.text
            assert check.json()["deployment_id"] == deployment["id"]

        missing = client.post(
            "/admin/api/deployments/nope%2Fnope/health-check", headers=admin_headers
        )
        assert missing.status_code == 404

        status = client.get("/admin/api/providers/status", headers=admin_headers)
        assert status.status_code == 200

    def test_config_reload(self, client: TestClient, admin_headers: dict[str, str]) -> None:
        response = client.post("/admin/api/config/reload", headers=admin_headers)
        assert response.status_code == 200, response.text
        assert "test-model" in response.json()["models"]

    def test_system_info_reports_features(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        info = client.get("/admin/api/system/info", headers=admin_headers).json()
        assert info["database_connected"] is True
        assert info["redis_connected"] is True
        assert info["features"]["rag"] is True
        assert info["features"]["mcp"] is True
        assert info["features"]["guardrails"] is True

    def test_guardrail_policies_are_listed(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        policies = client.get("/admin/api/guardrails/policies", headers=admin_headers).json()
        assert {"default", "strict"} <= {p["name"] for p in policies["items"]}

    def test_playground_chat(
        self, client: TestClient, admin_headers: dict[str, str], primary: FakeProvider
    ) -> None:
        response = client.post(
            "/admin/api/playground/chat",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
            headers=admin_headers,
        )
        assert response.status_code == 200, response.text
        assert "primary answer" in response.text
        assert primary.calls == 1

    def test_playground_streams_with_metadata_trailer(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        with client.stream(
            "POST",
            "/admin/api/playground/chat",
            json={
                "model": "test-model",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=admin_headers,
        ) as response:
            text = "".join(response.iter_text())
        assert response.status_code == 200
        assert "primary " in text
        assert "event: metadata" in text
        assert text.rstrip().endswith("data: [DONE]")

    def test_playground_stream_failure_is_reported_in_band(
        self,
        client: TestClient,
        admin_headers: dict[str, str],
        primary: FakeProvider,
        backup: FakeProvider,
    ) -> None:
        for provider in (primary, backup):
            provider.fail_times = 99
            provider.error = ProviderError(ErrorCode.PROVIDER_UNAVAILABLE, "down")
        with client.stream(
            "POST",
            "/admin/api/playground/chat",
            json={
                "model": "test-model",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=admin_headers,
        ) as response:
            text = "".join(response.iter_text())
        assert "event: error" in text
        assert "all_providers_failed" in text
        assert text.rstrip().endswith("data: [DONE]")


@pytest.mark.parametrize("extra_env", [{"CACHE_ENABLED": "true"}])
class TestCacheAdmin:
    def test_stats_entries_and_invalidate(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
    ) -> None:
        body = chat_body("cache me please", temperature=0)
        client.post("/v1/chat/completions", json=body, headers=auth_headers)
        client.post("/v1/chat/completions", json=body, headers=auth_headers)

        stats = client.get("/admin/api/cache/stats", headers=admin_headers).json()
        assert stats["enabled"] is True
        assert stats["available"] is True
        assert stats["hits"] == 1
        assert stats["entries"] == 1

        entries = client.get("/admin/api/cache/entries", headers=admin_headers).json()
        assert entries["total"] == 1

        invalidated = client.post(
            "/admin/api/cache/invalidate", json={"all_entries": True}, headers=admin_headers
        ).json()
        assert invalidated["invalidated"] == 1
        assert client.get("/admin/api/cache/stats", headers=admin_headers).json()["entries"] == 0
