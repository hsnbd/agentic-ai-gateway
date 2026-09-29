"""Every request, however it ends, must be logged, costed, and charged.

Regression tests for accounting gaps found during verification: streamed and
failed requests were never logged, budgets were never charged, and semantic
cache hits were billed at full price with zero recorded savings.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.errors import ErrorCode, ProviderError
from tests.integration.conftest import (
    FakeProvider,
    chat_body,
    create_virtual_key,
    metric_value,
    request_logs,
)

# Deployment pricing in MODELS_YAML is 1000 / 2000 USD per MTok and the fake
# provider reports 10 prompt + 5 completion tokens: 0.01 + 0.01 USD.
COST_PER_REQUEST = 0.02


class TestRequestLog:
    def test_unary_request_is_logged_with_cost(
        self, client: TestClient, auth_headers: dict[str, str], admin_headers: dict[str, str]
    ) -> None:
        assert client.post(
            "/v1/chat/completions", json=chat_body(), headers=auth_headers
        ).is_success
        [log] = request_logs(client, admin_headers)
        assert log["status"] == "success"
        assert log["cost_usd"] == pytest.approx(COST_PER_REQUEST)

    def test_streamed_request_is_logged(
        self, client: TestClient, auth_headers: dict[str, str], admin_headers: dict[str, str]
    ) -> None:
        with client.stream(
            "POST", "/v1/chat/completions", json=chat_body(stream=True), headers=auth_headers
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())
        assert "data: [DONE]" in body

        [log] = request_logs(client, admin_headers)
        assert log["stream"] is True
        assert log["status"] == "success"
        assert log["total_tokens"] == 15
        assert log["cost_usd"] == pytest.approx(COST_PER_REQUEST)

    def test_failed_request_is_logged(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
        primary: FakeProvider,
        backup: FakeProvider,
    ) -> None:
        down = ProviderError(ErrorCode.PROVIDER_UNAVAILABLE, "down")
        primary.fail_times = backup.fail_times = 100
        primary.error = backup.error = down

        response = client.post("/v1/chat/completions", json=chat_body(), headers=auth_headers)
        assert response.status_code >= 500

        [log] = request_logs(client, admin_headers)
        assert log["status"] == "error"
        assert log["error_code"]
        assert log["cost_usd"] == 0

    def test_unauthenticated_requests_are_not_logged(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/v1/chat/completions", json=chat_body(), headers={"Authorization": "Bearer nope"}
        )
        assert response.status_code == 401
        assert request_logs(client, admin_headers) == []


class TestBudgets:
    def test_key_budget_is_charged_and_enforced(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        secret, key = create_virtual_key(client, admin_headers, max_budget_usd=0.03)
        headers = {"Authorization": f"Bearer {secret}"}

        for _ in range(2):
            ok = client.post("/v1/chat/completions", json=chat_body(), headers=headers)
            assert ok.status_code == 200, ok.text

        refreshed = client.get(f"/admin/api/keys/{key['id']}", headers=admin_headers).json()
        assert refreshed["spend_usd"] == pytest.approx(2 * COST_PER_REQUEST)

        blocked = client.post("/v1/chat/completions", json=chat_body(), headers=headers)
        assert blocked.status_code == 402
        assert blocked.json()["error"]["code"] == "budget_exceeded"

    def test_streamed_spend_counts_toward_budget(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        secret, _ = create_virtual_key(client, admin_headers, max_budget_usd=0.01)
        headers = {"Authorization": f"Bearer {secret}"}
        with client.stream(
            "POST", "/v1/chat/completions", json=chat_body(stream=True), headers=headers
        ) as response:
            "".join(response.iter_text())
        blocked = client.post("/v1/chat/completions", json=chat_body(), headers=headers)
        assert blocked.status_code == 402

    def test_team_budget_is_enforced(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        team = client.post(
            "/admin/api/teams", json={"name": "t1", "max_budget_usd": 0.01}, headers=admin_headers
        )
        assert team.status_code == 201, team.text
        secret, _ = create_virtual_key(client, admin_headers, team_id=team.json()["id"])
        headers = {"Authorization": f"Bearer {secret}"}
        assert client.post("/v1/chat/completions", json=chat_body(), headers=headers).is_success
        blocked = client.post("/v1/chat/completions", json=chat_body(), headers=headers)
        assert blocked.status_code == 402


@pytest.mark.parametrize("extra_env", [{"CACHE_ENABLED": "true"}])
class TestCacheAccounting:
    def test_cache_hit_is_free_and_records_savings(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
        primary: FakeProvider,
    ) -> None:
        saved_before = metric_value(client, "aigw_cache_cost_saved_usd_total")
        body = chat_body("What is the capital of France?", temperature=0)
        first = client.post("/v1/chat/completions", json=body, headers=auth_headers)
        second = client.post("/v1/chat/completions", json=body, headers=auth_headers)
        assert first.status_code == second.status_code == 200
        assert primary.calls == 1, "second request should be served from the cache"

        logs = request_logs(client, admin_headers)
        hit = next(log for log in logs if log["cache_hit"])
        miss = next(log for log in logs if not log["cache_hit"])
        assert miss["cost_usd"] == pytest.approx(COST_PER_REQUEST)
        assert hit["cost_usd"] == 0

        saved = metric_value(client, "aigw_cache_cost_saved_usd_total") - saved_before
        assert saved == pytest.approx(COST_PER_REQUEST)

    def test_cache_lookups_are_counted_once(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        before = metric_value(client, "aigw_cache_lookups_total")
        body = chat_body("Count me once", temperature=0)
        client.post("/v1/chat/completions", json=body, headers=auth_headers)
        client.post("/v1/chat/completions", json=body, headers=auth_headers)
        assert metric_value(client, "aigw_cache_lookups_total") - before == 2
