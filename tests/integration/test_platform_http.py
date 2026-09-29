"""The rest of the public surface: embeddings, legacy completions, token
counting, metrics, retries and streaming fallback, tracing, the CLI, and the
console mount."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.errors import ErrorCode, ProviderError
from tests.integration.conftest import FakeProvider, chat_body, metric_value, request_logs

ROOT = Path(__file__).resolve().parents[2]


class TestEmbeddings:
    def test_embeddings_round_trip(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.post(
            "/v1/embeddings",
            json={"model": "embed-model", "input": ["hello world", "goodbye"]},
            headers=auth_headers,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "list"
        assert [item["index"] for item in body["data"]] == [0, 1]
        assert len(body["data"][0]["embedding"]) == 32

    def test_embeddings_require_auth(self, client: TestClient) -> None:
        response = client.post("/v1/embeddings", json={"model": "embed-model", "input": "x"})
        assert response.status_code == 401

    def test_chat_only_model_has_no_embeddings(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/v1/embeddings", json={"model": "test-model", "input": "x"}, headers=auth_headers
        )
        assert response.status_code == 404

    def test_invalid_input_is_rejected(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/v1/embeddings", json={"model": "embed-model", "input": [1, 2]}, headers=auth_headers
        )
        assert response.status_code == 400


class TestDialectSurface:
    def test_model_detail(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.get("/v1/models/test-model", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["id"] == "test-model"
        assert client.get("/v1/models/nope", headers=auth_headers).status_code == 404

    def test_legacy_completions(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "Say hi"},
            headers=auth_headers,
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["text"] == "primary answer"

    def test_legacy_completions_stream(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        with client.stream(
            "POST",
            "/v1/completions",
            json={"model": "test-model", "prompt": "Say hi", "stream": True},
            headers=auth_headers,
        ) as response:
            text = "".join(response.iter_text())
        assert response.status_code == 200
        assert "data: [DONE]" in text

    def test_count_tokens(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.post(
            "/v1/messages/count_tokens",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hello there"}]},
            headers=auth_headers,
        )
        assert response.status_code == 200, response.text
        assert response.json()["input_tokens"] > 0

    def test_anthropic_streaming_events(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 50,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"x-api-key": auth_headers["Authorization"].split()[1]},
        ) as response:
            text = "".join(response.iter_text())
        assert response.status_code == 200
        for event in ("message_start", "content_block_delta", "message_stop"):
            assert f"event: {event}" in text
        # Anthropic SDKs dispatch on the payload's "type", not the SSE event line.
        blocks = [block for block in text.split("\n\n") if block.strip()]
        for block in blocks:
            event_line, data_line = block.split("\n", 1)
            assert json.loads(data_line.removeprefix("data: "))["type"] == event_line.removeprefix(
                "event: "
            )

    def test_stream_include_usage(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json=chat_body(stream=True, stream_options={"include_usage": True}),
            headers=auth_headers,
        ) as response:
            events = [
                json.loads(line[len("data: ") :])
                for line in response.iter_lines()
                if line.startswith("data: {")
            ]
        assert any(event.get("usage") for event in events)

    def test_malformed_json_is_a_400(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/v1/chat/completions",
            content=b"{not json",
            headers={**auth_headers, "content-type": "application/json"},
        )
        assert response.status_code == 400
        assert "error" in response.json()


class TestResilience:
    @pytest.mark.parametrize("extra_env", [{"MAX_RETRIES": "2"}])
    def test_max_retries_means_retries_after_the_first_attempt(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        primary: FakeProvider,
        backup: FakeProvider,
    ) -> None:
        primary.fail_times = 2
        primary.error = ProviderError(ErrorCode.PROVIDER_UNAVAILABLE, "blip")
        response = client.post("/v1/chat/completions", json=chat_body(), headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "primary answer"
        assert primary.calls == 3, "1 attempt + MAX_RETRIES=2 retries on the same deployment"
        assert backup.calls == 0

    @pytest.mark.parametrize("extra_env", [{"MAX_RETRIES": "0"}])
    def test_falls_back_when_retries_are_exhausted(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
        primary: FakeProvider,
        backup: FakeProvider,
    ) -> None:
        primary.fail_times = 99
        primary.error = ProviderError(ErrorCode.PROVIDER_UNAVAILABLE, "down")
        response = client.post("/v1/chat/completions", json=chat_body(), headers=auth_headers)
        assert response.json()["choices"][0]["message"]["content"] == "backup answer"
        [log] = request_logs(client, admin_headers)
        assert log["fallback_count"] == 1
        assert log["attempt_count"] == 2

    @pytest.mark.parametrize("extra_env", [{"MAX_RETRIES": "0"}])
    def test_log_and_cost_are_attributed_to_the_deployment_that_served(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
        primary: FakeProvider,
    ) -> None:
        primary.fail_times = 99
        primary.error = ProviderError(ErrorCode.PROVIDER_UNAVAILABLE, "down")
        response = client.post("/v1/chat/completions", json=chat_body(), headers=auth_headers)
        assert response.headers["X-Gateway-Deployment"] == "fake-backup/test-model"
        [log] = request_logs(client, admin_headers)
        assert log["deployment_id"] == "fake-backup/test-model"
        assert log["provider"] == "fake-backup"

    def test_non_retryable_error_is_not_retried(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        primary: FakeProvider,
    ) -> None:
        primary.fail_times = 99
        primary.error = ProviderError(ErrorCode.INVALID_REQUEST, "bad request")
        response = client.post("/v1/chat/completions", json=chat_body(), headers=auth_headers)
        assert response.status_code == 400
        assert primary.calls == 1

    def test_stream_falls_back_before_first_chunk(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        primary: FakeProvider,
    ) -> None:
        primary.fail_times = 99
        primary.error = ProviderError(ErrorCode.PROVIDER_UNAVAILABLE, "down")
        with client.stream(
            "POST", "/v1/chat/completions", json=chat_body(stream=True), headers=auth_headers
        ) as response:
            text = "".join(response.iter_text())
        assert response.status_code == 200
        assert "backup" in text

    def test_fallback_metric_is_recorded_once(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        primary: FakeProvider,
    ) -> None:
        before = metric_value(client, "aigw_fallbacks_total")
        primary.fail_times = 99
        primary.error = ProviderError(ErrorCode.PROVIDER_UNAVAILABLE, "down")
        client.post("/v1/chat/completions", json=chat_body(), headers=auth_headers)
        assert metric_value(client, "aigw_fallbacks_total") - before == 1


class TestMetrics:
    def test_metrics_exposes_gateway_series(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        before = metric_value(client, "aigw_requests_total", model="test-model", status="success")
        client.post("/v1/chat/completions", json=chat_body(), headers=auth_headers)
        after = metric_value(client, "aigw_requests_total", model="test-model", status="success")
        assert after - before == 1
        assert metric_value(client, "aigw_tokens_total", model="test-model") > 0

    def test_failed_requests_are_metered(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        primary: FakeProvider,
        backup: FakeProvider,
    ) -> None:
        before = metric_value(client, "aigw_requests_total", model="test-model", status="error")
        for provider in (primary, backup):
            provider.fail_times = 99
            provider.error = ProviderError(ErrorCode.PROVIDER_UNAVAILABLE, "down")
        client.post("/v1/chat/completions", json=chat_body(), headers=auth_headers)
        after = metric_value(client, "aigw_requests_total", model="test-model", status="error")
        assert after - before == 1

    @pytest.mark.parametrize("extra_env", [{"METRICS_ENABLED": "false"}])
    def test_metrics_can_be_disabled(self, client: TestClient) -> None:
        assert client.get("/metrics").status_code == 404


@pytest.mark.parametrize("extra_env", [{"TRACING_ENABLED": "true"}])
def test_tracing_can_be_enabled(client: TestClient, auth_headers: dict[str, str]) -> None:
    from app.observability import tracing

    assert tracing._enabled is True
    assert client.post("/v1/chat/completions", json=chat_body(), headers=auth_headers).is_success


class TestCli:
    def _run(self, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", "import sys; from app.cli import main; sys.exit(main())", *args],
            cwd=ROOT,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def test_init_db_create_admin_and_create_key(
        self, client: TestClient, gateway_env: dict[str, str]
    ) -> None:
        assert self._run(gateway_env, "init-db").returncode == 0

        admin = self._run(
            gateway_env,
            "create-admin",
            "--email",
            "cli@example.com",
            "--password",
            "cli-admin-password",
        )
        assert admin.returncode == 0, admin.stderr
        login = client.post(
            "/admin/api/auth/login",
            json={"email": "cli@example.com", "password": "cli-admin-password"},
        )
        assert login.status_code == 200

        created = self._run(gateway_env, "create-key", "--name", "cli", "--budget", "5")
        assert created.returncode == 0, created.stderr
        key = created.stdout.strip().splitlines()[-1]
        assert key.startswith("sk-aigw-")
        response = client.post(
            "/v1/chat/completions",
            json=chat_body(),
            headers={"Authorization": f"Bearer {key}"},
        )
        assert response.status_code == 200, response.text

    def test_routes_lists_the_api(self, gateway_env: dict[str, str]) -> None:
        result = self._run(gateway_env, "routes", "--json")
        assert result.returncode == 0, result.stderr
        paths = {row["path"] for row in json.loads(result.stdout)}
        assert {"/v1/chat/completions", "/v1/rag/collections", "/admin/api/auth/login"} <= paths


@pytest.mark.skipif(
    not (ROOT / "app" / "ui_static" / "index.html").exists(),
    reason="console not built (run `make ui-build`)",
)
class TestConsoleMount:
    def test_console_index_and_spa_fallback(self, client: TestClient) -> None:
        index = client.get("/ui/")
        assert index.status_code == 200
        assert "<div id=\"root\">" in index.text

        deep_link = client.get("/ui/logs")
        assert deep_link.status_code == 200
        assert deep_link.text == index.text

    def test_missing_asset_is_404_not_html(self, client: TestClient) -> None:
        assert client.get("/ui/assets/missing.js").status_code == 404
