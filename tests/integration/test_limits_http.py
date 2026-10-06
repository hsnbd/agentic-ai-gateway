"""Key and deployment limits, and output guardrails on streams."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.integration.conftest import (
    FakeProvider,
    chat_body,
    create_virtual_key,
    metric_value,
)

SECRET = "sk-abcdefghijklmnopqrstuvwxyz123456"


def _bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


class TestKeyLimits:
    def test_tokens_per_minute(self, client: TestClient, admin_headers: dict[str, str]) -> None:
        before = metric_value(client, "aigw_rate_limit_hits_total", scope="key_tpm")
        secret, _ = create_virtual_key(client, admin_headers, tpm_limit=40)
        statuses = [
            client.post("/v1/chat/completions", json=chat_body(), headers=_bearer(secret))
            for _ in range(4)
        ]
        assert statuses[0].status_code == 200
        limited = next(r for r in statuses if r.status_code != 200)
        assert limited.status_code == 429
        assert limited.json()["error"]["code"] == "rate_limit_exceeded"
        assert 0 < int(limited.headers["Retry-After"]) <= 60
        assert metric_value(client, "aigw_rate_limit_hits_total", scope="key_tpm") > before

    def test_max_parallel_requests(
        self, client: TestClient, admin_headers: dict[str, str], primary: FakeProvider
    ) -> None:
        secret, _ = create_virtual_key(client, admin_headers, max_parallel_requests=1)
        primary.latency = 0.6

        def send(_: int) -> int:
            return client.post(
                "/v1/chat/completions", json=chat_body(), headers=_bearer(secret)
            ).status_code

        with ThreadPoolExecutor(max_workers=3) as pool:
            codes = sorted(pool.map(send, range(3)))
        assert codes[0] == 200
        assert 429 in codes

        # Slots are released when requests finish, including failed ones.
        primary.latency = 0
        assert send(0) == 200

    def test_allowed_routes(self, client: TestClient, admin_headers: dict[str, str]) -> None:
        secret, _ = create_virtual_key(client, admin_headers, allowed_routes=["/v1/embeddings"])
        chat = client.post("/v1/chat/completions", json=chat_body(), headers=_bearer(secret))
        assert chat.status_code == 403
        embed = client.post(
            "/v1/embeddings",
            json={"model": "embed-model", "input": "hello"},
            headers=_bearer(secret),
        )
        assert embed.status_code == 200, embed.text

    def test_limits_apply_to_an_old_cached_key_snapshot(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        """Changing a key's limits takes effect immediately (snapshot invalidated)."""
        secret, key = create_virtual_key(client, admin_headers)
        assert client.post(
            "/v1/chat/completions", json=chat_body(), headers=_bearer(secret)
        ).is_success
        client.patch(
            f"/admin/api/keys/{key['id']}",
            json={"allowed_routes": ["/v1/embeddings"]},
            headers=admin_headers,
        )
        blocked = client.post("/v1/chat/completions", json=chat_body(), headers=_bearer(secret))
        assert blocked.status_code == 403


DEPLOYMENT_LIMITS_YAML = """
model_list:
  - model_name: test-model
    params: {provider: fake, model: fake-1, api_key: unused}
    priority: 1
    rpm_limit: 1
  - model_name: test-model
    params: {provider: fake-backup, model: fake-2, api_key: unused}
    priority: 2
    rpm_limit: 1
  - model_name: tpm-model
    params: {provider: fake, model: fake-1, api_key: unused}
    priority: 1
    tpm_limit: 10
  - model_name: tpm-model
    params: {provider: fake-backup, model: fake-2, api_key: unused}
    priority: 2
  - model_name: embed-model
    params: {provider: fake, model: fake-embed, api_key: unused}
    capabilities: {chat: false, embeddings: true}
"""


class TestDeploymentLimits:
    @pytest.fixture
    def models_yaml(self) -> str:
        return DEPLOYMENT_LIMITS_YAML

    def test_saturated_deployment_spills_over_then_429(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        first = client.post("/v1/chat/completions", json=chat_body(), headers=auth_headers)
        assert first.headers["X-Gateway-Deployment"] == "fake/test-model"
        second = client.post("/v1/chat/completions", json=chat_body(), headers=auth_headers)
        assert second.status_code == 200
        assert second.headers["X-Gateway-Deployment"] == "fake-backup/test-model"
        third = client.post("/v1/chat/completions", json=chat_body(), headers=auth_headers)
        assert third.status_code == 429
        assert third.json()["error"]["code"] == "rate_limit_exceeded"

    def test_token_limit_spills_over(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        body = chat_body(model="tpm-model")
        first = client.post("/v1/chat/completions", json=body, headers=auth_headers)
        assert first.headers["X-Gateway-Deployment"] == "fake/tpm-model"
        # The fake reports 15 tokens, over the primary's 10-token budget.
        second = client.post("/v1/chat/completions", json=body, headers=auth_headers)
        assert second.headers["X-Gateway-Deployment"] == "fake-backup/tpm-model"


def _stream_text(client: TestClient, headers: dict[str, str], **extra: object) -> tuple[int, str]:
    with client.stream(
        "POST", "/v1/chat/completions", json=chat_body(stream=True, **extra), headers=headers
    ) as response:
        return response.status_code, "".join(response.iter_text())


class TestStreamingGuardrails:
    @pytest.mark.parametrize("chunk_size", [1, 4, 9])
    def test_secret_is_redacted_across_chunks(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
        primary: FakeProvider,
        chunk_size: int,
    ) -> None:
        primary.reply = f"Your key is {SECRET}, keep it safe."
        primary.chunk_size = chunk_size
        status, text = _stream_text(client, auth_headers)
        assert status == 200
        assert "sk-abcdef" not in text
        content = "".join(
            part.split('"content":"', 1)[1].split('"', 1)[0]
            for part in text.split("data: ")
            if '"content":"' in part
        )
        assert content == "Your key is [API_KEY_REDACTED], keep it safe."

        violations = client.get("/admin/api/guardrails/violations", headers=admin_headers).json()[
            "items"
        ]
        assert any(v["rule"] == "redact-leaked-api-keys" for v in violations)


BLOCKING_POLICY = """
policies:
  default:
    output:
      - name: block-forbidden
        type: denylist
        terms: [forbidden]
        action: block
"""


@pytest.fixture
def blocking_policy(tmp_path: Path) -> Iterator[str]:
    path = tmp_path / "guardrails.yaml"
    path.write_text(BLOCKING_POLICY)
    yield str(path)


class TestStreamingBlock:
    @pytest.fixture
    def extra_env(self, blocking_policy: str) -> dict[str, str]:
        return {"GUARDRAILS_CONFIG_PATH": blocking_policy}

    def test_blocked_output_is_never_streamed(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
        primary: FakeProvider,
    ) -> None:
        primary.reply = "This answer contains forbidden material."
        status, text = _stream_text(client, auth_headers)
        assert "forbidden material" not in text
        assert status == 422 or "guardrail_violation" in text
        violations = client.get("/admin/api/guardrails/violations", headers=admin_headers).json()[
            "items"
        ]
        assert any(v["rule"] == "block-forbidden" for v in violations)
