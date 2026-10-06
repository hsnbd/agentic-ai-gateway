"""End-to-end tests through the real HTTP surface.

Every test here sends a real request to the real app and asserts on the real
response. If the pipeline is mis-wired — a stage constructed wrongly, a
dialect that forgets to pass the API key, a router that never reaches the
provider — these fail, and unit tests would not have noticed.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.integration.conftest import FakeProvider


class TestHealth:
    def test_liveness_needs_no_auth(self, client: TestClient) -> None:
        assert client.get("/healthz").status_code == 200

    def test_readiness_reports_subsystems(self, client: TestClient) -> None:
        response = client.get("/readyz")
        assert response.status_code == 200

    def test_openapi_schema_is_served(self, client: TestClient) -> None:
        """The console generates its client from this, so it must stay valid."""
        schema = client.get("/openapi.json")
        assert schema.status_code == 200
        assert "/v1/chat/completions" in schema.json()["paths"]


class TestOpenAIDialect:
    def test_chat_completion_round_trip(
        self, client: TestClient, auth_headers: dict[str, str], primary: FakeProvider
    ) -> None:
        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "primary answer"
        assert body["object"] == "chat.completion"
        assert body["usage"]["total_tokens"] == 15
        assert primary.calls == 1

    def test_alias_resolves_to_deployment(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """`gpt-4o` is aliased to test-model, so clients can keep their names."""
        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200, response.text

    def test_missing_key_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 401

    def test_bad_key_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-not-a-real-key"},
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 401

    def test_unknown_model_returns_structured_error(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code in (400, 404)
        assert "error" in response.json()

    def test_streaming_emits_sse_and_terminates(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

        assert "data: " in body
        assert body.rstrip().endswith("data: [DONE]")

        payloads = [
            json.loads(line[6:])
            for line in body.splitlines()
            if line.startswith("data: ") and line[6:].strip() != "[DONE]"
        ]
        assert payloads, "stream produced no chunks"
        assert payloads[0]["object"] == "chat.completion.chunk"
        text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in payloads)
        assert "primary" in text

    def test_models_endpoint_lists_catalogue(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.get("/v1/models", headers=auth_headers)
        assert response.status_code == 200
        ids = {entry["id"] for entry in response.json()["data"]}
        assert "test-model" in ids


class TestAnthropicDialect:
    def test_messages_round_trip(self, client: TestClient, primary: FakeProvider) -> None:
        """Anthropic SDK clients authenticate with x-api-key, not bearer."""
        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-integration-master-key"},
            json={
                "model": "test-model",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["content"][0]["text"] == "primary answer"

    def test_messages_requires_auth(self, client: TestClient) -> None:
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 401


class TestResilience:
    def test_retry_recovers_from_a_transient_failure(
        self, client: TestClient, auth_headers: dict[str, str], primary: FakeProvider
    ) -> None:
        """One flaky call should be invisible to the client."""
        from app.core.errors import ErrorCode, ProviderError

        primary.fail_times = 1
        primary.error = ProviderError(ErrorCode.PROVIDER_UNAVAILABLE, "temporary blip")

        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200, response.text
        assert primary.calls >= 2, "expected the executor to retry"

    def test_exhausted_provider_surfaces_a_clean_error(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        primary: FakeProvider,
        backup: FakeProvider,
    ) -> None:
        """When everything upstream is down the client still gets a structured
        gateway error, never a stack trace."""
        from app.core.errors import ErrorCode, ProviderError

        primary.fail_times = 99
        primary.error = ProviderError(ErrorCode.PROVIDER_UNAVAILABLE, "down")
        backup.fail_times = 99
        backup.error = ProviderError(ErrorCode.PROVIDER_UNAVAILABLE, "down")

        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code >= 500
        payload = response.json()
        assert "error" in payload
        assert payload["error"].get("type") or payload["error"].get("code")


class TestGuardrails:
    def test_prompt_is_processed_with_guardrails_active(
        self, client: TestClient, auth_headers: dict[str, str], primary: FakeProvider
    ) -> None:
        """A benign prompt passes through untouched."""
        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "what is the capital of France?"}],
            },
        )
        assert response.status_code == 200
        assert primary.calls == 1


class TestDegradedMode:
    def test_gateway_serves_traffic_without_redis(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Losing the cache must not lose the gateway. The fixture boots with
        caching disabled, so a successful request proves the degraded path."""
        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200


class TestDataPlaneAuth:
    """RAG and MCP mutate gateway state outside the pipeline, so they cannot
    rely on the pipeline's auth stage and are guarded at router mount."""

    UNAUTHENTICATED_ROUTES = (
        ("get", "/v1/rag/collections"),
        ("get", "/v1/mcp/servers"),
        ("get", "/v1/mcp/tools"),
    )

    @pytest.mark.parametrize(("method", "path"), UNAUTHENTICATED_ROUTES)
    def test_routes_reject_anonymous_callers(
        self, client: TestClient, method: str, path: str
    ) -> None:
        response = getattr(client, method)(path)
        assert response.status_code == 401

    @pytest.mark.parametrize(("method", "path"), UNAUTHENTICATED_ROUTES)
    def test_routes_reject_a_bogus_credential(
        self, client: TestClient, method: str, path: str
    ) -> None:
        response = getattr(client, method)(
            path, headers={"Authorization": "Bearer sk-not-a-real-key"}
        )
        assert response.status_code == 401

    @pytest.mark.parametrize(("method", "path"), UNAUTHENTICATED_ROUTES)
    def test_master_key_is_admitted(
        self, client: TestClient, auth_headers: dict[str, str], method: str, path: str
    ) -> None:
        """A valid credential must get past the guard. Anything other than 401
        means authentication succeeded and the route itself ran."""
        response = getattr(client, method)(path, headers=auth_headers)
        assert response.status_code != 401
