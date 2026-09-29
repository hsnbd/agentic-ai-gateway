"""Tag-based conditional routing and cache hit reporting."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.integration.conftest import FakeProvider, chat_body

TAGGED_MODELS_YAML = """
model_list:
  - model_name: test-model
    params: {provider: fake, model: fake-1, api_key: unused}
    priority: 1
    tags: [us]
    pricing: {input_per_mtok: 1.0, output_per_mtok: 1.0}
  - model_name: test-model
    params: {provider: fake-backup, model: fake-2, api_key: unused}
    priority: 2
    tags: [eu, gdpr]
    pricing: {input_per_mtok: 5.0, output_per_mtok: 5.0}
  - model_name: embed-model
    params: {provider: fake, model: fake-embed, api_key: unused}
    capabilities: {chat: false, embeddings: true}
"""


class TestTagRouting:
    @pytest.fixture
    def models_yaml(self) -> str:
        return TAGGED_MODELS_YAML

    @pytest.mark.parametrize(
        ("tags", "deployment"),
        [
            (["eu"], "fake-backup/test-model"),
            (["gdpr", "other"], "fake-backup/test-model"),
            (["us"], "fake/test-model"),
            (["unknown"], "fake/test-model"),  # no match: cheapest wins
        ],
    )
    def test_request_tags_steer_conditional_routing(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        tags: list[str],
        deployment: str,
    ) -> None:
        response = client.post(
            "/v1/chat/completions",
            json=chat_body(tags=tags, routing_strategy="conditional"),
            headers=auth_headers,
        )
        assert response.status_code == 200, response.text
        assert response.headers["X-Gateway-Deployment"] == deployment

    def test_tags_via_aigw_extension(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/v1/chat/completions",
            json=chat_body(aigw={"tags": ["eu"], "routing_strategy": "conditional"}),
            headers=auth_headers,
        )
        assert response.headers["X-Gateway-Deployment"] == "fake-backup/test-model"

    def test_other_strategies_ignore_tags(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/v1/chat/completions",
            json=chat_body(tags=["eu"], routing_strategy="priority"),
            headers=auth_headers,
        )
        assert response.headers["X-Gateway-Deployment"] == "fake/test-model"


@pytest.mark.parametrize("extra_env", [{"CACHE_ENABLED": "true"}])
class TestCacheHitReporting:
    def test_similarity_header_and_latency_saved(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
        primary: FakeProvider,
    ) -> None:
        primary.latency = 0.3
        body = chat_body("How far away is the moon?", temperature=0)
        miss = client.post("/v1/chat/completions", json=body, headers=auth_headers)
        assert "X-Gateway-Cache-Similarity" not in miss.headers
        hit = client.post("/v1/chat/completions", json=body, headers=auth_headers)
        assert hit.headers["X-Gateway-Cache"] == "hit"
        assert float(hit.headers["X-Gateway-Cache-Similarity"]) == pytest.approx(1.0, abs=1e-3)

        stats = client.get("/admin/api/cache/stats", headers=admin_headers).json()
        assert stats["estimated_latency_saved_ms"] > 200

    def test_streamed_cache_hit_has_the_similarity_header(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        body = chat_body("Stream me from cache", temperature=0)
        client.post("/v1/chat/completions", json=body, headers=auth_headers)
        with client.stream(
            "POST", "/v1/chat/completions", json={**body, "stream": True}, headers=auth_headers
        ) as response:
            "".join(response.iter_text())
        assert response.headers["X-Gateway-Cache"] == "hit"
        assert "X-Gateway-Cache-Similarity" in response.headers
