"""Optional LLM reranking of search results, through the real provider registry."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tests.integration.conftest import FakeProvider, metric_value

DOCS = ("alpha apples", "alpha bananas", "alpha cherries")


def _collection(client: TestClient, headers: dict[str, str], **metadata: Any) -> str:
    created = client.post(
        "/v1/rag/collections", json={"name": "fruit", "metadata": metadata}, headers=headers
    )
    collection_id: str = created.json()["id"]
    for index, content in enumerate(DOCS):
        client.post(
            f"/v1/rag/collections/{collection_id}/documents",
            json={"content": content, "source": f"{index}.md"},
            headers=headers,
        )
    return collection_id


def _sources(
    client: TestClient, headers: dict[str, str], collection_id: str, **body: Any
) -> list[str]:
    response = client.post(
        "/v1/rag/search",
        json={"collection_id": collection_id, "query": "alpha", "top_k": 3, **body},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return [item["source"] for item in response.json()["results"]]


def test_a_rerank_model_reorders_the_results(
    client: TestClient, auth_headers: dict[str, str], primary: FakeProvider
) -> None:
    collection_id = _collection(client, auth_headers)
    plain = _sources(client, auth_headers, collection_id)
    primary.reply = "[2, 1, 0]"
    reranked = _sources(client, auth_headers, collection_id, rerank_model="test-model")
    assert reranked == list(reversed(plain))
    assert metric_value(client, "aigw_rag_reranks_total", outcome="reranked") >= 1


def test_an_unavailable_rerank_model_keeps_the_retrieval_order(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    collection_id = _collection(client, auth_headers)
    plain = _sources(client, auth_headers, collection_id)
    before = metric_value(client, "aigw_rag_reranks_total", outcome="fallback")
    assert _sources(client, auth_headers, collection_id, rerank_model="no-such-model") == plain
    assert metric_value(client, "aigw_rag_reranks_total", outcome="fallback") == before + 1


def test_a_collection_can_rerank_by_default(
    client: TestClient, auth_headers: dict[str, str], primary: FakeProvider
) -> None:
    collection_id = _collection(client, auth_headers, rerank_model="test-model")
    primary.reply = "[1]"
    sources = _sources(client, auth_headers, collection_id, top_k=1)
    assert len(sources) == 1 and primary.seen_requests  # the model was asked
