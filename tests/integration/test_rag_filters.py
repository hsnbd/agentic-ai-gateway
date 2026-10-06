"""Collections can declare document metadata fields to filter searches by."""

from __future__ import annotations

from typing import Any

import redis
from fastapi.testclient import TestClient

from tests.integration.conftest import TEST_REDIS_URL

DOCS = [
    (
        "Holiday policy for engineers: 25 days.",
        {"department": "engineering", "regions": ["eu", "us"]},
    ),
    ("Holiday policy for sales: 22 days.", {"department": "sales"}),  # no regions
    ("Holiday policy for support: 24 days.", {"department": "support", "regions": ["eu"]}),
]


def _collection(client: TestClient, headers: dict[str, str]) -> str:
    created = client.post(
        "/v1/rag/collections",
        json={"name": "hr", "filterable_fields": ["department", "regions"]},
        headers=headers,
    )
    assert created.status_code == 201, created.text
    assert created.json()["filterable_fields"] == ["department", "regions"]
    collection_id: str = created.json()["id"]
    for index, (content, metadata) in enumerate(DOCS):
        response = client.post(
            f"/v1/rag/collections/{collection_id}/documents",
            json={"content": content, "source": f"doc{index}.md", "metadata": metadata},
            headers=headers,
        )
        assert response.status_code == 201, response.text
    return collection_id


def _search(client: TestClient, headers: dict[str, str], collection_id: str, **body: Any) -> Any:
    return client.post(
        "/v1/rag/search",
        json={"collection_id": collection_id, "query": "holiday policy", "top_k": 5, **body},
        headers=headers,
    )


def _sources(response: Any) -> list[str]:
    assert response.status_code == 200, response.text
    return sorted(item["source"] for item in response.json()["results"])


def test_searches_filter_by_declared_fields(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    collection_id = _collection(client, auth_headers)
    sales = _search(client, auth_headers, collection_id, filters={"department": "sales"})
    assert _sources(sales) == ["doc1.md"]
    eu = _search(
        client, auth_headers, collection_id, filters={"regions": "eu"}, search_mode="hybrid"
    )
    assert _sources(eu) == ["doc0.md", "doc2.md"]
    both = _search(
        client, auth_headers, collection_id, filters={"regions": "eu", "source": "doc2.md"}
    )
    assert _sources(both) == ["doc2.md"]


def test_undeclared_fields_and_bad_names_are_refused(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    collection_id = _collection(client, auth_headers)
    refused = _search(client, auth_headers, collection_id, filters={"team": "x"})
    assert refused.status_code == 400
    assert "can filter by: document_id, source, department, regions" in refused.text

    for bad in (["source"], ["has space"], [f"f{i}" for i in range(17)]):
        response = client.post(
            "/v1/rag/collections",
            json={"name": f"bad-{len(bad)}", "filterable_fields": bad},
            headers=auth_headers,
        )
        assert response.status_code == 400, response.text


def test_declared_fields_survive_a_rebuilt_index(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    collection_id = _collection(client, auth_headers)
    redis.Redis.from_url(TEST_REDIS_URL).flushall()
    assert (
        client.post(
            f"/v1/rag/collections/{collection_id}/reindex", headers=auth_headers
        ).status_code
        == 202
    )
    support = _search(client, auth_headers, collection_id, filters={"department": "support"})
    assert _sources(support) == ["doc2.md"]
