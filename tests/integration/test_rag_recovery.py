"""RAG survives Redis losing its data: the index heals itself, and a reindex
rebuilds lost vectors from the chunk text kept in Postgres."""

from __future__ import annotations

from typing import Any

import pytest
import redis
from fastapi.testclient import TestClient

from tests.integration.conftest import TEST_REDIS_URL

PARIS = "Paris is the capital of France. The Eiffel Tower stands in Paris."
TOKYO = "Tokyo is the capital of Japan. Tokyo has the busiest railway station."


@pytest.fixture
def collection(client: TestClient, auth_headers: dict[str, str]) -> dict[str, Any]:
    created = client.post("/v1/rag/collections", json={"name": "geo"}, headers=auth_headers)
    assert created.status_code == 201, created.text
    for text, source in ((PARIS, "paris.md"), (TOKYO, "tokyo.md")):
        doc = client.post(
            f"/v1/rag/collections/{created.json()['id']}/documents",
            json={"content": text, "source": source},
            headers=auth_headers,
        )
        assert doc.status_code == 201, doc.text
    body: dict[str, Any] = created.json()
    return body


def _search(client: TestClient, headers: dict[str, str], collection_id: str) -> list[Any]:
    response = client.post(
        "/v1/rag/search",
        json={"collection_id": collection_id, "query": "Eiffel Tower Paris", "top_k": 2},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    results: list[Any] = response.json()["results"]
    return results


def _status(client: TestClient, headers: dict[str, str], collection_id: str) -> dict[str, Any]:
    response = client.get(f"/v1/rag/collections/{collection_id}/index", headers=headers)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_index_status_reports_a_healthy_collection(
    client: TestClient, auth_headers: dict[str, str], collection: dict[str, Any]
) -> None:
    status = _status(client, auth_headers, collection["id"])
    assert status["index_exists"] and status["in_sync"]
    assert status["indexed_chunks"] == status["stored_chunks"] == 2
    assert status["reindex"] is None


def test_a_dropped_index_is_recreated_and_backfilled_on_the_next_search(
    client: TestClient, auth_headers: dict[str, str], collection: dict[str, Any]
) -> None:
    redis.Redis.from_url(TEST_REDIS_URL).execute_command(
        "FT.DROPINDEX", f"aigw:rag:{collection['id']}:idx"
    )
    # The gateway still believes the index exists: the search itself must notice.
    results = _search(client, auth_headers, collection["id"])
    assert results and "Eiffel" in results[0]["text"]
    assert _status(client, auth_headers, collection["id"])["in_sync"]


def test_index_status_notices_a_missing_index(
    client: TestClient, auth_headers: dict[str, str], collection: dict[str, Any]
) -> None:
    redis.Redis.from_url(TEST_REDIS_URL).execute_command(
        "FT.DROPINDEX", f"aigw:rag:{collection['id']}:idx"
    )
    status = _status(client, auth_headers, collection["id"])
    assert status["index_exists"] is False and status["in_sync"] is False


def test_after_flushall_a_reindex_restores_search(
    client: TestClient, auth_headers: dict[str, str], collection: dict[str, Any]
) -> None:
    redis.Redis.from_url(TEST_REDIS_URL).flushall()
    lost = _status(client, auth_headers, collection["id"])
    assert lost["in_sync"] is False and lost["stored_chunks"] == 2
    assert _search(client, auth_headers, collection["id"]) == []  # heals, but vectors are gone

    started = client.post(f"/v1/rag/collections/{collection['id']}/reindex", headers=auth_headers)
    assert started.status_code == 202, started.text

    # TestClient runs background tasks before returning, so the job has finished.
    status = _status(client, auth_headers, collection["id"])
    assert status["in_sync"] and status["indexed_chunks"] == 2
    assert status["reindex"]["status"] == "completed" and status["reindex"]["chunks"] == 2
    results = _search(client, auth_headers, collection["id"])
    assert results and "Eiffel" in results[0]["text"] and results[0]["source"] == "paris.md"


def test_documents_can_be_deleted_without_the_index(
    client: TestClient, auth_headers: dict[str, str], collection: dict[str, Any]
) -> None:
    documents = client.get(
        f"/v1/rag/collections/{collection['id']}/documents", headers=auth_headers
    ).json()
    redis.Redis.from_url(TEST_REDIS_URL).execute_command(
        "FT.DROPINDEX", f"aigw:rag:{collection['id']}:idx"
    )
    paris = next(d for d in documents if d["source"] == "paris.md")
    deleted = client.delete(
        f"/v1/rag/collections/{collection['id']}/documents/{paris['id']}", headers=auth_headers
    )
    assert deleted.status_code == 200, deleted.text
    results = _search(client, auth_headers, collection["id"])
    assert [r["source"] for r in results] == ["tokyo.md"]


def test_reindex_permissions_and_concurrency(
    client: TestClient,
    auth_headers: dict[str, str],
    viewer_headers: dict[str, str],
    collection: dict[str, Any],
) -> None:
    path = f"/v1/rag/collections/{collection['id']}/reindex"
    assert client.post(path, headers=viewer_headers).status_code == 403
    assert (
        client.post("/v1/rag/collections/missing/reindex", headers=auth_headers).status_code == 404
    )

    state = client.app.state.gateway  # type: ignore[attr-defined]
    service = state.components["rag_service"]
    original = service.reindex

    async def never_finishes(collection_id: str) -> None:
        return None

    service.reindex = never_finishes
    try:
        assert client.post(path, headers=auth_headers).status_code == 202
        again = client.post(path, headers=auth_headers)
        assert again.status_code == 409
    finally:
        service.reindex = original


def test_a_failed_reindex_is_recorded_and_a_deleted_collection_is_ignored(
    client: TestClient, auth_headers: dict[str, str], collection: dict[str, Any]
) -> None:
    service = client.app.state.gateway.components["rag_service"]  # type: ignore[attr-defined]
    store = service.store
    original = store.reset

    async def broken(collection_id: str, dims: int) -> None:
        raise RuntimeError("redis exploded")

    store.reset = broken
    try:
        path = f"/v1/rag/collections/{collection['id']}"
        assert client.post(f"{path}/reindex", headers=auth_headers).status_code == 202
        failed = _status(client, auth_headers, collection["id"])["reindex"]
        assert failed["status"] == "failed" and "redis exploded" in failed["error"]

        async def delete_then_fail(collection_id: str, dims: int) -> None:
            await service.delete_collection(collection_id)
            raise RuntimeError("gone")

        store.reset = delete_then_fail
        assert client.post(f"{path}/reindex", headers=auth_headers).status_code == 202
        assert client.get(path, headers=auth_headers).status_code == 404
    finally:
        store.reset = original
