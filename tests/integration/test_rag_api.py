"""RAG over HTTP: collections, ingestion, chunks, search, and query.

Runs against real Redis Stack vector indexes and real Postgres rows, with the
fake provider producing deterministic bag-of-words embeddings.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.integration.conftest import FakeProvider, chat_body, create_virtual_key

PARIS = "Paris is the capital of France. The Eiffel Tower stands in Paris."
TOKYO = "Tokyo is the capital of Japan. Tokyo has the busiest railway station."


def _collection(client: TestClient, headers: dict[str, str], **fields: Any) -> dict[str, Any]:
    response = client.post("/v1/rag/collections", json={"name": "geo", **fields}, headers=headers)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def _ingest(
    client: TestClient, headers: dict[str, str], collection_id: str, content: str, **fields: Any
) -> dict[str, Any]:
    response = client.post(
        f"/v1/rag/collections/{collection_id}/documents",
        json={"content": content, **fields},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


class TestCollections:
    def test_create_list_get_delete(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        created = _collection(client, auth_headers, description="places", chunk_size=200)
        assert created["embedding_model"] == "embed-model"
        assert created["chunk_size"] == 200
        assert created["document_count"] == 0

        listed = client.get("/v1/rag/collections", headers=auth_headers).json()
        assert [c["id"] for c in listed] == [created["id"]]

        fetched = client.get(f"/v1/rag/collections/{created['id']}", headers=auth_headers)
        assert fetched.json()["name"] == "geo"

        deleted = client.delete(f"/v1/rag/collections/{created['id']}", headers=auth_headers)
        assert deleted.json() == {"deleted": True}
        missing = client.get(f"/v1/rag/collections/{created['id']}", headers=auth_headers)
        assert missing.status_code == 404

    def test_validation_errors(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.post("/v1/rag/collections", json={"name": ""}, headers=auth_headers)
        assert response.status_code == 422


class TestDocuments:
    def test_ingest_json_document_creates_chunks(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        collection = _collection(client, auth_headers)
        document = _ingest(
            client, auth_headers, collection["id"], PARIS, title="Paris", source="wiki"
        )
        assert document["status"] == "ready"
        assert document["chunk_count"] >= 1
        assert document["ingested_at"] is not None

        docs = client.get(
            f"/v1/rag/collections/{collection['id']}/documents", headers=auth_headers
        ).json()
        assert [d["title"] for d in docs] == ["Paris"]

        refreshed = client.get(
            f"/v1/rag/collections/{collection['id']}", headers=auth_headers
        ).json()
        assert refreshed["document_count"] == 1
        assert refreshed["chunk_count"] == document["chunk_count"]

    def test_ingest_multipart_upload(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        collection = _collection(client, auth_headers)
        response = client.post(
            f"/v1/rag/collections/{collection['id']}/documents",
            files={"file": ("tokyo.txt", TOKYO.encode(), "text/plain")},
            headers=auth_headers,
        )
        assert response.status_code == 201, response.text
        assert response.json()["status"] == "ready"

    def test_reingesting_identical_content_is_idempotent(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        collection = _collection(client, auth_headers)
        first = _ingest(client, auth_headers, collection["id"], PARIS)
        second = _ingest(client, auth_headers, collection["id"], PARIS)
        assert first["content_hash"] == second["content_hash"]
        docs = client.get(
            f"/v1/rag/collections/{collection['id']}/documents", headers=auth_headers
        ).json()
        assert len(docs) == 1

    def test_chunks_are_inspectable(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        collection = _collection(client, auth_headers)
        document = _ingest(client, auth_headers, collection["id"], PARIS)
        chunks = client.get(
            f"/v1/rag/collections/{collection['id']}/chunks",
            params={"document_id": document["id"]},
            headers=auth_headers,
        ).json()
        assert chunks["total"] == document["chunk_count"]
        first = chunks["items"][0]
        assert "Paris" in first["text"]

        detail = client.get(
            f"/v1/rag/collections/{collection['id']}/chunks/{first['id']}", headers=auth_headers
        ).json()
        assert detail["embedding"]["dimensions"] == 32
        assert len(detail["embedding"]["preview"]) == 8

    def test_delete_one_and_all_documents(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        collection = _collection(client, auth_headers)
        paris = _ingest(client, auth_headers, collection["id"], PARIS)
        _ingest(client, auth_headers, collection["id"], TOKYO)
        base = f"/v1/rag/collections/{collection['id']}/documents"

        assert client.delete(f"{base}/{paris['id']}", headers=auth_headers).json() == {
            "deleted": True
        }
        assert client.get(f"{base}/{paris['id']}", headers=auth_headers).status_code == 404
        assert client.delete(base, headers=auth_headers).json() == {"deleted": 1}
        assert client.get(base, headers=auth_headers).json() == []


class TestRetrieval:
    def test_search_ranks_the_relevant_document_first(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        collection = _collection(client, auth_headers)
        _ingest(client, auth_headers, collection["id"], PARIS, source="paris.txt")
        _ingest(client, auth_headers, collection["id"], TOKYO, source="tokyo.txt")

        response = client.post(
            "/v1/rag/search",
            json={"collection_id": collection["id"], "query": "Tokyo railway", "top_k": 2},
            headers=auth_headers,
        )
        assert response.status_code == 200, response.text
        results = response.json()["results"]
        assert results, "search returned nothing"
        assert "Tokyo" in results[0]["text"]
        assert results[0]["score"] >= results[-1]["score"]

    def test_query_retrieves_context_and_answers(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        primary: FakeProvider,
    ) -> None:
        collection = _collection(client, auth_headers)
        _ingest(client, auth_headers, collection["id"], PARIS)

        response = client.post(
            "/v1/rag/query",
            json={
                "collection_id": collection["id"],
                "request": chat_body("What is the capital of France?"),
                "top_k": 1,
            },
            headers=auth_headers,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["response"]["choices"][0]["message"]["content"] == "primary answer"
        assert "Paris" in body["sources"][0]["text"]

        # The retrieved context reached the model.
        sent = primary.seen_requests[-1]
        assert any("Eiffel" in message.text() for message in sent.messages)

    def test_query_runs_with_the_calling_virtual_keys_permissions(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        secret, _ = create_virtual_key(client, admin_headers, allowed_models=["other-model"])
        headers = {"Authorization": f"Bearer {secret}"}
        collection = _collection(client, headers)
        _ingest(client, headers, collection["id"], PARIS)

        # The key may not use test-model, and rag/query must not bypass that.
        denied = client.post(
            "/v1/rag/query",
            json={"collection_id": collection["id"], "request": chat_body("France?")},
            headers=headers,
        )
        assert denied.status_code == 403

    def test_search_unknown_collection_is_404(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/v1/rag/search",
            json={"collection_id": "missing", "query": "x"},
            headers=auth_headers,
        )
        assert response.status_code == 404


class TestAuthorization:
    def test_viewer_can_read_but_not_write(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        viewer_headers: dict[str, str],
    ) -> None:
        collection = _collection(client, auth_headers)
        assert client.get("/v1/rag/collections", headers=viewer_headers).status_code == 200

        assert (
            client.post(
                "/v1/rag/collections", json={"name": "nope"}, headers=viewer_headers
            ).status_code
            == 403
        )
        assert (
            client.delete(
                f"/v1/rag/collections/{collection['id']}", headers=viewer_headers
            ).status_code
            == 403
        )
        assert (
            client.post(
                f"/v1/rag/collections/{collection['id']}/documents",
                json={"content": PARIS},
                headers=viewer_headers,
            ).status_code
            == 403
        )

    def test_admin_console_user_can_write(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        _collection(client, admin_headers)

    @pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer bogus"}])
    def test_anonymous_is_rejected(self, client: TestClient, headers: dict[str, str]) -> None:
        assert client.get("/v1/rag/collections", headers=headers).status_code == 401
