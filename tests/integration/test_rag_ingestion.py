"""RAG ingestion at scale: size limits, files, background jobs, replacement by
source, and recovery of ingestions interrupted by a restart."""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.support.documents import make_docx, make_pdf


@pytest.fixture
def extra_env() -> dict[str, str]:
    return {"RAG_MAX_DOCUMENT_BYTES": "20000", "RAG_BACKGROUND_INGEST_BYTES": "500"}


@pytest.fixture
def collection_id(client: TestClient, auth_headers: dict[str, str]) -> str:
    response = client.post("/v1/rag/collections", json={"name": "docs"}, headers=auth_headers)
    assert response.status_code == 201, response.text
    value: str = response.json()["id"]
    return value


def _post(client: TestClient, headers: dict[str, str], collection_id: str, **body: Any) -> Any:
    return client.post(f"/v1/rag/collections/{collection_id}/documents", json=body, headers=headers)


def _documents(client: TestClient, headers: dict[str, str], collection_id: str) -> list[Any]:
    response = client.get(f"/v1/rag/collections/{collection_id}/documents", headers=headers)
    items: list[Any] = response.json()
    return items


def _wait_for(
    client: TestClient, headers: dict[str, str], collection_id: str, document_id: str
) -> Any:
    deadline = time.monotonic() + 10
    while True:
        doc = client.get(
            f"/v1/rag/collections/{collection_id}/documents/{document_id}", headers=headers
        ).json()
        if doc["status"] != "processing" or time.monotonic() > deadline:
            return doc
        time.sleep(0.05)


def test_oversized_documents_and_uploads_are_refused(
    client: TestClient, auth_headers: dict[str, str], collection_id: str
) -> None:
    too_big = _post(client, auth_headers, collection_id, content="x" * 20001, source="big.md")
    assert too_big.status_code == 413
    upload = client.post(
        f"/v1/rag/collections/{collection_id}/documents",
        files={"file": ("huge.txt", b"y" * 80001, "text/plain")},
        headers=auth_headers,
    )
    assert upload.status_code == 413 and "Upload exceeds" in upload.text


def test_large_documents_ingest_in_the_background(
    client: TestClient, auth_headers: dict[str, str], collection_id: str
) -> None:
    content = "Shipping policy. " + "Parcels travel by road and rail across the region. " * 20
    accepted = _post(client, auth_headers, collection_id, content=content, source="ship.md")
    assert accepted.status_code == 202, accepted.text
    assert accepted.json()["status"] == "processing"

    again = _post(client, auth_headers, collection_id, content=content, source="ship.md")
    assert again.json()["id"] == accepted.json()["id"]  # same work, not a second job

    done = _wait_for(client, auth_headers, collection_id, accepted.json()["id"])
    assert done["status"] == "ready" and done["chunk_count"] >= 1

    kept = _post(
        client,
        auth_headers,
        collection_id,
        content=content + " v2",
        source="ship.md",
        replace_existing=False,
    )
    assert kept.status_code == 202
    _wait_for(client, auth_headers, collection_id, kept.json()["id"])
    assert len(_documents(client, auth_headers, collection_id)) == 2
    results = client.post(
        "/v1/rag/search",
        json={"collection_id": collection_id, "query": "parcels by rail"},
        headers=auth_headers,
    ).json()["results"]
    assert results and results[0]["source"] == "ship.md"


def test_a_failed_background_ingestion_records_why(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    created = client.post(
        "/v1/rag/collections",
        json={"name": "broken", "embedding_model": "no-such-embedder"},
        headers=auth_headers,
    ).json()
    accepted = _post(client, auth_headers, created["id"], content="z " * 300, source="z.md")
    assert accepted.status_code == 202
    failed = _wait_for(client, auth_headers, created["id"], accepted.json()["id"])
    assert failed["status"] == "failed" and failed["error_message"]

    retried = _post(client, auth_headers, created["id"], content="z " * 300, source="z.md")
    assert retried.status_code == 202 and retried.json()["status"] == "processing"
    _wait_for(client, auth_headers, created["id"], retried.json()["id"])


def test_reingesting_a_source_replaces_the_older_version(
    client: TestClient, auth_headers: dict[str, str], collection_id: str
) -> None:
    _post(client, auth_headers, collection_id, content="Refunds take 14 days.", source="refunds.md")
    _post(client, auth_headers, collection_id, content="Refunds take 30 days.", source="refunds.md")
    docs = _documents(client, auth_headers, collection_id)
    assert len(docs) == 1
    chunks = client.get(f"/v1/rag/collections/{collection_id}/chunks", headers=auth_headers).json()
    assert [item["text"] for item in chunks["items"]] == ["Refunds take 30 days."]

    _post(
        client,
        auth_headers,
        collection_id,
        content="Refunds take 45 days.",
        source="refunds.md",
        replace_existing=False,
    )
    assert len(_documents(client, auth_headers, collection_id)) == 2


def test_pdf_and_word_uploads_become_searchable(
    client: TestClient, auth_headers: dict[str, str], collection_id: str
) -> None:
    url = f"/v1/rag/collections/{collection_id}/documents"
    pdf = client.post(
        url,
        files={
            "file": (
                "codes.pdf",
                make_pdf("Error E-4471 means a postcode mismatch"),
                "application/pdf",
            )
        },
        headers=auth_headers,
    )
    assert pdf.status_code == 201, pdf.text
    assert pdf.json()["content_type"] == "application/pdf"
    word = client.post(
        url,
        files={"file": ("travel.docx", make_docx(["Meals are capped at 60 USD per day."]), "x")},
        data={"replace_existing": "false"},
        headers=auth_headers,
    )
    assert word.status_code == 201, word.text
    hits = client.post(
        "/v1/rag/search",
        json={
            "collection_id": collection_id,
            "query": "postcode mismatch E-4471",
            "search_mode": "hybrid",
        },
        headers=auth_headers,
    ).json()["results"]
    assert hits[0]["source"] == "codes.pdf"


def test_interrupted_ingestions_are_marked_failed_on_startup(
    client: TestClient, auth_headers: dict[str, str], collection_id: str
) -> None:
    service = client.app.state.gateway.components["rag_service"]  # type: ignore[attr-defined]

    async def interrupted() -> None:
        async def never(*args: Any) -> None:
            return None

        original = service._ingest_later
        service._ingest_later = never
        try:
            await service.ingest(collection_id, "w " * 300, source="w.md")
        finally:
            service._ingest_later = original

    client.portal.call(interrupted)  # type: ignore[union-attr]
    [doc] = _documents(client, auth_headers, collection_id)
    assert doc["status"] == "processing"

    assert client.portal.call(service.recover_interrupted) == 1  # type: ignore[union-attr]
    [doc] = _documents(client, auth_headers, collection_id)
    assert doc["status"] == "failed" and "restart" in doc["error_message"]
