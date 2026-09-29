"""HTTP edge cases on the real app: malformed bodies, stream failures, legacy
completions, embeddings, RAG ingestion variants, readiness, and console tokens
on data-plane routes."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import jwt
from fastapi.testclient import TestClient

from app.core.errors import ErrorCode, ProviderError
from app.core.schemas import StreamChunk
from tests.integration.conftest import (
    JWT_SECRET,
    VIEWER_EMAIL,
    FakeProvider,
    chat_body,
)


def _stream(chunks: list[StreamChunk], error: Exception | None = None) -> Any:
    def stream(request: Any, deployment: Any) -> AsyncIterator[StreamChunk]:
        async def gen() -> AsyncIterator[StreamChunk]:
            for chunk in chunks:
                yield chunk
            if error is not None:
                raise error

        return gen()

    return stream


def _sse(text: str) -> list[str]:
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


class TestChatBodies:
    def test_non_object_body_is_rejected(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post("/v1/chat/completions", json=[1, 2], headers=auth_headers)
        assert response.status_code == 400
        assert "must be an object" in response.json()["error"]["message"]

    def test_empty_upstream_stream_is_an_internal_error(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        primary: FakeProvider,
        backup: FakeProvider,
    ) -> None:
        primary.stream = _stream([])  # type: ignore[method-assign]
        backup.stream = _stream([])  # type: ignore[method-assign]
        # Nothing has been sent yet, so the failure is a normal error response.
        response = client.post(
            "/v1/chat/completions", json=chat_body(stream=True), headers=auth_headers
        )
        assert response.status_code == 500
        assert response.json()["error"]["message"] == "Provider returned an empty stream"


class TestLegacyCompletions:
    def test_prompt_list_is_joined(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": ["line one", "line two"]},
            headers=auth_headers,
        )
        assert response.status_code == 200, response.text
        assert response.json()["object"] == "text_completion"

    def test_invalid_prompt_and_invalid_body(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        bad_prompt = client.post(
            "/v1/completions", json={"model": "test-model", "prompt": 5}, headers=auth_headers
        )
        assert bad_prompt.status_code == 400
        bad_body = client.post("/v1/completions", content=b"{not json", headers=auth_headers)
        assert bad_body.status_code == 400

    def test_stream_error_before_first_chunk_is_plain_json(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/v1/completions",
            json={"model": "unknown-model", "prompt": "hi", "stream": True},
            headers=auth_headers,
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_stream_error_after_first_chunk_is_forwarded(
        self, client: TestClient, auth_headers: dict[str, str], primary: FakeProvider
    ) -> None:
        # Longer than the output guardrail's stream holdback, so text is released
        # to the client before the failure.
        primary.stream = _stream(  # type: ignore[method-assign]
            [StreamChunk(model="test-model", content="x" * 400)],
            ProviderError(ErrorCode.PROVIDER_ERROR, "upstream died"),
        )
        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hi", "stream": True},
            headers=auth_headers,
        )
        frames = _sse(response.text)
        assert json.loads(frames[0])["choices"][0]["text"].startswith("xxx")
        assert "error" in json.loads(frames[-2])
        assert frames[-1] == "[DONE]"


class TestEmbeddingsAndModels:
    def test_embedding_request_without_model(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post("/v1/embeddings", json={"input": "hi"}, headers=auth_headers)
        assert response.status_code == 400
        assert "Invalid embedding request" in response.json()["error"]["message"]

    def test_model_with_only_disabled_deployments_is_not_found(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        registry = client.app.state.gateway.registry  # type: ignore[attr-defined]
        for deployment in registry.deployments_for("other-model"):
            registry._deployments[deployment.id] = deployment.model_copy(update={"enabled": False})
        response = client.get("/v1/models/other-model", headers=auth_headers)
        assert response.status_code == 404
        listed = [m["id"] for m in client.get("/v1/models", headers=auth_headers).json()["data"]]
        assert "other-model" not in listed


class TestReadiness:
    def test_not_ready_without_deployments(self, client: TestClient) -> None:
        registry = client.app.state.gateway.registry  # type: ignore[attr-defined]
        saved = dict(registry._deployments)
        registry._deployments.clear()
        try:
            response = client.get("/readyz")
        finally:
            registry._deployments.update(saved)
        assert response.status_code == 503
        assert response.json()["checks"]["providers"] is False


class TestRagIngestionVariants:
    def _collection(self, client: TestClient, headers: dict[str, str]) -> str:
        response = client.post("/v1/rag/collections", json={"name": "edge"}, headers=headers)
        assert response.status_code == 201, response.text
        collection_id: str = response.json()["id"]
        return collection_id

    def test_multipart_validation(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        collection_id = self._collection(client, auth_headers)
        url = f"/v1/rag/collections/{collection_id}/documents"
        cases = [
            ({"files": {"note": (None, "no file here")}}, "requires a file field"),
            ({"files": {"file": ("doc.pdf", b"%PDF", "application/pdf")}}, "Only .txt and .md"),
            (
                {"files": {"file": ("a.md", b"# A", "text/markdown")}, "data": {"metadata": "{"}},
                "valid JSON",
            ),
            (
                {"files": {"file": ("a.md", b"# A", "text/markdown")}, "data": {"metadata": "[1]"}},
                "JSON object",
            ),
        ]
        for kwargs, message in cases:
            response = client.post(url, headers=auth_headers, **kwargs)
            assert response.status_code == 400, response.text
            assert message in response.json()["error"]["message"]

        ok = client.post(
            url,
            headers=auth_headers,
            files={"file": ("guide.md", b"# Guide\n\nSome text.", "text/markdown")},
            data={"metadata": json.dumps({"team": "docs"})},
        )
        assert ok.status_code == 201, ok.text
        assert ok.json()["content_type"] == "text/markdown"

        plain = client.post(
            url, content=b"raw", headers={**auth_headers, "content-type": "text/plain"}
        )
        assert plain.status_code == 400
        assert "application/json or multipart" in plain.json()["error"]["message"]

    def test_document_and_chunk_lookups(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        collection_id = self._collection(client, auth_headers)
        base = f"/v1/rag/collections/{collection_id}"
        missing = client.get(f"{base}/documents/nope", headers=auth_headers)
        assert missing.status_code == 404

        client.post(
            f"{base}/documents", json={"content": "Paris is in France."}, headers=auth_headers
        )
        chunks = client.get(f"{base}/chunks?include_embeddings=true", headers=auth_headers).json()[
            "items"
        ]
        assert chunks and chunks[0]["embedding"] is not None
        chunk = client.get(
            f"{base}/chunks/{chunks[0]['id']}?include_embedding=false", headers=auth_headers
        ).json()
        assert chunk["embedding"] is None

    def test_query_with_invalid_rag_options(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        collection_id = self._collection(client, auth_headers)
        response = client.post(
            "/v1/rag/query",
            json={"collection_id": collection_id, "request": chat_body(), "mode": "bogus"},
            headers=auth_headers,
        )
        assert response.status_code == 400
        assert "Invalid RAG options" in response.json()["error"]["message"]

    def test_get_document_by_id(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        collection_id = self._collection(client, auth_headers)
        base = f"/v1/rag/collections/{collection_id}/documents"
        document = client.post(base, json={"content": "Rome is in Italy."}, headers=auth_headers)
        fetched = client.get(f"{base}/{document.json()['id']}", headers=auth_headers)
        assert fetched.status_code == 200
        assert fetched.json()["id"] == document.json()["id"]


class TestAnthropicTokenCounting:
    def test_invalid_messages_are_rejected(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/v1/messages/count_tokens",
            json={"model": "test-model", "messages": "nope"},
            headers=auth_headers,
        )
        assert response.status_code == 400
        assert response.json()["type"] == "error"


class TestConsoleTokensOnDataPlane:
    def test_token_without_subject_is_rejected(self, client: TestClient) -> None:
        token = jwt.encode(
            {"type": "access", "exp": int(time.time()) + 60}, JWT_SECRET, algorithm="HS256"
        )
        response = client.get("/v1/rag/collections", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 401
        assert "subject" in response.json()["detail"]

    def test_deactivated_console_user_is_rejected(
        self,
        client: TestClient,
        admin_headers: dict[str, str],
        viewer_headers: dict[str, str],
    ) -> None:
        users = client.get("/admin/api/users", headers=admin_headers).json()["items"]
        viewer = next(user for user in users if user["email"] == VIEWER_EMAIL)
        assert client.get("/v1/rag/collections", headers=viewer_headers).status_code == 200
        client.patch(
            f"/admin/api/users/{viewer['id']}", json={"is_active": False}, headers=admin_headers
        )
        response = client.get("/v1/rag/collections", headers=viewer_headers)
        assert response.status_code == 401

    def test_inactive_user_with_an_unrevoked_token_is_rejected(
        self, client: TestClient, viewer_headers: dict[str, str]
    ) -> None:
        """Deactivation outside the API (e.g. in SQL) does not revoke tokens."""
        import asyncio

        from sqlalchemy import update
        from sqlalchemy.ext.asyncio import create_async_engine

        from app.db.models import AdminUser
        from tests.integration.conftest import TEST_DATABASE_URL

        async def deactivate() -> None:
            engine = create_async_engine(TEST_DATABASE_URL)
            try:
                async with engine.begin() as conn:
                    await conn.execute(
                        update(AdminUser)
                        .where(AdminUser.email == VIEWER_EMAIL)
                        .values(is_active=False)
                    )
            finally:
                await engine.dispose()

        asyncio.run(deactivate())
        response = client.get("/v1/rag/collections", headers=viewer_headers)
        assert response.status_code == 401
        assert "inactive" in response.json()["detail"]


class TestPostgresOnlyQueries:
    def test_usage_grouped_by_hour_uses_date_trunc(
        self, client: TestClient, auth_headers: dict[str, str], admin_headers: dict[str, str]
    ) -> None:
        assert client.post(
            "/v1/chat/completions", json=chat_body(), headers=auth_headers
        ).is_success
        rows = client.get("/admin/api/usage?group_by=hour", headers=admin_headers).json()["rows"]
        assert sum(row["requests"] for row in rows) == 1
        assert rows[0]["group"].endswith(":00:00+00:00")
