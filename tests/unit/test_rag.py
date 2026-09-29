"""Unit tests for the gateway-owned RAG subsystem."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import func, select

from app.api.rag import (
    EMBEDDING_PREVIEW_VALUES,
    ChunkResponse,
    CollectionResponse,
    DocumentResponse,
    RetrievedChunkResponse,
    _chunk_response,
)
from app.config.settings import Settings
from app.core.errors import InvalidRequestError
from app.core.schemas import ChatRequest, Message, Role
from app.db.models import RagChunk, RagCollection, RagDocument
from app.db.session import Database
from app.rag.chunking import chunk_text
from app.rag.ingest import ingest_document
from app.rag.retrieve import RetrievedChunk, augment_request, build_context, retrieve
from app.rag.service import MAX_CHUNK_PAGE_SIZE, RagService
from app.rag.store import RagVectorStore


class StubEmbedder:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [1.0, 0.0]
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vector[:] for _ in texts]


class InMemoryStore:
    def __init__(self, matches: list[tuple[str, float, dict[str, Any]]] | None = None) -> None:
        self.matches = matches or []
        self.upserts: list[list[dict[str, Any]]] = []
        self.deleted: list[str] = []

    async def upsert(self, collection_id: str, payloads: list[dict[str, Any]]) -> None:
        self.upserts.append(payloads)

    async def search(
        self,
        collection_id: str,
        vector: list[float],
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        return self.matches[:k]

    async def delete_document(self, collection_id: str, document_id: str) -> int:
        self.deleted.append(document_id)
        return 1


class FakeSearchIndex:
    def __init__(self, docs: list[Any]) -> None:
        self.docs = docs

    async def info(self) -> dict[str, Any]:
        return {}

    async def search(self, query: Any, query_params: dict[str, bytes] | None = None) -> Any:
        return SimpleNamespace(docs=self.docs)


class FakeRedis:
    def __init__(self, docs: list[Any]) -> None:
        self.index = FakeSearchIndex(docs)

    def ft(self, name: str) -> FakeSearchIndex:
        return self.index


class FailingVectorStore(InMemoryStore):
    async def upsert(self, collection_id: str, payloads: list[dict[str, Any]]) -> None:
        raise RuntimeError("redis write failed")


def test_recursive_splitter_keeps_boundaries_and_overlaps() -> None:
    text = "First sentence is here. Second sentence is here. Third sentence is here."
    chunks = chunk_text(text, chunk_size=34, overlap=8)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 34 for chunk in chunks)
    assert any(
        chunks[index].end_char > chunks[index + 1].start_char for index in range(len(chunks) - 1)
    )
    assert all(chunk.text.rstrip().endswith((".", "here")) for chunk in chunks)
    assert all(chunk.index == index for index, chunk in enumerate(chunks))


def test_markdown_chunks_retain_heading_trail() -> None:
    text = "# Guide\n\nIntro.\n\n## Auth\n\n" + ("Authentication details. " * 5)
    chunks = chunk_text(text, chunk_size=45, overlap=5, strategy="markdown")
    auth_chunks = [chunk for chunk in chunks if "Authentication details" in chunk.text]
    assert auth_chunks
    assert all("# Guide > ## Auth" in chunk.text for chunk in auth_chunks)


def test_invalid_overlap_raises() -> None:
    with pytest.raises(InvalidRequestError):
        chunk_text("text", chunk_size=10, overlap=10)


def test_empty_short_giant_word_and_unicode_text() -> None:
    assert chunk_text("") == []
    short = chunk_text("short text", chunk_size=1000)
    assert len(short) == 1 and short[0].text == "short text"
    giant = chunk_text("界" * 30, chunk_size=10, overlap=3)
    assert all(len(chunk.text) <= 10 for chunk in giant)
    assert any(
        giant[index].end_char > giant[index + 1].start_char for index in range(len(giant) - 1)
    )
    assert (
        "".join(chunk.text for chunk in chunk_text("café résumé", chunk_size=1000)) == "café résumé"
    )


@pytest.mark.asyncio
async def test_store_converts_cosine_distance_to_similarity() -> None:
    row = SimpleNamespace(
        chunk_id=b"chunk-1",
        document_id=b"doc-1",
        source=b"guide.md",
        content=b"retrieved text",
        chunk_index=0,
        metadata=b'{"section":"intro"}',
        vector=bytes(8),
        vector_distance=b"0.25",
    )
    store = RagVectorStore(FakeRedis([row]), embedder_dims=2)
    results = await store.search("collection", [1.0, 0.0], 1)
    assert results[0][0] == "chunk-1"
    assert results[0][1] == pytest.approx(0.75)
    assert results[0][2]["section"] == "intro"


@pytest.mark.asyncio
async def test_mmr_reduces_near_duplicate_results() -> None:
    matches = [
        ("a", 0.95, {"content": "first", "_vector": [1.0, 0.0]}),
        ("b", 0.94, {"content": "near duplicate", "_vector": [0.999, 0.01]}),
        ("c", 0.70, {"content": "different", "_vector": [0.0, 1.0]}),
    ]
    store = InMemoryStore(matches)
    embedder = StubEmbedder()
    plain = await retrieve(store, embedder, collection_id="c", query="q", top_k=2)
    diversified = await retrieve(
        store, embedder, collection_id="c", query="q", top_k=2, diversity=1.0
    )
    assert [item.id for item in plain] == ["a", "b"]
    assert [item.id for item in diversified] == ["a", "c"]


def test_build_context_obeys_token_budget() -> None:
    chunks = [
        RetrievedChunk(id=str(index), text="x" * 300, score=0.8, source="doc.txt")
        for index in range(3)
    ]
    context = build_context(chunks, max_tokens=30)
    from app.accounting.tokens import count_tokens

    assert count_tokens(context, "gpt-4o") <= 30
    assert "Source: doc.txt" in context


def test_augment_request_does_not_mutate_original() -> None:
    request = ChatRequest(model="gpt-test", messages=[Message(role=Role.USER, content="Question?")])
    augmented = augment_request(request, "a source snippet")
    assert len(request.messages) == 1
    assert len(augmented.messages) == 2
    assert augmented.messages[0].role == Role.SYSTEM
    assert "a source snippet" in augmented.system_prompt()


@pytest.mark.asyncio
async def test_duplicate_ingest_is_idempotent() -> None:
    db = Database(Settings(database_url="sqlite+aiosqlite:///:memory:"))
    await db.startup()
    await db.create_all()
    store = InMemoryStore()
    embedder = StubEmbedder()
    try:
        async with db.session() as session:
            session.add(
                RagCollection(
                    id="collection-1",
                    name="test",
                    embedding_model="stub",
                    embedding_dimensions=2,
                    chunk_size=40,
                    chunk_overlap=5,
                )
            )
        first = await ingest_document(
            db,
            store,
            embedder,
            collection_id="collection-1",
            content="Same text.",
            source="file.txt",
        )
        second = await ingest_document(
            db,
            store,
            embedder,
            collection_id="collection-1",
            content="Same text.",
            source="file.txt",
        )
        assert first.id == second.id
        assert len(store.upserts) == 1
        assert len(embedder.calls) == 1
        async with db.session() as session:
            documents = await session.scalar(select(func.count()).select_from(RagDocument))
            chunks = await session.scalar(select(func.count()).select_from(RagChunk))
        assert documents == chunks == 1
    finally:
        await db.shutdown()


@pytest.mark.asyncio
async def test_failed_vector_write_rolls_back_postgres_rows() -> None:
    db = Database(Settings(database_url="sqlite+aiosqlite:///:memory:"))
    await db.startup()
    await db.create_all()
    try:
        async with db.session() as session:
            session.add(
                RagCollection(
                    id="collection-1",
                    name="test",
                    embedding_model="stub",
                    embedding_dimensions=2,
                    chunk_size=40,
                    chunk_overlap=5,
                )
            )
        with pytest.raises(RuntimeError, match="redis write failed"):
            await ingest_document(
                db,
                FailingVectorStore(),
                StubEmbedder(),
                collection_id="collection-1",
                content="Must roll back.",
                source="file.txt",
            )
        # The document row survives only as a failure marker; no chunks may leak
        # through, since Redis never accepted their vectors.
        async with db.session() as session:
            chunks = await session.scalar(select(func.count()).select_from(RagChunk))
            document = await session.scalar(select(RagDocument))
        assert chunks == 0
        assert document is not None
        assert document.status == "failed"
        assert document.chunk_count == 0
    finally:
        await db.shutdown()


def test_retrieved_chunks_convert_to_api_response_models() -> None:
    chunks = [RetrievedChunk(id="chunk", text="passage", score=0.7, source="guide.md")]
    converted = _chunk_response(chunks)
    assert isinstance(converted[0], RetrievedChunkResponse)
    assert converted[0].source == "guide.md"


async def _collection_db(**kwargs: Any) -> Database:
    db = Database(Settings(database_url="sqlite+aiosqlite:///:memory:"))
    await db.startup()
    await db.create_all()
    async with db.session() as session:
        session.add(
            RagCollection(
                id="collection-1",
                name="test",
                embedding_model="stub",
                embedding_dimensions=2,
                chunk_size=kwargs.get("chunk_size", 40),
                chunk_overlap=kwargs.get("chunk_overlap", 5),
            )
        )
    return db


def _service_for(db: Database, store: Any) -> RagService:
    state = SimpleNamespace(
        db=db,
        redis=object(),
        settings=Settings(database_url="sqlite+aiosqlite:///:memory:"),
        components={},
    )
    service = RagService(state)  # type: ignore[arg-type]
    service._store = store
    return service


class VectorStore(InMemoryStore):
    """In-memory store that also serves stored vectors back by chunk id."""

    def __init__(self) -> None:
        super().__init__()
        self.vectors: dict[str, list[float]] = {}

    async def upsert(self, collection_id: str, payloads: list[dict[str, Any]]) -> None:
        await super().upsert(collection_id, payloads)
        for payload in payloads:
            self.vectors[payload["chunk_id"]] = list(payload["vector"])

    async def get_vectors(
        self, collection_id: str, chunk_ids: list[str]
    ) -> dict[str, list[float]]:
        return {
            chunk_id: self.vectors[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self.vectors
        }


@pytest.mark.asyncio
async def test_list_chunks_paginates_and_filters_by_document() -> None:
    db = await _collection_db()
    store = VectorStore()
    service = _service_for(db, store)
    try:
        long_text = "Sentence number one here. " * 8
        first = await ingest_document(
            db,
            store,
            StubEmbedder(),
            collection_id="collection-1",
            content=long_text,
            source="first.txt",
        )
        second = await ingest_document(
            db,
            store,
            StubEmbedder(),
            collection_id="collection-1",
            content="A separate short document.",
            source="second.txt",
        )

        page_one, total = await service.list_chunks("collection-1", limit=2, offset=0)
        page_two, total_again = await service.list_chunks("collection-1", limit=2, offset=2)
        assert total == total_again > 3
        assert len(page_one) == 2
        assert {chunk.id for chunk in page_one}.isdisjoint({chunk.id for chunk in page_two})

        filtered, filtered_total = await service.list_chunks(
            "collection-1", document_id=second.id
        )
        assert filtered_total == len(filtered) >= 1
        assert {chunk.document_id for chunk in filtered} == {second.id}
        assert first.id not in {chunk.document_id for chunk in filtered}
    finally:
        await db.shutdown()


@pytest.mark.asyncio
async def test_list_chunks_enforces_hard_page_cap_and_rejects_bad_paging() -> None:
    db = await _collection_db()
    service = _service_for(db, VectorStore())
    try:
        await ingest_document(
            db,
            VectorStore(),
            StubEmbedder(),
            collection_id="collection-1",
            content="Short doc.",
            source="doc.txt",
        )
        chunks, _ = await service.list_chunks("collection-1", limit=100_000)
        assert len(chunks) <= MAX_CHUNK_PAGE_SIZE
        with pytest.raises(InvalidRequestError):
            await service.list_chunks("collection-1", limit=0)
        with pytest.raises(InvalidRequestError):
            await service.list_chunks("collection-1", offset=-1)
    finally:
        await db.shutdown()


@pytest.mark.asyncio
async def test_chunk_response_truncates_embedding_preview() -> None:
    db = await _collection_db()
    store = VectorStore()
    service = _service_for(db, store)
    try:
        await ingest_document(
            db,
            store,
            StubEmbedder(vector=[float(index) for index in range(32)]),
            collection_id="collection-1",
            content="Document for embedding preview.",
            source="doc.txt",
        )
        chunks, _ = await service.list_chunks("collection-1")
        vectors = await service.chunk_vectors("collection-1", [chunk.id for chunk in chunks])
        response = ChunkResponse.of(chunks[0], vectors[chunks[0].id])

        assert response.embedding is not None
        assert response.embedding.dimensions == 32
        assert len(response.embedding.preview) == EMBEDDING_PREVIEW_VALUES
        assert response.embedding.truncated is True
        assert response.char_count == len(chunks[0].content)
        assert response.text == chunks[0].content
    finally:
        await db.shutdown()


@pytest.mark.asyncio
async def test_failed_ingest_surfaces_error_on_document() -> None:
    db = await _collection_db()
    service = _service_for(db, VectorStore())
    try:
        with pytest.raises(RuntimeError, match="redis write failed"):
            await ingest_document(
                db,
                FailingVectorStore(),
                StubEmbedder(),
                collection_id="collection-1",
                content="This ingestion fails.",
                source="broken.txt",
            )
        documents = await service.list_documents("collection-1")
        assert len(documents) == 1
        assert documents[0].status == "failed"
        assert "redis write failed" in (documents[0].error_message or "")

        response = DocumentResponse.of(documents[0])
        assert response.error_message == documents[0].error_message
        assert response.ingested_at is None
        assert response.created_at is not None

        # A retry after a failure must still be able to succeed.
        retried = await ingest_document(
            db,
            VectorStore(),
            StubEmbedder(),
            collection_id="collection-1",
            content="This ingestion fails.",
            source="broken.txt",
        )
        assert retried.status == "ready"
        assert retried.error_message is None
        assert DocumentResponse.of(retried).ingested_at is not None
    finally:
        await db.shutdown()


def test_collection_response_exposes_timestamps() -> None:
    collection = RagCollection(
        id="collection-1",
        name="test",
        embedding_model="stub",
        embedding_dimensions=2,
        chunk_size=1000,
        chunk_overlap=200,
        document_count=0,
        chunk_count=0,
        metadata_={},
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    response = CollectionResponse.model_validate(collection)
    assert response.created_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert response.updated_at == datetime(2026, 1, 2, tzinfo=UTC)
