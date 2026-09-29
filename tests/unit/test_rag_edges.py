"""Edge cases for RAG chunking, ingestion, retrieval, storage, the service, and the stage."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select

from app.config.settings import Settings
from app.core.errors import GatewayError, InvalidRequestError, NotFoundError
from app.core.pipeline import RequestContext
from app.core.schemas import (
    ChatRequest,
    EmbeddingResponse,
    EmbeddingVector,
    Message,
    RagOptions,
    Role,
    TextPart,
)
from app.db.models import RagCollection, RagDocument
from app.db.session import Database
from app.providers.base import Capabilities, Deployment
from app.rag import ingest as ingest_module
from app.rag.chunking import chunk_text
from app.rag.ingest import _embed_batch, delete_document, ingest_document
from app.rag.retrieve import RetrievedChunk, _cosine, augment_request, build_context, retrieve
from app.rag.service import RagService, _ProviderEmbedder
from app.rag.stage import RagStage, retrieve_and_augment
from app.rag.store import RagVectorStore, _escape_tag, _text, _value
from tests.unit.test_rag import FailingVectorStore, InMemoryStore, StubEmbedder

# -- Chunking ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_size": 0, "overlap": -1}, "greater than zero"),
        ({"chunk_size": 10, "overlap": -1}, "negative"),
        ({"chunk_size": 10, "overlap": 2, "strategy": "semantic"}, "Unsupported chunking"),
    ],
)
def test_chunking_validates_arguments(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidRequestError, match=message):
        chunk_text("text", **kwargs)


def test_markdown_chunking_without_headings_and_heading_only_documents() -> None:
    plain = chunk_text("just some text", chunk_size=100, overlap=0, strategy="markdown")
    assert [chunk.text for chunk in plain] == ["just some text"]
    # A document that is only a heading has no section body, so it is kept whole.
    headings = chunk_text("# Title\n", chunk_size=100, overlap=0, strategy="markdown")
    assert [chunk.text for chunk in headings] == ["# Title\n"]
    nested = chunk_text(
        "# A\ntext a\n## B\ntext b\n# C\ntext c", chunk_size=100, overlap=0, strategy="markdown"
    )
    assert [chunk.text.split("\n")[0] for chunk in nested] == ["# A", "# A > ## B", "# C"]


def test_splitter_falls_back_when_boundaries_are_inside_the_overlap() -> None:
    # The only whitespace sits within the overlap window, so a hard cut is used.
    chunks = chunk_text("a bcdefghijklmnop", chunk_size=6, overlap=3)
    assert all(len(chunk.text) <= 6 for chunk in chunks)


# -- Retrieval helpers ------------------------------------------------------


class SyncEmbedder:
    def embed(self, texts: list[str]) -> EmbeddingResponse:
        return EmbeddingResponse(
            model="m",
            data=[EmbeddingVector(index=i, embedding=[1.0, 0.0]) for i in range(len(texts))],
        )


class EmptyEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return []


def test_cosine_of_zero_vector_is_zero() -> None:
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


async def test_retrieve_guards_and_score_threshold() -> None:
    store = InMemoryStore([("a", 0.9, {"content": "keep"}), ("b", 0.1, {"content": "drop"})])
    assert await retrieve(store, StubEmbedder(), collection_id="c", query="q", top_k=0) == []
    with pytest.raises(InvalidRequestError, match="diversity"):
        await retrieve(store, StubEmbedder(), collection_id="c", query="q", diversity=2)
    assert await retrieve(store, EmptyEmbedder(), collection_id="c", query="q") == []
    kept = await retrieve(store, SyncEmbedder(), collection_id="c", query="q", min_score=0.5)
    assert [chunk.id for chunk in kept] == ["a"]


async def test_mmr_embeds_candidates_missing_stored_vectors() -> None:
    store = InMemoryStore(
        [("a", 0.9, {"content": "x"}), ("b", 0.8, {"content": "y", "_vector": [0.0, 1.0]})]
    )
    embedder = StubEmbedder()
    chosen = await retrieve(store, embedder, collection_id="c", query="q", top_k=2, diversity=0.5)
    assert [chunk.id for chunk in chosen] == ["a", "b"]
    assert embedder.calls[-1] == ["x"]


def _chunk(text: str, source: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(id="id", text=text, score=1.0, source=source)


def test_build_context_budgets() -> None:
    assert build_context([_chunk("text")], max_tokens=0) == ""
    # The header alone exceeds the budget, so nothing is added.
    assert build_context([_chunk("text", "a-very-long-source-name " * 20)], max_tokens=3) == ""
    truncated = build_context([_chunk("word " * 200)], max_tokens=20)
    assert truncated.startswith("[1] Source: unknown")
    assert len(truncated) < len("word " * 200)
    # Too little room after the header to fit even one character of content.
    tight = build_context([_chunk("first"), _chunk("second " * 50)], max_tokens=12)
    assert "second" not in tight


def test_augment_request_modes() -> None:
    request = ChatRequest(model="m", messages=[Message(role=Role.USER, content="question")])
    with pytest.raises(InvalidRequestError, match="mode"):
        augment_request(request, "ctx", mode="assistant")
    assert augment_request(request, "", mode="user").messages == request.messages
    user = augment_request(request, "ctx", mode="user")
    assert user.messages[0].content == "Retrieved context:\nctx\n\nquestion"

    parts = ChatRequest(model="m", messages=[Message(role=Role.USER, content=[TextPart(text="q")])])
    prefixed = augment_request(parts, "ctx", mode="user").messages[0].content
    assert isinstance(prefixed, list) and prefixed[0].text.startswith("Retrieved context")  # type: ignore[union-attr]

    no_user = ChatRequest(model="m", messages=[Message(role=Role.ASSISTANT, content="hi")])
    appended = augment_request(no_user, "ctx", mode="user")
    assert appended.messages[-1].role == Role.USER
    assert request.messages[0].content == "question"


# -- Vector store -----------------------------------------------------------


class ScriptedIndex:
    def __init__(self, redis: ScriptedRedis) -> None:
        self.redis = redis

    async def info(self) -> dict[str, Any]:
        if not self.redis.index_exists:
            raise RuntimeError("Unknown index name")
        return {}

    async def create_index(self, schema: Any, definition: Any) -> None:
        if self.redis.fail_create:
            raise RuntimeError("unknown command FT.CREATE")
        self.redis.index_exists = True

    async def search(self, query: Any, query_params: Any = None) -> Any:
        page = self.redis.pages.pop(0) if self.redis.pages else []
        return SimpleNamespace(docs=page, total=self.redis.total)

    async def dropindex(self, delete_documents: bool = False) -> None:
        if self.redis.fail_drop:
            raise RuntimeError("drop failed")


class ScriptedPipeline:
    def __init__(self, redis: ScriptedRedis) -> None:
        self.redis = redis
        self.ops: list[tuple[str, Any]] = []

    def hset(self, key: str, mapping: dict[str, Any]) -> None:
        self.ops.append(("hset", key))
        self.redis.hashes[key] = mapping

    def hget(self, key: str, field: str) -> None:
        self.ops.append(("hget", key))

    async def execute(self) -> list[Any]:
        if self.redis.fail_pipeline:
            raise ConnectionError("down")
        return [self.redis.vectors.get(key) for op, key in self.ops if op == "hget"]


class ScriptedRedis:
    def __init__(self) -> None:
        self.index_exists = False
        self.fail_create = False
        self.fail_drop = False
        self.fail_pipeline = False
        self.pages: list[list[Any]] = []
        self.total = 0
        self.hashes: dict[str, Any] = {}
        self.vectors: dict[str, Any] = {}
        self.deleted: list[str] = []

    def ft(self, name: str) -> ScriptedIndex:
        return ScriptedIndex(self)

    def pipeline(self, transaction: bool = True) -> ScriptedPipeline:
        return ScriptedPipeline(self)

    async def delete(self, *keys: str) -> int:
        self.deleted.extend(keys)
        return len(keys)


async def test_store_creates_indexes_once_and_marks_unavailable() -> None:
    redis = ScriptedRedis()
    store = RagVectorStore(redis, 2)
    await store.ensure_index("c", 2)
    assert redis.index_exists and store.available
    await store.ensure_index("c", 2)  # cached

    broken = ScriptedRedis()
    broken.fail_create = True
    unavailable = RagVectorStore(broken, 2)
    await unavailable.ensure_index("c", 2)
    assert not unavailable.available
    with pytest.raises(RuntimeError, match="unavailable"):
        await unavailable.upsert("c", [{"vector": [1.0]}])
    assert await unavailable.search("c", [1.0], 5) == []
    assert await unavailable.delete_document("c", "d") == 0


async def test_upsert_packs_vectors_and_rejects_when_index_cannot_be_made() -> None:
    redis = ScriptedRedis()
    store = RagVectorStore(redis, 2)
    await store.upsert("c", [])
    await store.upsert(
        "c",
        [
            {"chunk_id": "1", "document_id": "d", "content": "x", "vector": [1.0, 0.0]},
            {"chunk_id": "2", "document_id": "d", "content": "y", "vector": b"\x00" * 8},
        ],
    )
    assert set(redis.hashes) == {"aigw:rag:c:1", "aigw:rag:c:2"}

    failing = ScriptedRedis()
    failing.fail_create = True
    with pytest.raises(RuntimeError, match="unavailable"):
        await RagVectorStore(failing, 2).upsert("c", [{"chunk_id": "1", "vector": [1.0]}])


async def test_search_filters_and_row_decoding() -> None:
    redis = ScriptedRedis()
    redis.index_exists = True
    redis.pages = [[{"chunk_id": "a", "vector_distance": 0.5, "metadata": "not json"}]]
    store = RagVectorStore(redis, 2)
    assert await store.search("c", [1.0], 0) == []
    with pytest.raises(InvalidRequestError, match="Unsupported RAG filter"):
        await store.search("c", [1.0], 3, filters={"colour": "red"})
    [(chunk_id, score, metadata)] = await store.search("c", [1.0], 3, filters={"source": "a b"})
    assert (chunk_id, score) == ("a", 0.5)
    assert metadata["content"] == ""
    assert metadata["_vector"] == []  # the row carried no vector bytes

    failing = ScriptedRedis()
    failing.fail_create = True
    assert await RagVectorStore(failing, 2).search("c", [1.0], 3) == []


async def test_get_vectors_and_deletes() -> None:
    redis = ScriptedRedis()
    store = RagVectorStore(redis, 2)
    assert await store.get_vectors("c", []) == {}
    redis.vectors = {"aigw:rag:c:a": b"\x00\x00\x80?", "aigw:rag:c:b": b"\x00"}
    assert await store.get_vectors("c", ["a", "b", "missing"]) == {"a": [1.0]}
    redis.fail_pipeline = True
    assert await store.get_vectors("c", ["a"]) == {}

    redis.total = 3
    redis.pages = [[SimpleNamespace(id="k1"), SimpleNamespace(id="k2")], [SimpleNamespace(id="k3")]]
    assert await store.delete_document("c", "doc") == 3
    redis.total = 5
    redis.pages = [[]]
    assert await store.delete_document("c", "doc") == 0

    redis.fail_drop = True
    await store.drop_collection("c")
    store.available = False
    await store.drop_collection("c")


def test_store_helpers() -> None:
    assert _escape_tag("a b-c") == "a\\ b\\-c"
    assert _value({"x": 1}, "x", 0) == 1
    assert _value(SimpleNamespace(), "x", 0) == 0
    assert _text(b"bytes") == "bytes" and _text(3) == "3"


# -- Ingestion --------------------------------------------------------------


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    database = Database(Settings(database_url="sqlite+aiosqlite:///:memory:"))
    await database.startup()
    await database.create_all()
    yield database
    await database.shutdown()


async def _collection(db: Database, **fields: Any) -> RagCollection:
    async with db.session() as session:
        collection = RagCollection(
            name=fields.pop("name", "docs"),
            embedding_model="embed",
            embedding_dimensions=2,
            chunk_size=fields.pop("chunk_size", 100),
            chunk_overlap=0,
            **fields,
        )
        session.add(collection)
    return collection


async def test_embed_batch_accepts_sync_responses_and_checks_counts() -> None:
    assert await _embed_batch(SyncEmbedder(), ["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    with pytest.raises(ValueError, match="different number"):
        await _embed_batch(EmptyEmbedder(), ["a"])


async def test_ingest_failure_is_recorded_then_retried(db: Database) -> None:
    collection = await _collection(db)
    with pytest.raises(RuntimeError, match="redis write failed"):
        await ingest_document(
            db,
            FailingVectorStore(),
            StubEmbedder(),
            collection_id=collection.id,
            content="hello",
            source="",
            metadata={"title": "Greeting"},
        )
    async with db.session() as session:
        [failed] = (await session.scalars(select(RagDocument))).all()
    assert failed.status == "failed" and failed.title == "Greeting"

    # A second failure updates the same placeholder row.
    with pytest.raises(RuntimeError):
        await ingest_document(
            db,
            FailingVectorStore(),
            StubEmbedder(),
            collection_id=collection.id,
            content="hello",
            source="",
        )

    document = await ingest_document(
        db,
        InMemoryStore(),
        StubEmbedder(),
        collection_id=collection.id,
        content="hello",
        source="greeting.txt",
    )
    assert document.status == "ready"
    # Re-ingesting identical content returns the existing document.
    again = await ingest_document(
        db,
        InMemoryStore(),
        StubEmbedder(),
        collection_id=collection.id,
        content="hello",
        source="greeting.txt",
    )
    assert again.id == document.id


async def test_ingest_into_unknown_collection_and_empty_documents(db: Database) -> None:
    with pytest.raises(ValueError, match="Unknown RAG collection"):
        await ingest_document(
            db, InMemoryStore(), StubEmbedder(), collection_id="missing", content="x", source=""
        )
    collection = await _collection(db, name="empty")
    store = InMemoryStore()
    document = await ingest_document(
        db, store, StubEmbedder(), collection_id=collection.id, content="", source=""
    )
    assert document.chunk_count == 0 and store.upserts == []


async def test_vector_cleanup_failure_still_raises_the_original_error(db: Database) -> None:
    class DoubleFailure(FailingVectorStore):
        async def delete_document(self, collection_id: str, document_id: str) -> int:
            raise RuntimeError("cleanup failed")

    collection = await _collection(db)
    with pytest.raises(RuntimeError, match="redis write failed"):
        await ingest_document(
            db,
            DoubleFailure(),
            StubEmbedder(),
            collection_id=collection.id,
            content="text",
            source="",
        )


async def test_recording_a_failure_tolerates_database_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenDb:
        def session(self) -> Any:
            raise RuntimeError("db down")

    await ingest_module._record_failure(
        BrokenDb(),
        collection_id="c",
        content="x",
        source="",
        metadata=None,
        error="e",  # type: ignore[arg-type]
    )


async def test_delete_document_paths(db: Database) -> None:
    store = InMemoryStore()
    assert not await delete_document(db, store, collection_id="c", document_id="missing")

    collection = await _collection(db)
    document = await ingest_document(
        db, store, StubEmbedder(), collection_id=collection.id, content="text", source=""
    )
    async with db.session() as session:
        await session.delete(await session.get(RagCollection, collection.id))
        orphan = RagDocument(
            id=document.id, collection_id="gone", title="t", content_hash="h", byte_size=1
        )
        session.add(orphan)
    assert await delete_document(db, store, collection_id="gone", document_id=document.id)


# -- Service ----------------------------------------------------------------


class _Registry:
    def __init__(self, embeddings: bool = True) -> None:
        self.deployment = Deployment(
            id="e",
            model_name="embed",
            provider="fake",
            provider_model="e",
            capabilities=Capabilities(embeddings=embeddings),
        )

    def deployments_for(self, model: str) -> list[Deployment]:
        return [self.deployment]

    def provider_for(self, deployment: Deployment) -> Any:
        class _Provider:
            async def embed(self, request: Any, deployment: Any) -> EmbeddingResponse:
                return EmbeddingResponse(
                    model="e",
                    data=[
                        EmbeddingVector(index=i, embedding=[1.0, 0.0])
                        for i in range(len(request.input))
                    ],
                )

        return _Provider()


def _service(db: Database, redis: Any = None, registry: Any = None) -> RagService:
    state = SimpleNamespace(
        db=db,
        redis=redis,
        registry=registry or _Registry(),
        settings=Settings(
            rag_embedding_model="embed",
            rag_embedding_dimensions=2,
            rag_chunk_size=100,
            rag_chunk_overlap=0,
            rag_default_top_k=3,
        ),
    )
    return RagService(state)  # type: ignore[arg-type]


async def test_provider_embedder_requires_an_embedding_deployment() -> None:
    state = SimpleNamespace(registry=_Registry(embeddings=False))
    with pytest.raises(InvalidRequestError, match="No embedding deployment"):
        await _ProviderEmbedder(state, "embed", 2).embed(["x"])
    assert await _ProviderEmbedder(SimpleNamespace(registry=_Registry()), "embed", 2).embed(
        ["x"]
    ) == [[1.0, 0.0]]


async def test_service_requires_redis_and_validates_collections(db: Database) -> None:
    service = _service(db)
    with pytest.raises(RuntimeError, match="Redis"):
        _ = service.store
    service = _service(db, ScriptedRedis())
    with pytest.raises(InvalidRequestError, match="chunk_overlap"):
        await service.create_collection("bad", chunk_size=10, chunk_overlap=10)
    for missing in (service.get_collection, service.delete_collection):
        with pytest.raises(NotFoundError):
            await missing("missing")


async def test_service_crud_search_and_augment(db: Database) -> None:
    redis = ScriptedRedis()
    service = _service(db, redis)
    collection = await service.create_collection("docs", description="d", metadata={"k": 1})
    assert [c.name for c in await service.list_collections()] == ["docs"]
    service._store = InMemoryStore(  # type: ignore[assignment]
        [("chunk", 0.9, {"content": "grounding", "source": "s"})]
    )
    document = await service.ingest(collection.id, "some content", source="s.txt")
    assert [d.id for d in await service.list_documents(collection.id)] == [document.id]

    chunks, total = await service.list_chunks(collection.id, document_id=document.id, limit=1000)
    assert total == 1 and len(chunks) == 1
    with pytest.raises(InvalidRequestError, match="limit"):
        await service.list_chunks(collection.id, limit=0)
    with pytest.raises(InvalidRequestError, match="offset"):
        await service.list_chunks(collection.id, offset=-1)
    assert (await service.get_chunk(collection.id, chunks[0].id)).id == chunks[0].id
    with pytest.raises(NotFoundError):
        await service.get_chunk(collection.id, "missing")

    assert await service.chunk_vectors(collection.id, []) == {}
    assert await service.chunk_vectors(collection.id, ["a"]) == {}  # InMemoryStore lacks it

    results = await service.search(collection.id, "q")
    assert results[0].text == "grounding"

    request = ChatRequest(model="m", messages=[Message(role=Role.USER, content="question")])
    augmented = await service.augment(request, collection.id, mode="user", top_k=1)
    assert isinstance(augmented, ChatRequest)
    assert "grounding" in augmented.messages[0].text()
    ctx = RequestContext(request=request, state=SimpleNamespace())  # type: ignore[arg-type]
    assert await service.augment(ctx, collection.id, query="explicit") is ctx
    assert ctx.request.messages[0].role == Role.SYSTEM

    with pytest.raises(NotFoundError):
        await service.delete_document(collection.id, "missing")
    await service.delete_document(collection.id, document.id)
    service._store = RagVectorStore(redis, 2)
    await service.delete_collection(collection.id)


async def test_chunk_vectors_reads_through_the_store(db: Database) -> None:
    redis = ScriptedRedis()
    redis.vectors = {"aigw:rag:c:a": b"\x00\x00\x80?"}
    assert await _service(db, redis).chunk_vectors("c", ["a"]) == {"a": [1.0]}


# -- Stage ------------------------------------------------------------------


class _SearchService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def search(self, collection_id: str, query: str, **kwargs: Any) -> list[RetrievedChunk]:
        if self.error is not None:
            raise self.error
        return [_chunk("fact", "doc.md")]


async def test_retrieval_failures_become_rag_unavailable() -> None:
    request = ChatRequest(model="m", messages=[Message(role=Role.USER, content="q")])
    options = RagOptions(collection_id="c")
    with pytest.raises(GatewayError) as raised:
        await retrieve_and_augment(_SearchService(RuntimeError("redis down")), request, options)  # type: ignore[arg-type]
    assert raised.value.code.value == "rag_unavailable"
    with pytest.raises(NotFoundError):
        await retrieve_and_augment(_SearchService(NotFoundError("x")), request, options)  # type: ignore[arg-type]


async def test_rag_stage_records_sources_and_details() -> None:
    request = ChatRequest(
        model="m",
        messages=[Message(role=Role.USER, content="q")],
        rag=RagOptions(collection_id="c", query="explicit"),
    )
    ctx = RequestContext(
        request=request,
        state=SimpleNamespace(components={"rag_service": _SearchService()}),  # type: ignore[arg-type]
    )
    await RagStage().process(ctx)
    assert ctx.rag_sources[0]["source"] == "doc.md"
    assert ctx._rag_details["chunks"] == 1  # type: ignore[attr-defined]
    assert ctx.request.messages[0].role == Role.SYSTEM

    plain = RequestContext(
        request=ChatRequest(model="m", messages=[]),
        state=SimpleNamespace(),  # type: ignore[arg-type]
    )
    assert await RagStage().process(plain) is None


def test_empty_range_produces_no_chunks() -> None:
    from app.rag.chunking import _chunk_range

    assert _chunk_range("text", 2, 2, chunk_size=10, overlap=0, first_index=0) == []
