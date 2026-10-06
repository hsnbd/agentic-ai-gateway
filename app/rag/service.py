"""RAG service façade over collection metadata, embeddings, and vector search."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select

from app.core.errors import InvalidRequestError, NotFoundError, PermissionDeniedError
from app.core.pipeline import RequestContext
from app.core.schemas import ChatRequest, EmbeddingRequest, EmbeddingResponse
from app.core.state import GatewayState
from app.db.models import RagChunk, RagCollection, RagDocument
from app.observability.metrics import (
    RAG_INGEST_SECONDS,
    RAG_INGESTIONS,
    RAG_RETRIEVAL_SECONDS,
    RAG_RETRIEVALS,
)
from app.rag.access import UNRESTRICTED, RagAccess
from app.rag.ingest import delete_document as remove_document
from app.rag.ingest import ingest_document, recover_interrupted, start_background_ingest
from app.rag.rerank import rerank
from app.rag.retrieve import RetrievedChunk, augment_request, retrieve
from app.rag.store import BUILT_IN_FILTERS, RagVectorStore

logger = logging.getLogger(__name__)

#: Filterable field names become Redis Search tag fields.
_FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_MAX_FILTERABLE_FIELDS = 16
#: Most candidates retrieved for a reranker to order.
_MAX_RERANK_CANDIDATES = 50

#: Chunks embedded and written per round trip during a reindex.
_REINDEX_BATCH_SIZE = 256

#: Default and hard-capped page sizes for chunk listing.
DEFAULT_CHUNK_PAGE_SIZE = 50
MAX_CHUNK_PAGE_SIZE = 200


class _ProviderEmbedder:
    def __init__(self, state: Any, model: str, dimensions: int) -> None:
        self.state = state
        self.model = model
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        deployments = self.state.registry.deployments_for(self.model)
        supported = [deployment for deployment in deployments if deployment.capabilities.embeddings]
        if not supported:
            raise InvalidRequestError(f"No embedding deployment is configured for {self.model!r}")
        deployment = supported[0]
        provider = self.state.registry.provider_for(deployment)
        response: EmbeddingResponse = await provider.embed(
            EmbeddingRequest(model=self.model, input=texts, dimensions=self.dimensions), deployment
        )
        return [item.embedding for item in response.data]


class RagService:
    """Application-level RAG operations; database sessions are always scoped."""

    def __init__(self, state: GatewayState) -> None:
        self.state = state
        self.db = state.db
        self._store: RagVectorStore | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._ingest_slots = asyncio.Semaphore(
            max(1, int(getattr(state.settings, "rag_ingest_concurrency", 2)))
        )

    @property
    def store(self) -> RagVectorStore:
        if self._store is None:
            if self.state.redis is None:
                raise RuntimeError("Gateway Redis connection has not started")
            self._store = RagVectorStore(
                self.state.redis, self.state.settings.rag_embedding_dimensions
            )
        return self._store

    def _embedder(self, model: str, dimensions: int) -> _ProviderEmbedder:
        return _ProviderEmbedder(self.state, model, dimensions)

    async def create_collection(
        self,
        name: str,
        *,
        description: str | None = None,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        metadata: dict[str, Any] | None = None,
        access: RagAccess = UNRESTRICTED,
        filterable_fields: list[str] | None = None,
    ) -> RagCollection:
        owner = access.owner_fields()
        fields = list(dict.fromkeys(filterable_fields or []))
        for field in fields:
            if field in BUILT_IN_FILTERS or not _FIELD_NAME.fullmatch(field):
                raise InvalidRequestError(
                    f"Invalid filterable field {field!r}: use letters, digits, and "
                    "underscores, and not a built-in filter"
                )
        if len(fields) > _MAX_FILTERABLE_FIELDS:
            raise InvalidRequestError(f"At most {_MAX_FILTERABLE_FIELDS} filterable fields")
        if fields:
            metadata = {**(metadata or {}), "filterable_fields": fields}
        collection = RagCollection(
            name=name,
            **owner,
            description=description,
            embedding_model=embedding_model or self.state.settings.rag_embedding_model,
            embedding_dimensions=(
                embedding_dimensions or self.state.settings.rag_embedding_dimensions
            ),
            chunk_size=chunk_size or self.state.settings.rag_chunk_size,
            chunk_overlap=(
                self.state.settings.rag_chunk_overlap if chunk_overlap is None else chunk_overlap
            ),
            metadata_=metadata or {},
        )
        if collection.chunk_overlap >= collection.chunk_size:
            raise InvalidRequestError("chunk_overlap must be smaller than chunk_size")
        async with self.db.session() as session:
            clash = await session.scalar(
                select(RagCollection.id).where(
                    RagCollection.name == name,
                    _same_owner(RagCollection.owner_team_id, owner["owner_team_id"]),
                    _same_owner(RagCollection.owner_key_id, owner["owner_key_id"]),
                )
            )
            if clash is not None:
                raise InvalidRequestError(
                    f"A RAG collection named {name!r} already exists", status_code=409
                )
            session.add(collection)
            await session.flush()
            self.store.declare_fields(collection.id, collection.filterable_fields)
            await self.store.ensure_index(collection.id, collection.embedding_dimensions)
            return collection

    async def list_collections(self, access: RagAccess = UNRESTRICTED) -> list[RagCollection]:
        async with self.db.session() as session:
            result = await session.scalars(
                select(RagCollection).where(access.visible_filter()).order_by(RagCollection.name)
            )
            return list(result.all())

    async def get_collection(
        self, collection_id: str, access: RagAccess = UNRESTRICTED, *, write: bool = False
    ) -> RagCollection:
        """Fetch a collection the caller may read (or, with ``write``, change)."""
        async with self.db.session() as session:
            collection = await session.get(RagCollection, collection_id)
        authorized = _authorized(collection, collection_id, access, write=write)
        # The store needs the collection's indexed fields for filters, writes, and
        # recreating its index; every store operation goes through here first.
        self.store.declare_fields(collection_id, authorized.filterable_fields)
        return authorized

    async def delete_collection(self, collection_id: str, access: RagAccess = UNRESTRICTED) -> None:
        async with self.db.session() as session:
            collection = await session.get(RagCollection, collection_id)
            await session.delete(_authorized(collection, collection_id, access, write=True))
        await self.store.drop_collection(collection_id)

    async def ingest(
        self,
        collection_id: str,
        content: str,
        *,
        source: str = "",
        metadata: dict[str, Any] | None = None,
        access: RagAccess = UNRESTRICTED,
        replace_existing: bool = True,
    ) -> RagDocument:
        """Ingest a document; large ones continue in the background.

        A document of at least ``RAG_BACKGROUND_INGEST_BYTES`` returns at once
        with status ``processing`` (poll it until ``ready`` or ``failed``).
        With ``replace_existing``, older documents from the same ``source`` are
        removed once the new one is ready, so re-uploading a file updates it.
        """
        settings = self.state.settings
        collection = await self.get_collection(collection_id, access, write=True)
        size = len(content.encode("utf-8"))
        if size > settings.rag_max_document_bytes:
            raise InvalidRequestError(
                f"Document is {size} bytes; the limit is {settings.rag_max_document_bytes}",
                status_code=413,
            )
        embedder = self._embedder(collection.embedding_model, collection.embedding_dimensions)
        if size >= settings.rag_background_ingest_bytes:
            document, start = await start_background_ingest(
                self.db,
                collection_id=collection_id,
                content=content,
                source=source,
                metadata=metadata,
            )
            if start:
                task = asyncio.create_task(
                    self._ingest_later(
                        collection_id, content, source, metadata, embedder, replace_existing
                    )
                )
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            return document
        document = await self._timed_ingest(
            "inline", collection_id, content, source, metadata, embedder
        )
        if replace_existing:
            await self._replace_source(collection_id, document)
        return document

    async def _timed_ingest(
        self,
        mode: str,
        collection_id: str,
        content: str,
        source: str,
        metadata: dict[str, Any] | None,
        embedder: Any,
    ) -> RagDocument:
        started = time.perf_counter()
        try:
            document = await ingest_document(
                self.db,
                self.store,
                embedder,
                collection_id=collection_id,
                content=content,
                source=source,
                metadata=metadata,
            )
        except Exception:
            RAG_INGESTIONS.labels("failed", mode).inc()
            raise
        finally:
            RAG_INGEST_SECONDS.labels(mode).observe(time.perf_counter() - started)
        RAG_INGESTIONS.labels("ready", mode).inc()
        return document

    async def _ingest_later(
        self,
        collection_id: str,
        content: str,
        source: str,
        metadata: dict[str, Any] | None,
        embedder: Any,
        replace_existing: bool,
    ) -> None:
        async with self._ingest_slots:
            try:
                document = await self._timed_ingest(
                    "background", collection_id, content, source, metadata, embedder
                )
            except Exception:
                # ingest_document already recorded the reason on the row.
                logger.warning("Background ingestion into %s failed", collection_id, exc_info=True)
                return
            if replace_existing:
                await self._replace_source(collection_id, document)

    async def _replace_source(self, collection_id: str, document: RagDocument) -> None:
        """Remove older versions: other documents in the collection with the same source."""
        if not document.source:
            return
        async with self.db.session() as session:
            older = list(
                (
                    await session.scalars(
                        select(RagDocument.id).where(
                            RagDocument.collection_id == collection_id,
                            RagDocument.source == document.source,
                            RagDocument.id != document.id,
                        )
                    )
                ).all()
            )
        for document_id in older:
            await remove_document(
                self.db, self.store, collection_id=collection_id, document_id=document_id
            )

    async def recover_interrupted(self) -> int:
        """Fail documents a restart left half-ingested (called at startup)."""
        recovered = await recover_interrupted(self.db)
        if recovered:
            logger.warning("Marked %d interrupted RAG ingestions as failed", recovered)
        return recovered

    async def close(self) -> None:
        """Cancel background ingestions; they are recovered as failed on next start."""
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def list_documents(
        self, collection_id: str, access: RagAccess = UNRESTRICTED
    ) -> list[RagDocument]:
        await self.get_collection(collection_id, access)
        async with self.db.session() as session:
            result = await session.scalars(
                select(RagDocument)
                .where(RagDocument.collection_id == collection_id)
                .order_by(RagDocument.created_at.desc())
            )
            return list(result.all())

    async def list_chunks(
        self,
        collection_id: str,
        *,
        document_id: str | None = None,
        limit: int = DEFAULT_CHUNK_PAGE_SIZE,
        offset: int = 0,
        access: RagAccess = UNRESTRICTED,
    ) -> tuple[list[RagChunk], int]:
        """Return one capped page of chunks plus the total matching count."""
        await self.get_collection(collection_id, access)
        if limit <= 0:
            raise InvalidRequestError("limit must be greater than zero")
        if offset < 0:
            raise InvalidRequestError("offset cannot be negative")
        limit = min(limit, MAX_CHUNK_PAGE_SIZE)

        filters = [RagChunk.collection_id == collection_id]
        if document_id is not None:
            filters.append(RagChunk.document_id == document_id)
        async with self.db.session() as session:
            total = await session.scalar(select(func.count()).select_from(RagChunk).where(*filters))
            result = await session.scalars(
                select(RagChunk)
                .where(*filters)
                .order_by(RagChunk.document_id, RagChunk.chunk_index)
                .offset(offset)
                .limit(limit)
            )
            return list(result.all()), int(total or 0)

    async def get_chunk(
        self, collection_id: str, chunk_id: str, access: RagAccess = UNRESTRICTED
    ) -> RagChunk:
        await self.get_collection(collection_id, access)
        async with self.db.session() as session:
            chunk = await session.scalar(
                select(RagChunk).where(
                    RagChunk.id == chunk_id, RagChunk.collection_id == collection_id
                )
            )
            if chunk is None:
                raise NotFoundError(f"Unknown RAG chunk: {chunk_id}")
            return chunk

    async def chunk_vectors(
        self, collection_id: str, chunk_ids: list[str]
    ) -> dict[str, list[float]]:
        """Best-effort vector lookup; retrieval stays usable if Redis is down."""
        if not chunk_ids:
            return {}
        try:
            return await self.store.get_vectors(collection_id, chunk_ids)
        except Exception:
            return {}

    async def delete_document(
        self, collection_id: str, document_id: str, access: RagAccess = UNRESTRICTED
    ) -> None:
        await self.get_collection(collection_id, access, write=True)
        removed = await remove_document(
            self.db, self.store, collection_id=collection_id, document_id=document_id
        )
        if not removed:
            raise NotFoundError(f"Unknown RAG document: {document_id}")

    async def index_status(
        self, collection_id: str, access: RagAccess = UNRESTRICTED
    ) -> dict[str, Any]:
        """Compare the chunks stored in Postgres with the vectors Redis can search."""
        collection = await self.get_collection(collection_id, access)
        stats = await self.store.index_stats(collection_id)
        return {
            "collection_id": collection_id,
            **stats,
            "stored_chunks": collection.chunk_count,
            "in_sync": stats["index_exists"] and stats["indexed_chunks"] == collection.chunk_count,
            "reindex": collection.metadata_.get("reindex"),
        }

    async def start_reindex(
        self, collection_id: str, access: RagAccess = UNRESTRICTED
    ) -> RagCollection:
        """Mark a reindex as running; ``reindex`` then does the work in the background."""
        async with self.db.session() as session:
            collection = _authorized(
                await session.get(RagCollection, collection_id), collection_id, access, write=True
            )
            if (collection.metadata_.get("reindex") or {}).get("status") == "running":
                raise InvalidRequestError(
                    f"A reindex of {collection_id} is already running", status_code=409
                )
            collection.metadata_ = {
                **collection.metadata_,
                "reindex": {"status": "running", "started_at": _now()},
            }
            return collection

    async def reindex(self, collection_id: str) -> None:
        """Rebuild the collection's vectors from the chunk text stored in Postgres.

        Recovers a collection whose vectors were lost (FLUSHALL, a Redis restart
        without persistence) or that needs re-embedding. Chunks are embedded in
        batches, with the same payload ingestion writes.
        """
        started = time.perf_counter()
        try:
            collection = await self.get_collection(collection_id)
            embedder = self._embedder(collection.embedding_model, collection.embedding_dimensions)
            await self.store.reset(collection_id, collection.embedding_dimensions)
            total = 0
            offset = 0
            while True:
                async with self.db.session() as session:
                    rows = (
                        await session.execute(
                            select(RagChunk, RagDocument)
                            .join(RagDocument, RagChunk.document_id == RagDocument.id)
                            .where(RagChunk.collection_id == collection_id)
                            .order_by(RagChunk.document_id, RagChunk.chunk_index)
                            .offset(offset)
                            .limit(_REINDEX_BATCH_SIZE)
                        )
                    ).all()
                if not rows:
                    break
                vectors = await embedder.embed([chunk.content for chunk, _ in rows])
                await self.store.upsert(
                    collection_id,
                    [
                        {
                            "document_id": document.id,
                            "chunk_id": chunk.id,
                            "chunk_index": chunk.chunk_index,
                            "source": document.source or "",
                            "content": chunk.content,
                            "metadata": {**document.metadata_, **chunk.metadata_},
                            "vector": vector,
                        }
                        for (chunk, document), vector in zip(rows, vectors, strict=True)
                    ],
                )
                total += len(rows)
                offset += len(rows)
            outcome: dict[str, Any] = {"status": "completed", "chunks": total}
        except Exception as exc:
            logger.exception("Reindex of RAG collection %s failed", collection_id)
            outcome = {"status": "failed", "error": str(exc)}
        outcome.update(finished_at=_now(), seconds=round(time.perf_counter() - started, 3))
        async with self.db.session() as session:
            current = await session.get(RagCollection, collection_id)
            if current is not None:  # it may have been deleted meanwhile
                current.metadata_ = {**current.metadata_, "reindex": outcome}

    async def search(
        self,
        collection_id: str,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float = 0.0,
        filters: dict[str, str] | None = None,
        diversity: float = 0.0,
        search_mode: str = "vector",
        access: RagAccess = UNRESTRICTED,
        rerank_model: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks, optionally reranked by a chat model.

        ``rerank_model`` (or the collection's ``metadata.rerank_model``) asks a
        model to reorder three times as many candidates, keeping the best
        ``top_k``; see app/rag/rerank.py.
        """
        collection = await self.get_collection(collection_id, access)
        limit = self.state.settings.rag_default_top_k if top_k is None else top_k
        reranker = rerank_model or collection.metadata_.get("rerank_model")
        started = time.perf_counter()
        chunks = await retrieve(
            self.store,
            self._embedder(collection.embedding_model, collection.embedding_dimensions),
            collection_id=collection_id,
            query=query,
            top_k=min(limit * 3, _MAX_RERANK_CANDIDATES) if reranker else limit,
            min_score=min_score,
            filters=filters,
            diversity=diversity,
            search_mode=search_mode,
        )
        if reranker:
            chunks = await rerank(self._model_caller(), str(reranker), query, chunks, top_k=limit)
        RAG_RETRIEVAL_SECONDS.labels(search_mode).observe(time.perf_counter() - started)
        RAG_RETRIEVALS.labels(search_mode, "hit" if chunks else "empty").inc()
        return chunks

    def _model_caller(self) -> Any:
        from app.core.builder import direct_model_caller

        return direct_model_caller(self.state)

    async def augment(
        self,
        ctx_or_request: RequestContext | ChatRequest,
        collection_id: str,
        **opts: Any,
    ) -> RequestContext | ChatRequest:
        query = opts.pop("query", None)
        max_tokens = opts.pop("max_tokens", 4000)
        mode = opts.pop("mode", "system")
        if query is None:
            request = (
                ctx_or_request.request
                if isinstance(ctx_or_request, RequestContext)
                else ctx_or_request
            )
            query = next(
                (
                    message.text()
                    for message in reversed(request.messages)
                    if message.role.value == "user"
                ),
                "",
            )
        chunks = await self.search(collection_id, query, **opts)
        from app.rag.retrieve import build_context

        context = build_context(chunks, max_tokens=max_tokens)
        request = (
            ctx_or_request.request if isinstance(ctx_or_request, RequestContext) else ctx_or_request
        )
        updated = augment_request(request, context, mode=mode)
        if isinstance(ctx_or_request, RequestContext):
            ctx_or_request.request = updated
            return ctx_or_request
        return updated


def _same_owner(column: Any, value: str | None) -> Any:
    return column.is_(None) if value is None else column == value


def _authorized(
    collection: RagCollection | None, collection_id: str, access: RagAccess, *, write: bool
) -> RagCollection:
    # Collections the caller cannot read are "not found", so other tenants'
    # collection ids are not confirmed to exist.
    if collection is None or not access.can_read(collection):
        raise NotFoundError(f"Unknown RAG collection: {collection_id}")
    if write and not access.can_write(collection):
        raise PermissionDeniedError(
            f"RAG collection {collection_id} is shared; only an operator can change it"
        )
    return collection


def _now() -> str:
    return datetime.now(UTC).isoformat()
