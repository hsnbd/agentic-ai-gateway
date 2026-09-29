"""RAG service façade over collection metadata, embeddings, and vector search."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from app.core.errors import InvalidRequestError, NotFoundError
from app.core.pipeline import RequestContext
from app.core.schemas import ChatRequest, EmbeddingRequest, EmbeddingResponse
from app.core.state import GatewayState
from app.db.models import RagChunk, RagCollection, RagDocument
from app.rag.ingest import delete_document as remove_document
from app.rag.ingest import ingest_document
from app.rag.retrieve import RetrievedChunk, augment_request, retrieve
from app.rag.store import RagVectorStore

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
    ) -> RagCollection:
        collection = RagCollection(
            name=name,
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
            session.add(collection)
            await session.flush()
            await self.store.ensure_index(collection.id, collection.embedding_dimensions)
            return collection

    async def list_collections(self) -> list[RagCollection]:
        async with self.db.session() as session:
            result = await session.scalars(select(RagCollection).order_by(RagCollection.name))
            return list(result.all())

    async def get_collection(self, collection_id: str) -> RagCollection:
        async with self.db.session() as session:
            collection = await session.get(RagCollection, collection_id)
            if collection is None:
                raise NotFoundError(f"Unknown RAG collection: {collection_id}")
            return collection

    async def delete_collection(self, collection_id: str) -> None:
        async with self.db.session() as session:
            collection = await session.get(RagCollection, collection_id)
            if collection is None:
                raise NotFoundError(f"Unknown RAG collection: {collection_id}")
            await session.delete(collection)
        await self.store.drop_collection(collection_id)

    async def ingest(
        self,
        collection_id: str,
        content: str,
        *,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> RagDocument:
        collection = await self.get_collection(collection_id)
        return await ingest_document(
            self.db,
            self.store,
            self._embedder(collection.embedding_model, collection.embedding_dimensions),
            collection_id=collection_id,
            content=content,
            source=source,
            metadata=metadata,
        )

    async def list_documents(self, collection_id: str) -> list[RagDocument]:
        await self.get_collection(collection_id)
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
    ) -> tuple[list[RagChunk], int]:
        """Return one capped page of chunks plus the total matching count."""
        await self.get_collection(collection_id)
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

    async def get_chunk(self, collection_id: str, chunk_id: str) -> RagChunk:
        await self.get_collection(collection_id)
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

    async def delete_document(self, collection_id: str, document_id: str) -> None:
        await self.get_collection(collection_id)
        removed = await remove_document(
            self.db, self.store, collection_id=collection_id, document_id=document_id
        )
        if not removed:
            raise NotFoundError(f"Unknown RAG document: {document_id}")

    async def search(
        self,
        collection_id: str,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float = 0.0,
        filters: dict[str, str] | None = None,
        diversity: float = 0.0,
    ) -> list[RetrievedChunk]:
        collection = await self.get_collection(collection_id)
        return await retrieve(
            self.store,
            self._embedder(collection.embedding_model, collection.embedding_dimensions),
            collection_id=collection_id,
            query=query,
            top_k=(self.state.settings.rag_default_top_k if top_k is None else top_k),
            min_score=min_score,
            filters=filters,
            diversity=diversity,
        )

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
