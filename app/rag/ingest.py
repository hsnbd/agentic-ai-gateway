"""Document ingestion into PostgreSQL metadata and Redis vectors."""

from __future__ import annotations

import hashlib
import inspect
import logging
from collections.abc import Mapping
from typing import Any

from sqlalchemy import select

from app.db.models import RagChunk, RagCollection, RagDocument
from app.db.session import Database
from app.rag.chunking import chunk_text

logger = logging.getLogger(__name__)
_EMBED_BATCH_SIZE = 64


async def _embed_batch(embedder: Any, texts: list[str]) -> list[list[float]]:
    result = embedder.embed(texts)
    if inspect.isawaitable(result):
        result = await result
    if hasattr(result, "data"):
        result = [entry.embedding for entry in result.data]
    if len(result) != len(texts):
        raise ValueError("Embedding provider returned a different number of vectors than inputs")
    return [list(vector) for vector in result]


async def ingest_document(
    db: Database,
    store: Any,
    embedder: Any,
    *,
    collection_id: str,
    content: str,
    source: str,
    metadata: Mapping[str, Any] | None = None,
) -> RagDocument:
    """Ingest a document, recording the reason on the row when ingestion fails."""
    try:
        return await _ingest(
            db,
            store,
            embedder,
            collection_id=collection_id,
            content=content,
            source=source,
            metadata=metadata,
        )
    except Exception as exc:
        # The ingest transaction rolled back, so the failure is persisted separately;
        # otherwise a failed document would surface as merely "not ready" with no reason.
        await _record_failure(
            db,
            collection_id=collection_id,
            content=content,
            source=source,
            metadata=metadata,
            error=str(exc) or type(exc).__name__,
        )
        raise


async def _record_failure(
    db: Database,
    *,
    collection_id: str,
    content: str,
    source: str,
    metadata: Mapping[str, Any] | None,
    error: str,
) -> None:
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    try:
        async with db.session() as session:
            document = await session.scalar(
                select(RagDocument).where(
                    RagDocument.collection_id == collection_id,
                    RagDocument.content_hash == content_hash,
                )
            )
            chunk_metadata = dict(metadata or {})
            if document is None:
                document = RagDocument(
                    collection_id=collection_id,
                    title=str(chunk_metadata.get("title") or source or "Untitled document"),
                    source=source or None,
                    content_type=str(chunk_metadata.get("content_type", "text/plain")),
                    content_hash=content_hash,
                    byte_size=len(content.encode("utf-8")),
                    chunk_count=0,
                    metadata_=chunk_metadata,
                )
                session.add(document)
            document.status = "failed"
            document.error_message = error[:4000]
    except Exception:
        logger.exception("Could not record RAG ingestion failure for collection %s", collection_id)


async def _ingest(
    db: Database,
    store: Any,
    embedder: Any,
    *,
    collection_id: str,
    content: str,
    source: str,
    metadata: Mapping[str, Any] | None = None,
) -> RagDocument:
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    async with db.session() as session:
        existing = await session.scalar(
            select(RagDocument).where(
                RagDocument.collection_id == collection_id,
                RagDocument.content_hash == content_hash,
            )
        )
        if existing is not None and existing.status != "failed":
            return existing
        if existing is not None:
            # A previous attempt failed; drop the placeholder so a retry can succeed.
            await session.delete(existing)
            await session.flush()

        collection = await session.get(RagCollection, collection_id)
        if collection is None:
            raise ValueError(f"Unknown RAG collection: {collection_id}")
        chunk_metadata = dict(metadata or {})
        chunks = chunk_text(
            content,
            chunk_size=collection.chunk_size,
            overlap=collection.chunk_overlap,
            strategy=str(chunk_metadata.get("chunking_strategy", "recursive")),
        )
        vectors: list[list[float]] = []
        for offset in range(0, len(chunks), _EMBED_BATCH_SIZE):
            batch_texts = [item.text for item in chunks[offset : offset + _EMBED_BATCH_SIZE]]
            vectors.extend(await _embed_batch(embedder, batch_texts))

        document = RagDocument(
            collection_id=collection_id,
            title=str(chunk_metadata.get("title") or source or "Untitled document"),
            source=source or None,
            content_type=str(chunk_metadata.get("content_type", "text/plain")),
            content_hash=content_hash,
            byte_size=len(content.encode("utf-8")),
            status="ready",
            error_message=None,
            chunk_count=len(chunks),
            metadata_=chunk_metadata,
        )
        session.add(document)
        await session.flush()

        rows = [
            RagChunk(
                document_id=document.id,
                collection_id=collection_id,
                chunk_index=item.index,
                content=item.text,
                token_count=max(1, len(item.text) // 4),
                metadata_={"start_char": item.start_char, "end_char": item.end_char},
            )
            for item in chunks
        ]
        session.add_all(rows)
        await session.flush()

        vector_payloads = [
            {
                "document_id": document.id,
                "chunk_id": row.id,
                "chunk_index": item.index,
                "source": source,
                "content": item.text,
                "metadata": {
                    **chunk_metadata,
                    "start_char": item.start_char,
                    "end_char": item.end_char,
                },
                "vector": vector,
            }
            for row, item, vector in zip(rows, chunks, vectors, strict=True)
        ]
        for row in rows:
            row.vector_id = f"aigw:rag:{collection_id}:{row.id}"

        collection.document_count += 1
        collection.chunk_count += len(chunks)
        # Keep Postgres uncommitted until Redis confirms its batch write; Redis failure
        # makes Database.session roll back the metadata and chunk rows.
        if vector_payloads:
            try:
                await store.upsert(collection_id, vector_payloads)
            except Exception:
                try:
                    await store.delete_document(collection_id, document.id)
                except Exception:
                    logger.exception("Could not clean up partial RAG vectors for %s", document.id)
                raise
        logger.info("Ingested RAG document %s with %d chunks", document.id, len(chunks))
        return document


async def delete_document(db: Any, store: Any, *, collection_id: str, document_id: str) -> bool:
    async with db.session() as session:
        document = await session.scalar(
            select(RagDocument).where(
                RagDocument.id == document_id, RagDocument.collection_id == collection_id
            )
        )
        if document is None:
            return False
        collection = await session.get(RagCollection, collection_id)
        await store.delete_document(collection_id, document_id)
        if collection is not None:
            collection.document_count = max(0, collection.document_count - 1)
            collection.chunk_count = max(0, collection.chunk_count - document.chunk_count)
        await session.delete(document)
        return True
