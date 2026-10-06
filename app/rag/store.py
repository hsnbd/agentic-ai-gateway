"""Redis Stack vector storage for RAG chunks."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import struct
from collections.abc import Mapping, Sequence
from typing import Any

from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from app.core.errors import InvalidRequestError

logger = logging.getLogger(__name__)


class RagVectorStore:
    """Collection-scoped RediSearch HNSW index management."""

    def __init__(self, redis: Any, embedder_dims: int) -> None:
        self.redis = redis
        self.embedder_dims = embedder_dims
        self.available = True
        self._indexes: set[str] = set()
        # Standard BM25: weighs a rare term (an error code) above repeated
        # common words. Legacy "BM25" does not, and ranks keyword hits poorly.
        self._scorer = "BM25STD"
        #: Per collection, the metadata keys indexed for filtering (declare_fields).
        self._fields: dict[str, tuple[str, ...]] = {}

    @staticmethod
    def _index_name(collection_id: str) -> str:
        return f"aigw:rag:{collection_id}:idx"

    @staticmethod
    def _prefix(collection_id: str) -> str:
        return f"aigw:rag:{collection_id}:"

    def declare_fields(self, collection_id: str, fields: Sequence[str]) -> None:
        """Metadata keys this collection indexes for filtering (fixed at creation)."""
        self._fields[collection_id] = tuple(fields)

    async def ensure_index(self, collection_id: str, dims: int) -> None:
        name = self._index_name(collection_id)
        if name in self._indexes:
            return
        try:
            await self.redis.ft(name).info()
            self._indexes.add(name)
            self.available = True
            return
        except Exception:
            pass

        schema = (
            TagField("document_id"),
            TagField("source"),
            *(TagField(_meta_field(name)) for name in self._fields.get(collection_id, ())),
            TextField("content"),
            TextField("chunk_id"),
            NumericField("chunk_index"),
            VectorField(
                "vector",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": dims,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
            ),
        )
        try:
            await self.redis.ft(name).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self._prefix(collection_id)], index_type=IndexType.HASH
                ),
            )
            self._indexes.add(name)
            self.available = True
        except Exception as exc:
            self.available = False
            logger.warning("RediSearch unavailable for RAG collection %s: %s", collection_id, exc)
            return
        # A recreated index picks up vectors already stored under its prefix.
        await self._await_backfill(collection_id)

    async def upsert(
        self, collection_id: str, chunks_with_vectors: Sequence[Mapping[str, Any]]
    ) -> None:
        if not self.available:
            raise RuntimeError("Redis Search is unavailable; cannot store RAG vectors")
        if not chunks_with_vectors:
            return
        first_vector = chunks_with_vectors[0]["vector"]
        dimensions = (
            len(first_vector) // 4 if isinstance(first_vector, bytes) else len(first_vector)
        )
        await self.ensure_index(collection_id, dimensions)
        if not self.available:
            raise RuntimeError("Redis Search is unavailable; cannot store RAG vectors")
        for offset in range(0, len(chunks_with_vectors), 128):
            pipe = self.redis.pipeline(transaction=True)
            for item in chunks_with_vectors[offset : offset + 128]:
                chunk_id = str(item["chunk_id"])
                vector = item["vector"]
                if not isinstance(vector, bytes):
                    vector = struct.pack(f"{len(vector)}f", *vector)
                fields = {
                    "chunk_id": chunk_id,
                    "document_id": str(item["document_id"]),
                    "source": str(item.get("source") or ""),
                    "content": str(item["content"]),
                    "chunk_index": int(item.get("chunk_index", 0)),
                    "metadata": json.dumps(item.get("metadata", {}), ensure_ascii=False),
                    "vector": vector,
                }
                for name in self._fields.get(collection_id, ()):
                    value = (item.get("metadata") or {}).get(name)
                    if value is not None:
                        values = value if isinstance(value, list) else [value]
                        fields[_meta_field(name)] = ",".join(str(entry) for entry in values)
                pipe.hset(f"{self._prefix(collection_id)}{chunk_id}", mapping=fields)
            await pipe.execute()

    def _filter_clause(self, collection_id: str, filters: Mapping[str, str] | None) -> str:
        declared = self._fields.get(collection_id, ())
        unsupported = set(filters or {}) - set(BUILT_IN_FILTERS) - set(declared)
        if unsupported:
            allowed = ", ".join((*BUILT_IN_FILTERS, *declared))
            raise InvalidRequestError(
                f"Unsupported RAG filter fields: {', '.join(sorted(unsupported))} "
                f"(this collection can filter by: {allowed})"
            )
        clauses = []
        for field, value in (filters or {}).items():
            indexed = field if field in BUILT_IN_FILTERS else _meta_field(field)
            clauses.append(f"@{indexed}:{{{_escape_tag(value)}}}")
        return " ".join(clauses) if clauses else "*"

    async def search(
        self,
        collection_id: str,
        vector: Sequence[float],
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Nearest neighbours by cosine similarity (score 1.0 = identical)."""
        if not self.available or k <= 0:
            return []
        await self.ensure_index(collection_id, len(vector))
        if not self.available:
            return []
        base_query = self._filter_clause(collection_id, filters)
        query = (
            Query(f"({base_query})=>[KNN {k} @vector $query_vector AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields(*_RETURN_FIELDS, "vector_distance")
            .paging(0, k)
            .dialect(2)
        )
        packed = struct.pack(f"{len(vector)}f", *vector)
        results = await self._with_index(
            collection_id,
            len(vector),
            lambda index: index.search(query, query_params={"query_vector": packed}),
        )
        return [
            (
                _text(_value(row, "chunk_id", "")),
                1.0 - float(_value(row, "vector_distance", 1.0)),
                _row_metadata(row),
            )
            for row in results.docs
        ]

    async def text_search(
        self,
        collection_id: str,
        text: str,
        k: int,
        dims: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Keyword (BM25) matches on chunk text: finds exact codes and names that
        embeddings blur, such as an error code or a product number."""
        terms = sorted({term.lower() for term in re.findall(r"[A-Za-z0-9]+", text)})
        if not self.available or k <= 0 or not terms:
            return []
        await self.ensure_index(collection_id, dims)
        if not self.available:
            return []
        base_query = self._filter_clause(collection_id, filters)
        prefix = "" if base_query == "*" else f"{base_query} "

        def query() -> Query:
            return (
                Query(f"{prefix}@content:({'|'.join(terms)})")
                .scorer(self._scorer)
                .with_scores()
                .return_fields(*_RETURN_FIELDS)
                .paging(0, k)
                .dialect(2)
            )

        try:
            results = await self._with_index(
                collection_id, dims, lambda index: index.search(query())
            )
        except Exception as exc:
            if self._scorer == "BM25" or "scorer" not in str(exc).lower():
                raise
            # BM25STD needs Redis Stack 7.4+; older servers only have legacy BM25.
            logger.warning("Redis Search has no BM25STD scorer; using BM25")
            self._scorer = "BM25"
            results = await self._with_index(
                collection_id, dims, lambda index: index.search(query())
            )
        return [
            (
                _text(_value(row, "chunk_id", "")),
                float(_value(row, "score", 0.0)),
                _row_metadata(row),
            )
            for row in results.docs
        ]

    async def get_vectors(
        self, collection_id: str, chunk_ids: Sequence[str]
    ) -> dict[str, list[float]]:
        """Read raw vectors by chunk id. Works without the search module."""
        if not chunk_ids:
            return {}
        pipe = self.redis.pipeline(transaction=False)
        for chunk_id in chunk_ids:
            pipe.hget(f"{self._prefix(collection_id)}{chunk_id}", "vector")
        try:
            raw_values = await pipe.execute()
        except Exception as exc:
            logger.warning("Could not read RAG vectors for %s: %s", collection_id, exc)
            return {}

        vectors: dict[str, list[float]] = {}
        for chunk_id, raw in zip(chunk_ids, raw_values, strict=True):
            if not isinstance(raw, bytes) or len(raw) < 4:
                continue
            vectors[chunk_id] = list(struct.unpack(f"{len(raw) // 4}f", raw))
        return vectors

    async def delete_document(self, collection_id: str, document_id: str) -> int:
        if not self.available:
            return 0
        index = self.redis.ft(self._index_name(collection_id))
        keys: list[str] = []
        offset = 0
        page_size = 1000
        try:
            while True:
                query = (
                    Query(f"@document_id:{{{_escape_tag(document_id)}}}")
                    .paging(offset, page_size)
                    .dialect(2)
                )
                results = await index.search(query)
                keys.extend(row.id for row in results.docs)
                offset += len(results.docs)
                if offset >= results.total or not results.docs:
                    break
        except Exception as exc:
            if not _missing_index(exc):
                raise
            # Without the index, find the document's vectors by key prefix.
            self._indexes.discard(self._index_name(collection_id))
            keys = await self._keys_for_document(collection_id, document_id)
        deleted = 0
        for offset in range(0, len(keys), page_size):
            deleted += int(await self.redis.delete(*keys[offset : offset + page_size]))
        return deleted

    async def _with_index(self, collection_id: str, dims: int, run: Any) -> Any:
        """Run ``run(index)``, recreating the index once if it has disappeared.

        As with the semantic cache, the index can vanish at runtime (FLUSHALL, a
        Redis restart without persistence, failover to a fresh replica), and the
        cached "exists" flag would otherwise hide that forever. Vectors still
        stored under the prefix become searchable again when it is recreated;
        vectors that were lost are restored with ``RagService.reindex``.
        """
        for attempt in (1, 2):
            try:
                return await run(self.redis.ft(self._index_name(collection_id)))
            except Exception as exc:
                if attempt == 2 or not _missing_index(exc):
                    raise
                logger.warning("RAG index for %s is missing; recreating it", collection_id)
                self._indexes.discard(self._index_name(collection_id))
                await self.ensure_index(collection_id, dims)
        return None  # pragma: no cover - the second attempt always returns or raises

    async def _await_backfill(self, collection_id: str) -> None:
        """Wait (up to two seconds) for a recreated index to pick up stored vectors.

        Redis indexes existing hashes in the background; searching straight
        away would return a partial result that looks like a relevance problem.
        """
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(2.0):
                while True:
                    info = await self.redis.ft(self._index_name(collection_id)).info()
                    if str(_value(info, "indexing", 0)) in ("0", "0.0"):
                        return
                    await asyncio.sleep(0.02)

    async def _keys_for_document(self, collection_id: str, document_id: str) -> list[str]:
        keys: list[str] = []
        async for key in self.redis.scan_iter(match=f"{self._prefix(collection_id)}*", count=500):
            name = _text(key)
            if _text(await self.redis.hget(name, "document_id") or b"") == document_id:
                keys.append(name)
        return keys

    async def index_stats(self, collection_id: str) -> dict[str, Any]:
        """Whether the collection's index exists and how many vectors it holds."""
        try:
            info = await self.redis.ft(self._index_name(collection_id)).info()
        except Exception as exc:
            if not _missing_index(exc):
                raise
            self._indexes.discard(self._index_name(collection_id))
            return {"index_exists": False, "indexed_chunks": 0}
        return {"index_exists": True, "indexed_chunks": int(_value(info, "num_docs", 0))}

    async def reset(self, collection_id: str, dims: int) -> None:
        """Drop the index and every stored vector, then create an empty index."""
        name = self._index_name(collection_id)
        try:
            await self.redis.ft(name).dropindex(delete_documents=True)
        except Exception as exc:
            if not _missing_index(exc):
                raise
        stale = [key async for key in self.redis.scan_iter(match=f"{self._prefix(collection_id)}*")]
        for offset in range(0, len(stale), 1000):
            await self.redis.delete(*stale[offset : offset + 1000])
        self._indexes.discard(name)
        self.available = True
        await self.ensure_index(collection_id, dims)

    async def drop_collection(self, collection_id: str) -> None:
        name = self._index_name(collection_id)
        if self.available:
            try:
                await self.redis.ft(name).dropindex(delete_documents=True)
            except Exception as exc:
                logger.warning("Could not drop RAG index %s: %s", name, exc)
        self._indexes.discard(name)


#: Hash fields every search returns.
_RETURN_FIELDS = (
    "chunk_id",
    "document_id",
    "source",
    "content",
    "chunk_index",
    "metadata",
    "vector",
)

#: Fields every collection can be filtered by.
BUILT_IN_FILTERS = ("document_id", "source")


def _meta_field(name: str) -> str:
    return f"meta_{name}"


def _row_metadata(row: Any) -> dict[str, Any]:
    """A search row's metadata, with its content, source, and vector."""
    raw_metadata = _value(row, "metadata", "{}")
    if isinstance(raw_metadata, bytes):
        raw_metadata = raw_metadata.decode("utf-8")
    try:
        metadata: dict[str, Any] = json.loads(raw_metadata)
    except (TypeError, json.JSONDecodeError):
        metadata = {}
    row_vector = _value(row, "vector", b"")
    if isinstance(row_vector, bytes):
        metadata["_vector"] = list(struct.unpack(f"{len(row_vector) // 4}f", row_vector))
    metadata.update(
        document_id=_text(_value(row, "document_id", "")),
        source=_text(_value(row, "source", "")),
        content=_text(_value(row, "content", "")),
        chunk_index=int(_value(row, "chunk_index", 0)),
    )
    return metadata


def _missing_index(exc: Exception) -> bool:
    message = str(exc).lower()
    return "no such index" in message or "unknown index name" in message


def _escape_tag(value: str) -> str:
    special = set(r",.<>{}[]\"':;!@#$%^&*-=+~|\\/()?:")
    return "".join(f"\\{char}" if char in special or char.isspace() else char for char in value)


def _value(row: Any, name: str, default: Any) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
