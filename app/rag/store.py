"""Redis Stack vector storage for RAG chunks."""

from __future__ import annotations

import json
import logging
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

    @staticmethod
    def _index_name(collection_id: str) -> str:
        return f"aigw:rag:{collection_id}:idx"

    @staticmethod
    def _prefix(collection_id: str) -> str:
        return f"aigw:rag:{collection_id}:"

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
                pipe.hset(f"{self._prefix(collection_id)}{chunk_id}", mapping=fields)
            await pipe.execute()

    async def search(
        self,
        collection_id: str,
        vector: Sequence[float],
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        if not self.available or k <= 0:
            return []
        await self.ensure_index(collection_id, len(vector))
        if not self.available:
            return []

        unsupported = set(filters or {}) - {"document_id", "source"}
        if unsupported:
            raise InvalidRequestError(
                f"Unsupported RAG filter fields: {', '.join(sorted(unsupported))}"
            )
        clauses = [f"@{field}:{{{_escape_tag(value)}}}" for field, value in (filters or {}).items()]
        base_query = " ".join(clauses) if clauses else "*"
        query_text = f"({base_query})=>[KNN {k} @vector $query_vector AS vector_distance]"
        query = (
            Query(query_text)
            .sort_by("vector_distance")
            .return_fields(
                "chunk_id",
                "document_id",
                "source",
                "content",
                "chunk_index",
                "metadata",
                "vector",
                "vector_distance",
            )
            .paging(0, k)
            .dialect(2)
        )
        packed = struct.pack(f"{len(vector)}f", *vector)
        results = await self.redis.ft(self._index_name(collection_id)).search(
            query, query_params={"query_vector": packed}
        )
        found: list[tuple[str, float, dict[str, Any]]] = []
        for row in results.docs:
            distance = float(_value(row, "vector_distance", 1.0))
            raw_metadata = _value(row, "metadata", "{}")
            if isinstance(raw_metadata, bytes):
                raw_metadata = raw_metadata.decode("utf-8")
            try:
                metadata = json.loads(raw_metadata)
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
            found.append((_text(_value(row, "chunk_id", "")), 1.0 - distance, metadata))
        return found

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
        deleted = 0
        for offset in range(0, len(keys), page_size):
            deleted += int(await self.redis.delete(*keys[offset : offset + page_size]))
        return deleted

    async def drop_collection(self, collection_id: str) -> None:
        name = self._index_name(collection_id)
        if self.available:
            try:
                await self.redis.ft(name).dropindex(delete_documents=True)
            except Exception as exc:
                logger.warning("Could not drop RAG index %s: %s", name, exc)
        self._indexes.discard(name)


def _escape_tag(value: str) -> str:
    special = set(r",.<>{}[]\"':;!@#$%^&*-=+~|\\/()?:")
    return "".join(f"\\{char}" if char in special or char.isspace() else char for char in value)


def _value(row: Any, name: str, default: Any) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
