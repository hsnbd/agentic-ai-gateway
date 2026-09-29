"""Redis Stack-backed semantic response cache with tenant/model isolation."""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import time
import uuid
from typing import Any, Protocol

from app.config.settings import Settings
from app.core.pipeline import RequestContext
from app.core.schemas import ChatResponse, FinishReason, Role

logger = logging.getLogger(__name__)

_ENTRY_PREFIX = "aigw:cache:entry:"
_INDEX_PREFIX = "aigw:cache:"
_STATS_PREFIX = "aigw:cache:stats:"
_TAIL_MESSAGES = 4


class Embedder(Protocol):
    @property
    def dimensions(self) -> int | None: ...

    async def embed(self, text: str) -> list[float] | None: ...


class SemanticCache:
    """Best-effort vector cache; unavailable Redis search never blocks requests."""

    def __init__(self, redis: Any, embedder: Embedder, settings: Settings) -> None:
        self.redis = redis
        self.embedder = embedder
        self.settings = settings
        self.available = True
        self._index_ready = False

    async def ensure_index(self) -> None:
        """Create the vector index once an embedding dimension is known."""
        if not self.available or self._index_ready:
            return
        dimensions = self.embedder.dimensions
        if dimensions is None:
            return
        try:
            await self.redis.execute_command(
                "FT.CREATE",
                self.settings.cache_index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                _INDEX_PREFIX,
                "SCHEMA",
                "namespace",
                "TAG",
                "created_at",
                "NUMERIC",
                "embedding",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(dimensions),
                "DISTANCE_METRIC",
                "COSINE",
            )
            self._index_ready = True
        except Exception as exc:
            message = str(exc).lower()
            if "index already exists" in message:
                self._index_ready = True
                return
            missing_search = any(
                token in message for token in ("unknown command", "unknown subcommand", "module")
            )
            if missing_search:
                self.available = False
                logger.warning(
                    "Redis Stack search is unavailable; semantic caching is disabled: %s", exc
                )
            else:
                logger.warning(
                    "Unable to create semantic cache index; cache is skipped", exc_info=True
                )

    def build_namespace(self, ctx: RequestContext) -> str:
        """Hash request semantics and tenant identity into a strict search partition."""
        registry = ctx.state.registry
        model = registry.resolve_alias(ctx.request.model)
        tenant_id = ctx.key_id or getattr(ctx.virtual_key, "id", None) or ctx.team_id or "unscoped"
        system_prompt_hash = hashlib.sha256(
            ctx.request.system_prompt().encode("utf-8")
        ).hexdigest()
        tool_payload = [tool.model_dump(mode="json") for tool in ctx.request.tools]
        tool_hash = hashlib.sha256(
            json.dumps(tool_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        json_mode = bool(getattr(ctx.request, "json_mode", False))
        namespace_data = {
            "model": model,
            "tenant": str(tenant_id),
            "system_prompt": system_prompt_hash,
            "temperature_bucket": ctx.request.temperature,
            "top_p_bucket": ctx.request.top_p,
            "response_format": ctx.request.response_format,
            "tool_choice": (
                ctx.request.tool_choice.model_dump(mode="json")
                if ctx.request.tool_choice is not None
                else None
            ),
            "parallel_tool_calls": ctx.request.parallel_tool_calls,
            "json_mode": json_mode,
            "tools": tool_hash,
            "max_tokens": ctx.request.max_tokens,
            "stop": ctx.request.stop,
            "seed": ctx.request.seed,
            "presence_penalty": ctx.request.presence_penalty,
            "frequency_penalty": ctx.request.frequency_penalty,
            "n": ctx.request.n,
        }
        encoded = json.dumps(namespace_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    async def lookup(self, ctx: RequestContext) -> ChatResponse | None:
        """Find the closest eligible completion without allowing cross-namespace hits."""
        try:
            if not self.available or not self._request_is_eligible(ctx):
                return None
            query_text = self._embedding_text(ctx)
            vector = await self.embedder.embed(query_text)
            if vector is None:
                return None
            await self.ensure_index()
            if not self.available or not self._index_ready:
                return None

            namespace = self.build_namespace(ctx)
            packed_vector = struct.pack(f"<{len(vector)}f", *vector)
            result = await self._search(namespace, packed_vector)
            if result is None:
                return None
            matches = self._search_matches(result)
            best: tuple[float, dict[str, str]] | None = None
            for fields in matches:
                distance_text = fields.get("distance")
                if distance_text is None:
                    continue
                distance = float(distance_text)
                if best is None or distance < best[0]:
                    best = distance, fields

            if best is None:
                await self._increment_counter("misses")
                return None
            similarity = 1.0 - best[0]
            if similarity < self.settings.cache_similarity_threshold:
                await self._increment_counter("misses")
                return None

            response_json = best[1].get("response")
            if response_json is None:
                await self._increment_counter("misses")
                return None
            cached = ChatResponse.model_validate_json(response_json)
            response = cached.model_copy(
                update={
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "created": int(time.time()),
                    "cache_hit": True,
                    "cache_similarity": similarity,
                }
            )
            ctx.cache_hit = True
            ctx.cache_similarity = similarity
            ctx.cost_saved_usd = cached.cost_usd or 0.0
            await self._increment_counter("hits")
            return response
        except Exception:
            logger.warning("Semantic cache lookup failed; continuing without cache", exc_info=True)
            return None

    async def _search(self, namespace: str, packed_vector: bytes) -> Any | None:
        """Run the KNN search, rebuilding the index if it has disappeared.

        The index is normally created once at startup, but it can vanish under
        the gateway at runtime: `FLUSHALL`, a Redis restart without persistence,
        or failover to a replica that never had it. Because `ensure_index`
        short-circuits on a cached `_index_ready` flag, nothing would ever
        recreate it, and the cache would report a miss for every request until
        someone restarted the gateway — a silent, permanent loss of the feature
        rather than a visible failure.

        Recreating the index also backfills: entries written while it was gone
        match its key prefix, so they become searchable again rather than
        lingering as dead keys until their TTL expires.
        """
        for attempt in (1, 2):
            try:
                return await self.redis.execute_command(
                    "FT.SEARCH",
                    self.settings.cache_index_name,
                    f"@namespace:{{{namespace}}}=>[KNN 5 @embedding $vector AS distance]",
                    "PARAMS",
                    "2",
                    "vector",
                    packed_vector,
                    "SORTBY",
                    "distance",
                    "LIMIT",
                    "0",
                    "5",
                    "DIALECT",
                    "2",
                )
            except Exception as exc:
                if attempt == 2 or "no such index" not in str(exc).lower():
                    raise
                logger.warning(
                    "Semantic cache index %s is missing; recreating it",
                    self.settings.cache_index_name,
                )
                self._index_ready = False
                await self.ensure_index()
                if not self._index_ready:
                    return None
        return None

    async def store(self, ctx: RequestContext, response: ChatResponse) -> None:
        """Store only complete, ordinary assistant responses in the request namespace."""
        try:
            if not self.available or not self._request_is_eligible(ctx):
                return
            if any(choice.message.tool_calls for choice in response.choices):
                return
            if not response.choices or any(
                not choice.message.text().strip() for choice in response.choices
            ):
                return
            if any(choice.finish_reason != FinishReason.STOP for choice in response.choices):
                return

            vector = await self.embedder.embed(self._embedding_text(ctx))
            if vector is None:
                return
            await self.ensure_index()
            if not self.available or not self._index_ready:
                return

            namespace = self.build_namespace(ctx)
            key = f"{_ENTRY_PREFIX}{namespace}:{uuid.uuid4().hex}"
            await self.redis.hset(
                key,
                mapping={
                    "namespace": namespace,
                    "created_at": str(time.time()),
                    "embedding": struct.pack(f"<{len(vector)}f", *vector),
                    "response": response.model_dump_json(),
                },
            )
            ttl = ctx.request.cache_ttl or self.settings.cache_ttl_seconds
            await self.redis.expire(key, ttl)
            ctx.cache_key = key
        except Exception:
            logger.warning("Semantic cache store failed; continuing without cache", exc_info=True)

    async def invalidate(self, namespace: str | None = None) -> int:
        """Delete cache entries globally or from a single namespace."""
        if not self.available:
            return 0
        try:
            pattern = f"{_ENTRY_PREFIX}{namespace}:*" if namespace else f"{_ENTRY_PREFIX}*"
            keys = [self._as_text(key) async for key in self.redis.scan_iter(match=pattern)]
            if not keys:
                return 0
            return int(await self.redis.delete(*keys))
        except Exception:
            logger.warning("Semantic cache invalidation failed", exc_info=True)
            return 0

    async def stats(self) -> dict[str, int]:
        """Return entry and Redis-persisted hit/miss counts."""
        if not self.available:
            return {"entries": 0, "hits": 0, "misses": 0}
        try:
            entries = 0
            async for _ in self.redis.scan_iter(match=f"{_ENTRY_PREFIX}*"):
                entries += 1
            hits = self._to_int(await self.redis.get(f"{_STATS_PREFIX}hits"))
            misses = self._to_int(await self.redis.get(f"{_STATS_PREFIX}misses"))
            return {"entries": entries, "hits": hits, "misses": misses}
        except Exception:
            logger.warning("Semantic cache stats unavailable", exc_info=True)
            return {"entries": 0, "hits": 0, "misses": 0}

    def _request_is_eligible(self, ctx: RequestContext) -> bool:
        request = ctx.request
        if not self.settings.cache_enabled or request.no_cache:
            return False
        if (
            request.temperature is not None
            and request.temperature > self.settings.cache_max_temperature
        ):
            return False
        if request.tools or (
            request.stream and any(message.tool_calls for message in request.messages)
        ):
            return False
        if any(message.tool_calls or message.role == Role.TOOL for message in request.messages):
            return False
        return bool(request.messages and request.messages[-1].role == Role.USER)

    @staticmethod
    def _embedding_text(ctx: RequestContext) -> str:
        # The last user prompt plus only four prior non-system turns keeps old context from
        # diluting the vector while still retaining the immediate conversational context.
        recent = [
            message
            for message in ctx.request.messages[:-1]
            if message.role not in (Role.SYSTEM, Role.TOOL)
        ][-_TAIL_MESSAGES:]
        recent.append(ctx.request.messages[-1])
        return "\n".join(f"{message.role.value}: {message.text()}" for message in recent)

    async def _increment_counter(self, name: str) -> None:
        try:
            await self.redis.incr(f"{_STATS_PREFIX}{name}")
        except Exception:
            logger.warning("Unable to increment semantic cache %s counter", name, exc_info=True)

    @classmethod
    def _search_matches(cls, result: Any) -> list[dict[str, str]]:
        """Normalise an FT.SEARCH reply into a list of field maps.

        The reply shape depends on the negotiated RESP protocol, and the
        difference is easy to miss because the RESP3 form still "works" —
        it just parses as zero matches, so the cache degrades into a
        write-only store that never serves a hit.

        RESP2 is a flat list: ``[total, key1, [f, v, ...], key2, [...]]``.
        RESP3 is a map: ``{"total_results": n, "results": [{"id": ...,
        "extra_attributes": {...}}, ...]}``.
        """
        if isinstance(result, dict):
            return cls._resp3_matches(result)
        if not isinstance(result, (list, tuple)):
            return []
        matches: list[dict[str, str]] = []
        for index in range(1, len(result), 2):
            if index + 1 >= len(result):
                break
            raw_fields = result[index + 1]
            fields: dict[str, str] = {}
            if isinstance(raw_fields, (list, tuple)):
                for field_index in range(0, len(raw_fields) - 1, 2):
                    fields[cls._as_text(raw_fields[field_index])] = cls._as_text(
                        raw_fields[field_index + 1]
                    )
            matches.append(fields)
        return matches

    @classmethod
    def _resp3_matches(cls, result: dict[Any, Any]) -> list[dict[str, str]]:
        raw_results = cls._map_get(result, "results")
        if not isinstance(raw_results, (list, tuple)):
            return []
        matches: list[dict[str, str]] = []
        for entry in raw_results:
            if not isinstance(entry, dict):
                continue
            attributes = cls._map_get(entry, "extra_attributes")
            if not isinstance(attributes, dict):
                continue
            matches.append(
                {cls._as_text(k): cls._as_text(v) for k, v in attributes.items()}
            )
        return matches

    @staticmethod
    def _map_get(mapping: dict[Any, Any], key: str) -> Any:
        """Look a key up whether the client decoded responses or not."""
        if key in mapping:
            return mapping[key]
        return mapping.get(key.encode())

    @staticmethod
    def _as_text(value: Any) -> str:
        """Best-effort text conversion that can never raise.

        FT.SEARCH returns *every* stored field, including the embedding itself,
        which is packed float32 and therefore not valid UTF-8. A strict decode
        here raises mid-parse and takes the whole lookup with it — and because
        cache errors are deliberately non-fatal, the failure surfaces only as a
        permanent miss rather than an error the caller ever sees. Lossy decoding
        is safe because the binary fields are never read as text; only the id,
        score, and payload are.
        """
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @classmethod
    def _to_int(cls, value: Any) -> int:
        return int(cls._as_text(value)) if value is not None else 0
