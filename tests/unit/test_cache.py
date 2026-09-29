"""Semantic cache unit tests using an in-memory Redis Stack substitute."""

from __future__ import annotations

import fnmatch
import math
import re
import struct
from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.cache.embedder import CacheEmbedder
from app.cache.semantic import SemanticCache
from app.config.settings import Settings
from app.core.pipeline import RequestContext
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    Choice,
    EmbeddingResponse,
    EmbeddingVector,
    FinishReason,
    FunctionDef,
    Message,
    Role,
    ToolCall,
    ToolDef,
)
from app.providers.base import Capabilities, Deployment


class FakeState:
    def __init__(self) -> None:
        self.registry = FakeRegistry()


class FakeRegistry:
    @staticmethod
    def resolve_alias(model: str) -> str:
        return model


class StubEmbedder:
    dimensions = 2

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}
        self.calls = 0
        self.fail = False

    async def embed(self, text: str) -> list[float] | None:
        self.calls += 1
        if self.fail:
            return None
        return self.vectors.get(text, [1.0, 0.0])


class FakeRedis:
    def __init__(self, *, search_available: bool = True) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.values: dict[str, int] = {}
        self.ttls: dict[str, int] = {}
        self.index_exists = False
        self.search_available = search_available
        self.raise_search = False
        self.raise_store = False

    async def execute_command(self, *args: Any) -> list[Any] | str:
        command = str(args[0]).upper()
        if command == "FT.CREATE":
            if not self.search_available:
                raise RuntimeError("unknown command 'FT.CREATE' - Redis search module missing")
            if self.index_exists:
                raise RuntimeError("Index already exists")
            self.index_exists = True
            return "OK"
        if command != "FT.SEARCH":
            raise AssertionError(f"Unexpected Redis command {command}")
        if self.raise_search:
            raise ConnectionError("Redis is unavailable")
        namespace_match = re.search(r"@namespace:\{([0-9a-f]+)\}", str(args[2]))
        assert namespace_match is not None
        namespace = namespace_match.group(1)
        query_vector_bytes = next(value for value in args if isinstance(value, bytes))
        query_vector = struct.unpack("<2f", query_vector_bytes)
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for key, fields in self.entries.items():
            if fields["namespace"] != namespace:
                continue
            candidate = struct.unpack("<2f", fields["embedding"])
            dot = sum(a * b for a, b in zip(query_vector, candidate, strict=True))
            norm = math.sqrt(sum(value * value for value in query_vector)) * math.sqrt(
                sum(value * value for value in candidate)
            )
            distance = 1.0 - dot / norm
            scored.append((distance, key, fields))
        scored.sort(key=lambda item: item[0])
        result: list[Any] = [len(scored)]
        for distance, key, fields in scored[:5]:
            result.extend(
                [
                    key.encode(),
                    [
                        b"distance",
                        str(distance).encode(),
                        b"response",
                        fields["response"].encode(),
                    ],
                ]
            )
        return result

    async def hset(self, key: str, *, mapping: dict[str, Any]) -> int:
        if self.raise_store:
            raise ConnectionError("Redis is unavailable")
        self.entries[key] = dict(mapping)
        return 1

    async def expire(self, key: str, ttl: int) -> bool:
        self.ttls[key] = ttl
        return key in self.entries

    async def incr(self, key: str) -> int:
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    async def get(self, key: str) -> int | None:
        return self.values.get(key)

    async def scan_iter(self, *, match: str) -> AsyncIterator[str]:
        for key in list(self.entries):
            if fnmatch.fnmatch(key, match):
                yield key

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            deleted += int(self.entries.pop(key, None) is not None)
        return deleted


def make_context(**request_values: Any) -> RequestContext:
    request_values.setdefault("model", "model-a")
    request_values.setdefault("messages", [Message(role=Role.USER, content="prompt")])
    return RequestContext(
        request=ChatRequest(**request_values),
        state=FakeState(),  # type: ignore[arg-type]
        key_id="key-a",
    )


def make_response(
    *,
    content: str = "cached answer",
    finish_reason: FinishReason = FinishReason.STOP,
    tool_calls: list[ToolCall] | None = None,
    cost_usd: float = 0.012,
) -> ChatResponse:
    return ChatResponse(
        model="model-a",
        id="chatcmpl-original",
        created=1,
        cost_usd=cost_usd,
        choices=[
            Choice(
                message=Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls or []),
                finish_reason=finish_reason,
            )
        ],
    )


def make_cache(
    *, threshold: float = 0.95, search_available: bool = True
) -> tuple[SemanticCache, FakeRedis, StubEmbedder]:
    redis = FakeRedis(search_available=search_available)
    embedder = StubEmbedder()
    settings = Settings(
        cache_similarity_threshold=threshold,
        cache_embedding_dimensions=2,
        cache_embedding_model="embed-model",
    )
    return SemanticCache(redis, embedder, settings), redis, embedder


async def store_answer(cache: SemanticCache, ctx: RequestContext) -> ChatResponse:
    response = make_response()
    await cache.store(ctx, response)
    return response


@pytest.mark.asyncio
async def test_cosine_distance_is_converted_to_similarity_and_hit_gets_fresh_id() -> None:
    cache, redis, embedder = make_cache(threshold=0.95)
    ctx = make_context()
    original = await store_answer(cache, ctx)
    embedder.vectors["user: prompt"] = [0.98, math.sqrt(1.0 - 0.98**2)]

    hit = await cache.lookup(make_context())

    assert hit is not None
    assert hit.cache_hit is True
    assert hit.cache_similarity == pytest.approx(0.98)
    assert hit.id != original.id
    assert hit.created >= original.created
    assert ctx.cache_key is not None
    assert redis.ttls[ctx.cache_key] == Settings().cache_ttl_seconds
    lookup_ctx = make_context()
    assert await cache.lookup(lookup_ctx) is not None
    assert lookup_ctx.cache_hit is True
    assert lookup_ctx.cost_saved_usd == original.cost_usd
    assert len(redis.entries) == 1


@pytest.mark.asyncio
async def test_similarity_just_below_threshold_is_a_miss() -> None:
    cache, _, embedder = make_cache(threshold=0.95)
    await store_answer(cache, make_context())
    embedder.vectors["user: prompt"] = [0.949, math.sqrt(1.0 - 0.949**2)]

    assert await cache.lookup(make_context()) is None


@pytest.mark.asyncio
async def test_models_keys_and_system_prompts_never_cross_match() -> None:
    cache, _, _ = make_cache()
    base = make_context(messages=[
        Message(role=Role.SYSTEM, content="assistant one"),
        Message(role=Role.USER, content="prompt"),
    ])
    await store_answer(cache, base)
    other_model = make_context(
        model="model-b",
        messages=[
            Message(role=Role.SYSTEM, content="assistant one"),
            Message(role=Role.USER, content="prompt"),
        ],
    )
    other_model.key_id = base.key_id
    other_key = make_context(
        messages=[
            Message(role=Role.SYSTEM, content="assistant one"),
            Message(role=Role.USER, content="prompt"),
        ],
    )
    other_key.key_id = "key-b"
    other_system = make_context(
        messages=[
            Message(role=Role.SYSTEM, content="assistant two"),
            Message(role=Role.USER, content="prompt"),
        ],
    )
    other_system.key_id = base.key_id

    assert len(
        {
            cache.build_namespace(base),
            cache.build_namespace(other_model),
            cache.build_namespace(other_key),
            cache.build_namespace(other_system),
        }
    ) == 4
    assert await cache.lookup(other_model) is None
    assert await cache.lookup(other_key) is None
    assert await cache.lookup(other_system) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"no_cache": True},
        {"temperature": 0.9},
        {"tools": [ToolDef(function=FunctionDef(name="lookup"))]},
        {
            "messages": [
                Message(
                    role=Role.ASSISTANT,
                    content=None,
                    tool_calls=[ToolCall(name="lookup")],
                ),
                Message(role=Role.USER, content="prompt"),
            ]
        },
    ],
    ids=["no-cache", "high-temperature", "tools", "tool-calls"],
)
async def test_ineligible_requests_skip_lookup(overrides: dict[str, Any]) -> None:
    cache, redis, _ = make_cache()

    ctx = make_context(**overrides)
    assert await cache.lookup(ctx) is None
    await cache.store(ctx, make_response())
    assert not redis.index_exists
    assert not redis.entries


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        make_response(tool_calls=[ToolCall(name="lookup")]),
        make_response(content="  \n "),
        make_response(finish_reason=FinishReason.LENGTH),
    ],
    ids=["tool-calls", "empty-content", "truncated"],
)
async def test_unsafe_responses_are_not_stored(response: ChatResponse) -> None:
    cache, redis, _ = make_cache()

    await cache.store(make_context(), response)

    assert not redis.entries


@pytest.mark.asyncio
async def test_redis_errors_degrade_to_miss_and_noop_store() -> None:
    cache, redis, _ = make_cache()
    redis.raise_search = True
    assert await cache.lookup(make_context()) is None

    redis.raise_search = False
    redis.raise_store = True
    await cache.store(make_context(), make_response())
    assert not redis.entries


@pytest.mark.asyncio
async def test_missing_search_module_disables_all_cache_operations() -> None:
    cache, redis, _ = make_cache(search_available=False)

    assert await cache.lookup(make_context()) is None
    assert cache.available is False
    await cache.store(make_context(), make_response())
    assert await cache.invalidate() == 0
    assert await cache.stats() == {"entries": 0, "hits": 0, "misses": 0}
    assert not redis.entries


@pytest.mark.asyncio
async def test_stats_and_invalidation_use_entry_keys_and_redis_counters() -> None:
    cache, _, _ = make_cache()
    ctx = make_context()
    await store_answer(cache, ctx)
    await cache.lookup(make_context())
    await cache.lookup(make_context(temperature=0.1))

    assert await cache.stats() == {"entries": 1, "hits": 1, "misses": 1}
    assert await cache.invalidate(cache.build_namespace(ctx)) == 1
    assert await cache.stats() == {"entries": 0, "hits": 1, "misses": 1}


class FakeEmbeddingProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def embed(self, request: Any, deployment: Any) -> EmbeddingResponse:
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding provider unavailable")
        return EmbeddingResponse(
            model=request.model,
            data=[
                EmbeddingVector(index=index, embedding=[float(index + 1), 0.0])
                for index, _ in enumerate(request.input)
            ],
        )


class FakeEmbeddingRegistry:
    def __init__(self, provider: FakeEmbeddingProvider) -> None:
        self.provider = provider
        self.deployment = Deployment(
            id="embed-deployment",
            model_name="embed-model",
            provider="fake",
            provider_model="embed-model",
            capabilities=Capabilities(embeddings=True),
        )

    def deployments_for(self, model: str) -> list[Deployment]:
        assert model == "embed-model"
        return [self.deployment]

    def provider_for(self, deployment: Deployment) -> FakeEmbeddingProvider:
        assert deployment is self.deployment
        return self.provider


@pytest.mark.asyncio
async def test_cache_embedder_lru_and_dimension_discovery() -> None:
    provider = FakeEmbeddingProvider()
    embedder = CacheEmbedder(
        FakeEmbeddingRegistry(provider),  # type: ignore[arg-type]
        Settings(cache_embedding_model="embed-model", cache_embedding_dimensions=2),
    )

    assert await embedder.embed("same") == [1.0, 0.0]
    assert await embedder.embed("same") == [1.0, 0.0]
    assert provider.calls == 1
    assert embedder.dimensions == 2


@pytest.mark.asyncio
async def test_cache_embedder_failure_returns_none() -> None:
    provider = FakeEmbeddingProvider(fail=True)
    embedder = CacheEmbedder(
        FakeEmbeddingRegistry(provider),  # type: ignore[arg-type]
        Settings(cache_embedding_model="embed-model", cache_embedding_dimensions=2),
    )

    assert await embedder.embed("prompt") is None


def test_search_matches_parses_both_resp2_and_resp3_replies() -> None:
    """FT.SEARCH changes shape with the negotiated protocol.

    Redis Stack returns a flat list under RESP2 and a map under RESP3. Parsing
    only the list form is silently wrong rather than loud: every lookup yields
    zero matches, so the cache still stores entries but never serves one and
    the hit ratio sits at zero with no error anywhere.
    """
    resp2 = [
        1,
        b"aigw:cache:entry:ns:abc",
        [b"distance", b"0", b"response", b'{"id":"a"}'],
    ]
    resp3 = {
        b"total_results": 1,
        b"results": [
            {
                b"id": b"aigw:cache:entry:ns:abc",
                b"extra_attributes": {b"distance": b"0", b"response": b'{"id":"a"}'},
            }
        ],
    }
    decoded_resp3 = {
        "total_results": 1,
        "results": [
            {
                "id": "aigw:cache:entry:ns:abc",
                "extra_attributes": {"distance": "0", "response": '{"id":"a"}'},
            }
        ],
    }

    expected = [{"distance": "0", "response": '{"id":"a"}'}]
    assert SemanticCache._search_matches(resp2) == expected
    assert SemanticCache._search_matches(resp3) == expected
    assert SemanticCache._search_matches(decoded_resp3) == expected


def test_search_matches_tolerates_unexpected_replies() -> None:
    assert SemanticCache._search_matches(None) == []
    assert SemanticCache._search_matches({}) == []
    assert SemanticCache._search_matches({b"results": b"not-a-list"}) == []
    assert SemanticCache._search_matches({b"results": [b"not-a-map"]}) == []


def test_search_matches_survives_binary_embedding_field() -> None:
    """A non-UTF-8 field must not destroy the whole match.

    FT.SEARCH returns every stored field, and one of them is the embedding:
    packed float32 bytes that are not valid UTF-8. Decoding strictly raises
    part-way through parsing, and because semantic-cache errors are
    deliberately non-fatal the exception is swallowed and reported as a miss.
    The result is a cache that writes entries, never serves one, and logs
    nothing a dashboard would show.
    """
    binary_vector = b"\x00\x01\xa6\xff\xfe" * 8
    resp3 = {
        b"total_results": 1,
        b"results": [
            {
                b"id": b"aigw:cache:entry:ns:abc",
                b"extra_attributes": {
                    b"distance": b"0",
                    b"response": b'{"id":"a"}',
                    b"embedding": binary_vector,
                },
            }
        ],
    }

    matches = SemanticCache._search_matches(resp3)

    assert len(matches) == 1, "a binary field must not drop the match"
    # The fields we actually rely on survive intact.
    assert matches[0]["distance"] == "0"
    assert matches[0]["response"] == '{"id":"a"}'

    resp2 = [
        1,
        b"aigw:cache:entry:ns:abc",
        [b"distance", b"0", b"response", b'{"id":"a"}', b"embedding", binary_vector],
    ]
    resp2_matches = SemanticCache._search_matches(resp2)
    assert len(resp2_matches) == 1
    assert resp2_matches[0]["response"] == '{"id":"a"}'


@pytest.mark.asyncio
async def test_search_rebuilds_a_vanished_index() -> None:
    """The index can disappear while the gateway is running.

    FLUSHALL, a Redis restart without persistence, or failover to a replica
    that never had the index all remove it underneath us. `ensure_index`
    short-circuits on a cached readiness flag, so without explicit recovery
    nothing would ever recreate it: every lookup would report a miss forever
    while entries kept being written. That is invisible on a dashboard — the
    hit ratio simply sits at zero — so it must self-heal.
    """

    class VanishingRedis:
        def __init__(self) -> None:
            self.index_exists = False
            self.searches = 0
            self.created = 0

        async def execute_command(self, command: str, *args: object) -> object:
            if command == "FT.SEARCH":
                self.searches += 1
                if not self.index_exists:
                    raise RuntimeError("No such index aigw:cache:idx")
                return [0]
            if command == "FT.CREATE":
                self.created += 1
                self.index_exists = True
                return "OK"
            if command == "FT.INFO":
                if not self.index_exists:
                    raise RuntimeError("Unknown index name")
                return {}
            return None

    redis = VanishingRedis()
    cache = SemanticCache.__new__(SemanticCache)
    cache.redis = redis  # type: ignore[assignment]
    cache._index_ready = True  # the gateway believes the index is present
    cache.available = True

    async def ensure_index() -> None:
        await redis.execute_command("FT.CREATE")
        cache._index_ready = True

    cache.ensure_index = ensure_index  # type: ignore[assignment,method-assign]

    class _Settings:
        cache_index_name = "aigw:cache:idx"

    cache.settings = _Settings()  # type: ignore[assignment]

    result = await cache._search("ns", b"\x00" * 16)

    assert redis.created == 1, "the missing index must be recreated exactly once"
    assert redis.searches == 2, "the search must be retried after recovery"
    assert result == [0], "the retried search result must be returned"


@pytest.mark.asyncio
async def test_search_does_not_retry_unrelated_errors() -> None:
    """Only a missing index is recoverable; other failures must surface."""

    class BrokenRedis:
        def __init__(self) -> None:
            self.searches = 0

        async def execute_command(self, command: str, *args: object) -> object:
            self.searches += 1
            raise RuntimeError("READONLY You can't write against a read only replica")

    redis = BrokenRedis()
    cache = SemanticCache.__new__(SemanticCache)
    cache.redis = redis  # type: ignore[assignment]
    cache._index_ready = True
    cache.available = True

    class _Settings:
        cache_index_name = "aigw:cache:idx"

    cache.settings = _Settings()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="READONLY"):
        await cache._search("ns", b"\x00" * 16)

    assert redis.searches == 1, "an unrelated error must not be retried"
