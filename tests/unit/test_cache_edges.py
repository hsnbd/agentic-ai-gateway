"""Edge cases for the semantic cache, its embedder, and its pipeline stages."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from app.accounting.pricing import PriceTable
from app.cache.embedder import CacheEmbedder
from app.cache.semantic import SemanticCache
from app.cache.stage import CacheWriteStage, SemanticCacheStage
from app.config.settings import Settings
from app.core.pipeline import RoutingDecision
from app.core.schemas import EmbeddingResponse, EmbeddingVector, Message, Role
from app.providers.base import Capabilities, Deployment, Pricing
from tests.unit.test_cache import (
    FakeEmbeddingProvider,
    FakeEmbeddingRegistry,
    StubEmbedder,
    make_cache,
    make_context,
    make_response,
)

# -- CacheEmbedder ----------------------------------------------------------


class ScriptedEmbeddingProvider:
    def __init__(self, *replies: list[list[float]]) -> None:
        self.replies = list(replies)

    async def embed(self, request: Any, deployment: Any) -> EmbeddingResponse:
        vectors = self.replies.pop(0)
        return EmbeddingResponse(
            model=request.model,
            data=[EmbeddingVector(index=i, embedding=v) for i, v in enumerate(vectors)],
        )


def _embedder(provider: Any, registry: Any | None = None) -> CacheEmbedder:
    return CacheEmbedder(
        registry or FakeEmbeddingRegistry(provider),
        Settings(cache_embedding_model="embed-model", cache_embedding_dimensions=2),
    )


async def test_embed_batch_of_nothing_is_empty() -> None:
    assert await _embedder(FakeEmbeddingProvider()).embed_batch([]) == []


async def test_lru_evicts_oldest_entries() -> None:
    provider = FakeEmbeddingProvider()
    embedder = _embedder(provider)
    embedder._cache_size = 2
    await embedder.embed_batch(["a", "b", "c"])
    assert len(embedder._embeddings) == 2
    await embedder.embed("a")
    assert provider.calls == 2


@pytest.mark.parametrize(
    "vectors",
    [
        [[1.0, 0.0], [1.0, 0.0]],  # two vectors for one input
        [[]],  # zero dimensions
    ],
)
async def test_malformed_embedding_replies_are_rejected(vectors: list[list[float]]) -> None:
    assert await _embedder(ScriptedEmbeddingProvider(vectors)).embed("x") is None


async def test_inconsistent_dimensions_are_rejected() -> None:
    provider = ScriptedEmbeddingProvider([[1.0, 0.0], [1.0]])
    assert await _embedder(provider).embed_batch(["a", "b"]) is None


async def test_dimension_change_after_discovery_is_rejected() -> None:
    embedder = _embedder(ScriptedEmbeddingProvider([[1.0, 0.0]], [[1.0, 0.0, 0.0]]))
    assert await embedder.embed("a") == [1.0, 0.0]
    assert await embedder.embed("b") is None
    assert embedder.dimensions == 2


async def test_no_embedding_capable_deployment() -> None:
    registry = FakeEmbeddingRegistry(FakeEmbeddingProvider())
    registry.deployment = Deployment(
        id="chat-only",
        model_name="embed-model",
        provider="fake",
        provider_model="x",
        capabilities=Capabilities(embeddings=False),
    )
    assert await _embedder(None, registry).embed("x") is None


async def test_registry_failure_is_swallowed() -> None:
    class BrokenRegistry:
        def deployments_for(self, model: str) -> list[Deployment]:
            raise KeyError(model)

    assert await _embedder(None, BrokenRegistry()).embed("x") is None


async def test_unexpected_errors_in_the_batch_are_swallowed() -> None:
    embedder = _embedder(FakeEmbeddingProvider())

    async def explode(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("boom")

    embedder._embed_uncached = explode  # type: ignore[method-assign]
    assert await embedder.embed_batch(["x"]) is None


# -- SemanticCache ----------------------------------------------------------


class NoDimensions(StubEmbedder):
    dimensions = None  # type: ignore[assignment]


class ScriptedRedis:
    """Redis double whose commands follow a per-test script."""

    def __init__(self, *, create_error: Exception | None = None, search: Any = None) -> None:
        self.create_error = create_error
        self.search_results = [search]
        self.search_errors: list[Exception] = []
        self.counters: dict[str, float] = {}
        self.fail_counters = False
        self.fail_scan = False
        self.keys: list[Any] = []

    async def execute_command(self, command: str, *args: Any) -> Any:
        if command == "FT.CREATE":
            if self.create_error is not None:
                raise self.create_error
            return "OK"
        if self.search_errors:
            raise self.search_errors.pop(0)
        return self.search_results[0]

    async def incr(self, key: str) -> None:
        if self.fail_counters:
            raise ConnectionError("down")
        self.counters[key] = self.counters.get(key, 0) + 1

    async def incrbyfloat(self, key: str, amount: float) -> None:
        if self.fail_counters:
            raise ConnectionError("down")
        self.counters[key] = self.counters.get(key, 0) + amount

    async def get(self, key: str) -> Any:
        value = self.counters.get(key)
        return None if value is None else str(value).encode()

    async def scan_iter(self, *, match: str) -> AsyncIterator[Any]:
        if self.fail_scan:
            raise ConnectionError("down")
        for key in self.keys:
            yield key

    async def delete(self, *keys: str) -> int:
        return len(keys)

    async def hset(self, key: str, *, mapping: dict[str, Any]) -> None:
        return None

    async def expire(self, key: str, ttl: int) -> None:
        return None


def _cache(redis: Any, embedder: Any | None = None) -> SemanticCache:
    settings = Settings(
        cache_similarity_threshold=0.9,
        cache_embedding_dimensions=2,
        cache_embedding_model="embed-model",
    )
    return SemanticCache(redis, embedder or StubEmbedder(), settings)


async def test_index_waits_for_known_dimensions() -> None:
    cache = _cache(ScriptedRedis(), NoDimensions())
    await cache.ensure_index()
    assert not cache._index_ready
    assert await cache.lookup(make_context()) is None
    await cache.store(make_context(), make_response())
    assert make_context().cache_key is None


async def test_existing_index_is_adopted() -> None:
    cache = _cache(ScriptedRedis(create_error=RuntimeError("Index already exists")))
    await cache.ensure_index()
    assert cache._index_ready and cache.available


async def test_other_index_errors_keep_cache_enabled_but_not_ready() -> None:
    cache = _cache(ScriptedRedis(create_error=RuntimeError("OOM command not allowed")))
    await cache.ensure_index()
    assert cache.available and not cache._index_ready


async def test_lookup_misses_when_embedding_fails() -> None:
    embedder = StubEmbedder()
    embedder.fail = True
    assert await _cache(ScriptedRedis(), embedder).lookup(make_context()) is None


async def test_lookup_ignores_matches_without_distance_or_response() -> None:
    redis = ScriptedRedis(search=[2, b"k1", [b"response", b"{}"], b"k2", [b"distance", b"0.0"]])
    cache = _cache(redis)
    assert await cache.lookup(make_context()) is None
    assert redis.counters["aigw:cache:stats:misses"] == 1


async def test_lookup_without_any_match_is_a_miss() -> None:
    redis = ScriptedRedis(search=[0])
    assert await _cache(redis).lookup(make_context()) is None
    assert redis.counters["aigw:cache:stats:misses"] == 1


async def test_lookup_picks_closest_match_and_records_latency_saved() -> None:
    original = make_response(content="best").model_copy(update={"latency_ms": 60_000.0})
    worse = make_response(content="worse")
    redis = ScriptedRedis(
        search=[
            2,
            b"k1",
            [b"distance", b"0.01", b"response", original.model_dump_json().encode()],
            b"k2",
            [b"distance", b"0.05", b"response", worse.model_dump_json().encode()],
        ]
    )
    ctx = make_context()
    hit = await _cache(redis).lookup(ctx)
    assert hit is not None and hit.choices[0].message.text() == "best"
    assert redis.counters["aigw:cache:stats:latency_saved_ms"] > 0


async def test_counter_failures_do_not_break_a_hit() -> None:
    original = make_response().model_copy(update={"latency_ms": 60_000.0})
    redis = ScriptedRedis(
        search=[1, b"k", [b"distance", b"0", b"response", original.model_dump_json().encode()]]
    )
    redis.fail_counters = True
    assert await _cache(redis).lookup(make_context()) is not None


async def test_search_returning_nothing_after_failed_rebuild() -> None:
    redis = ScriptedRedis()
    redis.search_errors = [RuntimeError("no such index")]
    cache = _cache(redis)
    await cache.ensure_index()
    redis.create_error = RuntimeError("OOM")
    assert await cache.lookup(make_context()) is None
    assert not cache._index_ready


async def test_search_gives_up_after_second_missing_index() -> None:
    redis = ScriptedRedis()
    redis.search_errors = [RuntimeError("no such index"), RuntimeError("no such index")]
    cache = _cache(redis)
    # The second failure propagates; lookup turns it into a miss.
    assert await cache.lookup(make_context()) is None
    assert redis.search_errors == []


async def test_search_reraises_when_rebuilt_index_is_still_missing() -> None:
    cache = _cache(ScriptedRedis())
    calls = 0

    async def ready_but_missing() -> None:
        nonlocal calls
        calls += 1
        cache._index_ready = True

    cache.ensure_index = ready_but_missing  # type: ignore[method-assign]

    async def always_missing(*args: Any) -> Any:
        raise RuntimeError("no such index")

    cache.redis.execute_command = always_missing  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await cache._search("ns", b"")
    assert calls == 1


async def test_store_skips_when_embedding_fails_or_response_is_empty() -> None:
    embedder = StubEmbedder()
    embedder.fail = True
    ctx = make_context()
    await _cache(ScriptedRedis(), embedder).store(ctx, make_response())
    assert ctx.cache_key is None

    cache = _cache(ScriptedRedis())
    empty = make_response().model_copy(update={"choices": []})
    await cache.store(ctx, empty)
    assert ctx.cache_key is None


async def test_store_uses_request_ttl() -> None:
    redis = ScriptedRedis()
    ttls: list[int] = []

    async def expire(key: str, ttl: int) -> None:
        ttls.append(ttl)

    redis.expire = expire  # type: ignore[method-assign]
    ctx = make_context(cache_ttl=42)
    await _cache(redis).store(ctx, make_response())
    assert ttls == [42]
    assert ctx.cache_key is not None


async def test_invalidate_and_stats_edge_cases() -> None:
    redis = ScriptedRedis()
    cache = _cache(redis)
    assert await cache.invalidate() == 0
    redis.keys = [b"aigw:cache:entry:ns:1", "aigw:cache:entry:ns:2"]
    assert await cache.invalidate("ns") == 2
    stats = await cache.stats()
    assert stats == {"entries": 2, "hits": 0, "misses": 0, "latency_saved_ms": 0.0}

    redis.fail_scan = True
    assert await cache.invalidate() == 0
    assert await cache.stats() == {"entries": 0, "hits": 0, "misses": 0}

    cache.available = False
    assert await cache.invalidate() == 0
    assert await cache.stats() == {"entries": 0, "hits": 0, "misses": 0}


def test_search_match_parsing_edge_cases() -> None:
    # A trailing key without a field list is ignored; non-list fields give an empty map.
    assert SemanticCache._search_matches([2, b"k1", b"not-a-list", b"k2"]) == [{}]
    resp3 = {
        b"results": [
            "not-a-dict",
            {"id": "k", "extra_attributes": "not-a-dict"},
            {"id": "k", "extra_attributes": {b"distance": b"0.1"}},
        ]
    }
    assert SemanticCache._search_matches(resp3) == [{"distance": "0.1"}]
    assert SemanticCache._search_matches({"results": "nope"}) == []


def test_embedding_text_keeps_recent_turns_only() -> None:
    messages = [Message(role=Role.SYSTEM, content="sys")]
    messages += [Message(role=Role.USER, content=f"turn {i}") for i in range(6)]
    ctx = make_context(messages=messages)
    text = SemanticCache._embedding_text(ctx)
    assert "sys" not in text
    assert "turn 0" not in text
    assert text.endswith("user: turn 5")


def test_ineligible_when_last_message_is_not_from_the_user() -> None:
    cache, _, _ = make_cache()
    ctx = make_context(messages=[Message(role=Role.ASSISTANT, content="hi")])
    assert not cache._request_is_eligible(ctx)
    streamed = make_context(stream=True)
    assert cache._request_is_eligible(streamed)


# -- Stages -----------------------------------------------------------------


async def test_cache_stage_classifies_hit_miss_and_skip() -> None:
    cache, _, _ = make_cache()
    stage = SemanticCacheStage(cache)

    ctx = make_context()
    assert await stage.process(ctx) is None
    assert ctx.cache_result == "miss"

    await cache.store(ctx, make_response())
    hit_ctx = make_context()
    assert await stage.process(hit_ctx) is not None
    assert hit_ctx.cache_result == "hit"

    skipped = make_context(no_cache=True)
    assert await stage.process(skipped) is None
    assert skipped.cache_result == "skip"

    cache.available = False
    offline = make_context()
    await stage.process(offline)
    assert offline.cache_result == "skip"


async def test_cache_write_stage_prices_and_times_before_storing() -> None:
    cache, redis, _ = make_cache()
    prices = PriceTable(Path("config/pricing.yaml"))
    stage = CacheWriteStage(cache, prices)

    ctx = make_context()
    ctx.started_at = time.perf_counter() - 0.01
    ctx.routing = RoutingDecision(
        deployment=Deployment(
            id="d",
            model_name="model-a",
            provider="p",
            provider_model="m",
            pricing=Pricing(input_per_mtok=1.0),
        ),
        strategy="s",
        reason="r",
    )
    response = make_response().model_copy(update={"cost_usd": None, "latency_ms": None})
    response.usage.prompt_tokens = 1_000_000
    await stage.finalize(ctx, response)
    assert response.cost_usd == pytest.approx(1.0)
    assert response.latency_ms is not None and response.latency_ms > 0
    assert len(redis.entries) == 1

    # Without routing or a price table the provider name falls back to "unknown".
    bare = make_response().model_copy(update={"cost_usd": None})
    await CacheWriteStage(cache, prices).finalize(make_context(), bare)
    assert bare.cost_usd == 0.0
    unpriced = make_response().model_copy(update={"cost_usd": None})
    await CacheWriteStage(cache).finalize(make_context(), unpriced)
    assert unpriced.cost_usd is None


async def test_cache_write_stage_skips_hits() -> None:
    cache, redis, _ = make_cache()
    ctx = make_context()
    ctx.cache_hit = True
    await CacheWriteStage(cache).finalize(ctx, make_response())
    assert redis.entries == {}


async def test_cache_write_stage_keeps_measured_latency() -> None:
    cache, _, _ = make_cache()
    response = make_response().model_copy(update={"latency_ms": 12.5})
    await CacheWriteStage(cache).finalize(make_context(), response)
    assert response.latency_ms == 12.5
