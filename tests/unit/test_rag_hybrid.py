"""Hybrid retrieval: reciprocal-rank fusion of vector and keyword (BM25) results."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.errors import InvalidRequestError
from app.rag.retrieve import fuse_rankings, retrieve


def _hit(chunk_id: str, score: float = 0.0) -> tuple[str, float, dict[str, Any]]:
    return chunk_id, score, {"content": f"text {chunk_id}", "_vector": [1.0, 0.0]}


def test_fusion_rewards_agreement_and_ignores_raw_scales() -> None:
    vector = [_hit("a", 0.91), _hit("b", 0.90), _hit("c", 0.89)]
    keyword = [_hit("c", 14.2), _hit("d", 9.0)]
    fused = fuse_rankings(vector, keyword)
    assert [chunk_id for chunk_id, _, _ in fused] == ["c", "a", "b", "d"]
    assert fused[0][1] == pytest.approx((1 / 63 + 1 / 61) / (2 / 61))
    assert fuse_rankings([_hit("x")], [_hit("x")])[0][1] == pytest.approx(1.0)
    assert fuse_rankings() == []


class _Store:
    def __init__(self) -> None:
        self.text_calls: list[tuple[str, int, int]] = []

    async def search(
        self, collection_id: str, vector: list[float], k: int, filters: Any = None
    ) -> list[Any]:
        return [_hit("near", 0.95), _hit("other", 0.9)][:k]

    async def text_search(
        self, collection_id: str, text: str, k: int, dims: int, filters: Any = None
    ) -> list[Any]:
        self.text_calls.append((text, k, dims))
        return [_hit("exact-code", 7.5)]


class _Embedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


async def test_hybrid_mode_surfaces_keyword_only_hits() -> None:
    store = _Store()
    vector_only = await retrieve(store, _Embedder(), collection_id="c", query="E-4471", top_k=3)
    assert [chunk.id for chunk in vector_only] == ["near", "other"]
    assert store.text_calls == []

    hybrid = await retrieve(
        store, _Embedder(), collection_id="c", query="E-4471", top_k=3, search_mode="hybrid"
    )
    assert "exact-code" in [chunk.id for chunk in hybrid]
    assert store.text_calls == [("E-4471", 9, 2)]
    assert all(0.0 < chunk.score <= 1.0 for chunk in hybrid)


async def test_hybrid_min_score_applies_to_fused_scores() -> None:
    chunks = await retrieve(
        _Store(),
        _Embedder(),
        collection_id="c",
        query="q",
        top_k=5,
        search_mode="hybrid",
        min_score=0.5,
    )
    # Each is first in one ranking (0.5); "other" (second by vector only) falls below.
    assert [chunk.id for chunk in chunks] == ["exact-code", "near"]


async def test_unknown_search_mode_is_rejected() -> None:
    with pytest.raises(InvalidRequestError, match="search_mode"):
        await retrieve(_Store(), _Embedder(), collection_id="c", query="q", search_mode="magic")
