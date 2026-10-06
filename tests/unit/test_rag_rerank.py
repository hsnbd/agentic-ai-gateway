"""LLM reranking: parsing the model's order, and falling back on any failure."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.rag import rerank as rerank_module
from app.rag.rerank import parse_order, rerank
from app.rag.retrieve import RetrievedChunk


def _chunks(count: int) -> list[RetrievedChunk]:
    return [RetrievedChunk(id=f"c{i}", text=f"passage {i}", score=1 - i / 10) for i in range(count)]


@pytest.mark.parametrize(
    ("answer", "order"),
    [
        ("[2, 0, 1]", [2, 0, 1]),
        ("Ranking: [1] then the rest", [1, 0, 2]),
        ('[2, 2, 7, -1, true, "0", 0]', [2, 0, 1]),
    ],
)
def test_parse_order(answer: str, order: list[int]) -> None:
    assert parse_order(answer, 3) == order


@pytest.mark.parametrize("answer", ["no array here", "[9, 8]", "[]"])
def test_parse_order_rejects_answers_without_valid_passages(answer: str) -> None:
    with pytest.raises(ValueError):
        parse_order(answer, 3)


async def test_rerank_reorders_and_keeps_top_k() -> None:
    seen: list[str] = []

    async def caller(model: str, instructions: str, text: str) -> str:
        seen.append(text)
        return "[3, 1]"

    result = await rerank(caller, "judge", "q", _chunks(4), top_k=3)
    assert [chunk.id for chunk in result] == ["c3", "c1", "c0"]
    assert "Query: q" in seen[0] and "[3] passage 3" in seen[0]


async def test_rerank_falls_back_on_errors_and_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(model: str, instructions: str, text: str) -> str:
        raise RuntimeError("model unavailable")

    assert [c.id for c in await rerank(broken, "m", "q", _chunks(3), top_k=2)] == ["c0", "c1"]

    async def slow(model: str, instructions: str, text: str) -> str:
        await asyncio.sleep(1)
        return "[1, 0]"

    monkeypatch.setattr(rerank_module, "RERANK_TIMEOUT_SECONDS", 0.01)
    assert [c.id for c in await rerank(slow, "m", "q", _chunks(2), top_k=2)] == ["c0", "c1"]


async def test_a_single_chunk_is_not_sent_to_the_model() -> None:
    async def never(*args: Any) -> str:
        raise AssertionError("should not be called")

    assert [c.id for c in await rerank(never, "m", "q", _chunks(1), top_k=5)] == ["c0"]
