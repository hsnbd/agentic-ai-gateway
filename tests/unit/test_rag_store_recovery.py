"""RagVectorStore error paths around a missing or broken RediSearch index."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.store import RagVectorStore


class _Index:
    def __init__(self, *, info: list[Any] | None = None, search: Any = None) -> None:
        self._infos = list(info or [])
        self.search = AsyncMock(side_effect=search)
        self.dropindex = AsyncMock()

    async def info(self) -> Any:
        item = self._infos.pop(0) if len(self._infos) > 1 else self._infos[0]
        if isinstance(item, Exception):
            raise item
        return item


def _store(index: _Index, keys: list[bytes] | None = None) -> RagVectorStore:
    redis = MagicMock()
    redis.ft.return_value = index

    async def scan_iter(**_: Any) -> Any:
        for key in keys or []:
            yield key

    redis.scan_iter = scan_iter
    redis.hget = AsyncMock(return_value=b"doc-1")
    redis.delete = AsyncMock(return_value=1)
    return RagVectorStore(redis, 4)


async def test_delete_document_reraises_unexpected_search_errors() -> None:
    store = _store(_Index(info=[{}], search=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        await store.delete_document("c", "doc-1")


async def test_delete_document_falls_back_to_scanning_keys() -> None:
    store = _store(_Index(info=[{}], search=RuntimeError("no such index")), keys=[b"aigw:rag:c:1"])
    assert await store.delete_document("c", "doc-1") == 1


async def test_index_stats_reraises_unexpected_errors() -> None:
    store = _store(_Index(info=[RuntimeError("connection reset")]))
    with pytest.raises(RuntimeError, match="connection reset"):
        await store.index_stats("c")


async def test_reset_tolerates_a_missing_index_and_deletes_stale_vectors() -> None:
    index = _Index(info=[{"indexing": "0"}])
    index.dropindex.side_effect = RuntimeError("Unknown index name")
    store = _store(index, keys=[b"aigw:rag:c:1", b"aigw:rag:c:2"])
    store.redis.ft.return_value.create_index = AsyncMock()
    await store.reset("c", 4)
    store.redis.delete.assert_awaited_once_with(b"aigw:rag:c:1", b"aigw:rag:c:2")

    index.dropindex.side_effect = RuntimeError("permission denied")
    with pytest.raises(RuntimeError, match="permission denied"):
        await store.reset("c", 4)


async def test_with_index_gives_up_after_one_rebuild() -> None:
    index = _Index(info=[{"indexing": "0"}])
    store = _store(index)
    index.create_index = AsyncMock()

    async def always_missing(_: Any) -> Any:
        raise RuntimeError("no such index")

    with pytest.raises(RuntimeError, match="no such index"):
        await store._with_index("c", 4, always_missing)


async def test_await_backfill_polls_until_indexing_finishes_or_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _Index(info=[{"indexing": "1"}, {"indexing": "0"}])
    await _store(index)._await_backfill("c")

    import app.rag.store as module

    class _Expired:
        async def __aenter__(self) -> None:
            raise TimeoutError

        async def __aexit__(self, *_: Any) -> None:
            return None

    monkeypatch.setattr(module.asyncio, "timeout", lambda _: _Expired())
    await _store(_Index(info=[{"indexing": "1"}]))._await_backfill("c")


async def test_text_search_falls_back_to_legacy_bm25_once() -> None:
    calls: list[list[Any]] = []

    async def search(query: Any) -> Any:
        calls.append(query.get_args())
        if "BM25STD" in query.get_args():
            raise RuntimeError("Unknown scorer BM25STD")
        return type("R", (), {"docs": []})()

    index = _Index(info=[{"indexing": "0"}])
    index.search = search  # type: ignore[assignment]
    store = _store(index)
    store._indexes.add(store._index_name("c"))
    assert await store.text_search("c", "error E-4471", 5, 4) == []
    assert await store.text_search("c", "again", 5, 4) == []
    scorers = [args[args.index("SCORER") + 1] for args in calls]
    assert scorers == ["BM25STD", "BM25", "BM25"]


async def test_text_search_reraises_other_errors_and_skips_empty_queries() -> None:
    index = _Index(info=[{"indexing": "0"}], search=RuntimeError("connection reset"))
    store = _store(index)
    store._indexes.add(store._index_name("c"))
    assert await store.text_search("c", "?!", 5, 4) == []
    with pytest.raises(RuntimeError, match="connection reset"):
        await store.text_search("c", "word", 5, 4)
    store.available = False
    assert await store.text_search("c", "word", 5, 4) == []


async def test_text_search_returns_nothing_when_the_index_cannot_be_created() -> None:
    index = _Index(info=[RuntimeError("no such index")])
    index.create_index = AsyncMock(side_effect=RuntimeError("unknown command FT.CREATE"))
    store = _store(index)
    assert await store.text_search("c", "word", 5, 4) == []
    assert store.available is False
