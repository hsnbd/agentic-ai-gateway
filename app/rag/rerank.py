"""Optional LLM reranking of retrieved chunks.

Vector and keyword scores say how *similar* a chunk is to the query, not how
well it *answers* it. A reranker shows the candidates to a chat model and
reorders them by relevance. It is off unless a request (or the collection)
names a ``rerank_model``, it adds a model round trip, and any failure — a
timeout, an unavailable model, an unparseable answer — falls back to the
original order, so reranking can only improve retrieval, never break it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable

from app.observability.metrics import RAG_RERANKS
from app.rag.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

#: How long a rerank may take before the original order is used.
RERANK_TIMEOUT_SECONDS = 10.0
#: Characters of each chunk shown to the reranker.
_PASSAGE_CHARS = 800

_INSTRUCTIONS = (
    "You rank passages by how well they answer a search query. Reply with only a JSON "
    "array of passage numbers, most relevant first, for example [2, 0, 1]."
)

ModelCaller = Callable[[str, str, str], Awaitable[str]]


async def rerank(
    caller: ModelCaller,
    model: str,
    query: str,
    chunks: list[RetrievedChunk],
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    """Reorder ``chunks`` by the model's judgement of relevance; keep ``top_k``."""
    if len(chunks) <= 1:
        return chunks[:top_k]
    passages = "\n\n".join(
        f"[{index}] {chunk.text[:_PASSAGE_CHARS]}" for index, chunk in enumerate(chunks)
    )
    try:
        async with asyncio.timeout(RERANK_TIMEOUT_SECONDS):
            answer = await caller(model, _INSTRUCTIONS, f"Query: {query}\n\nPassages:\n{passages}")
        order = parse_order(answer, len(chunks))
    except Exception as exc:
        RAG_RERANKS.labels("fallback").inc()
        logger.warning("Reranking with %s failed (%s); keeping retrieval order", model, exc)
        return chunks[:top_k]
    RAG_RERANKS.labels("reranked").inc()
    return [chunks[index] for index in order][:top_k]


def parse_order(answer: str, count: int) -> list[int]:
    """Indices from the model's JSON array; omitted indices keep their original order."""
    match = re.search(r"\[[^\[\]]*\]", answer)
    if match is None:
        raise ValueError(f"no JSON array in reranker answer: {answer[:100]!r}")
    ranked = json.loads(match.group(0))
    if not isinstance(ranked, list):  # pragma: no cover - the regex only matches arrays
        raise ValueError("reranker answer is not a list")
    order: list[int] = []
    for value in ranked:
        valid = isinstance(value, int) and not isinstance(value, bool) and 0 <= value < count
        if valid and value not in order:
            order.append(value)
    if not order:
        raise ValueError(f"reranker named no valid passages: {answer[:100]!r}")
    return order + [index for index in range(count) if index not in order]
