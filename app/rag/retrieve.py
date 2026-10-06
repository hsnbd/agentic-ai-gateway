"""Query embedding, vector retrieval, MMR ranking, and prompt augmentation."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.accounting.tokens import count_tokens
from app.core.errors import InvalidRequestError
from app.core.schemas import ChatRequest, Message, Role, TextPart


@dataclass(slots=True)
class RetrievedChunk:
    id: str
    text: str
    score: float
    source: str | None = None
    document_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


async def _embed(embedder: Any, texts: list[str]) -> list[list[float]]:
    result = embedder.embed(texts)
    if inspect.isawaitable(result):
        result = await result
    if hasattr(result, "data"):
        result = [item.embedding for item in result.data]
    return [list(vector) for vector in result]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


async def retrieve(
    store: Any,
    embedder: Any,
    *,
    collection_id: str,
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    filters: Mapping[str, str] | None = None,
    diversity: float = 0.0,
    search_mode: str = "vector",
) -> list[RetrievedChunk]:
    """Retrieve the ``top_k`` chunks most relevant to ``query``.

    ``search_mode="vector"`` ranks by embedding similarity. ``"hybrid"`` also
    runs a keyword (BM25) search and merges the two rankings by reciprocal-rank
    fusion, so a chunk containing an exact code or name surfaces even when its
    embedding is not the closest; scores are then fused ranks in [0, 1], and
    ``min_score`` applies to them.
    """
    if top_k <= 0:
        return []
    if not 0.0 <= diversity <= 1.0:
        raise InvalidRequestError("diversity must be between 0 and 1")
    if search_mode not in SEARCH_MODES:
        raise InvalidRequestError(f"search_mode must be one of: {', '.join(SEARCH_MODES)}")
    query_vectors = await _embed(embedder, [query])
    if not query_vectors:
        return []
    hybrid = search_mode == "hybrid"
    fetch_count = top_k * 3 if diversity > 0 or hybrid else top_k
    matches = await store.search(collection_id, query_vectors[0], fetch_count, filters=filters)
    if hybrid:
        keyword = await store.text_search(
            collection_id, query, fetch_count, len(query_vectors[0]), filters=filters
        )
        matches = fuse_rankings(matches, keyword)

    candidates: list[tuple[RetrievedChunk, list[float] | None]] = []
    missing_vectors: list[int] = []
    for chunk_id, score, metadata in matches:
        if score < min_score:
            continue
        details = dict(metadata)
        vector = details.pop("_vector", None)
        candidates.append(
            (
                RetrievedChunk(
                    id=str(chunk_id),
                    text=str(details.pop("content", "")),
                    score=float(score),
                    source=details.pop("source", None) or None,
                    document_id=details.pop("document_id", None) or None,
                    metadata=details,
                ),
                list(vector) if vector is not None else None,
            )
        )
        if candidates[-1][1] is None:
            missing_vectors.append(len(candidates) - 1)

    if diversity == 0:
        return [item for item, _ in candidates[:top_k]]
    if missing_vectors:
        texts = [candidates[index][0].text for index in missing_vectors]
        embedded = await _embed(embedder, texts)
        for index, vector in zip(missing_vectors, embedded, strict=True):
            candidates[index] = (candidates[index][0], vector)

    chosen: list[int] = []
    remaining = set(range(len(candidates)))
    while remaining and len(chosen) < top_k:

        def utility(index: int) -> float:
            chunk, vector = candidates[index]
            redundancy = max(
                (max(0.0, _cosine(vector or [], candidates[prior][1] or [])) for prior in chosen),
                default=0.0,
            )
            return chunk.score - diversity * redundancy

        selected = max(remaining, key=lambda index: (utility(index), -index))
        chosen.append(selected)
        remaining.remove(selected)
    return [candidates[index][0] for index in chosen]


#: Retrieval strategies accepted by ``retrieve``.
SEARCH_MODES = ("vector", "hybrid")

#: Reciprocal-rank-fusion damping: the conventional 60 keeps one list's top
#: hit from overwhelming agreement between both lists.
_RRF_K = 60


def fuse_rankings(
    *rankings: Sequence[tuple[str, float, dict[str, Any]]],
) -> list[tuple[str, float, dict[str, Any]]]:
    """Merge ranked result lists by reciprocal-rank fusion.

    Each list contributes ``1 / (60 + rank)`` for every chunk it ranks; the
    sum is normalised so a chunk ranked first by every list scores 1.0. Raw
    scores are ignored, which is the point: cosine similarities and BM25
    scores are on incomparable scales, but ranks are not.
    """
    fused: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        for rank, (chunk_id, _, details) in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
            metadata.setdefault(chunk_id, details)
    best = len(rankings) / (_RRF_K + 1)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [(chunk_id, score / best, metadata[chunk_id]) for chunk_id, score in ordered]


def _token_count(text: str) -> int:
    return count_tokens(text, "gpt-4o")


def build_context(
    chunks: Sequence[RetrievedChunk],
    *,
    max_tokens: int = 4000,
    template: str | None = None,
) -> str:
    if max_tokens <= 0:
        return ""
    template = template or "[{index}] Source: {source}\n{content}"
    entries: list[str] = []
    for number, chunk in enumerate(chunks, start=1):
        source = chunk.source or "unknown"
        entry = template.format(
            index=number,
            source=source,
            content=chunk.text,
            text=chunk.text,
            score=chunk.score,
        )
        candidate = "\n\n".join((*entries, entry))
        if _token_count(candidate) <= max_tokens:
            entries.append(entry)
            continue

        prefix = template.format(
            index=number, source=source, content="", text="", score=chunk.score
        )
        remaining_budget = max_tokens - _token_count("\n\n".join((*entries, prefix)))
        if remaining_budget <= 0:
            break
        low, high = 0, len(chunk.text)
        best = ""
        while low <= high:
            middle = (low + high) // 2
            shortened = template.format(
                index=number,
                source=source,
                content=chunk.text[:middle],
                text=chunk.text[:middle],
                score=chunk.score,
            )
            if _token_count("\n\n".join((*entries, shortened))) <= max_tokens:
                best = shortened
                low = middle + 1
            else:
                high = middle - 1
        # The prefix alone fits (remaining_budget > 0), so the search always finds
        # at least the header with an empty body.
        entries.append(best)
        break
    return "\n\n".join(entries)


def augment_request(request: ChatRequest, context: str, *, mode: str = "system") -> ChatRequest:
    """Return a deep copy with context added; the caller's request is unchanged."""
    if mode not in {"system", "user"}:
        raise InvalidRequestError("RAG augmentation mode must be 'system' or 'user'")
    augmented = request.model_copy(deep=True)
    if not context:
        return augmented
    if mode == "system":
        augmented.messages.insert(
            0,
            Message(
                role=Role.SYSTEM,
                content=f"Use the following retrieved context when answering.\n\n{context}",
            ),
        )
        return augmented

    for message in reversed(augmented.messages):
        if message.role != Role.USER:
            continue
        prefix = f"Retrieved context:\n{context}\n\n"
        if message.content is None or isinstance(message.content, str):
            message.content = prefix + (message.content or "")
        else:
            message.content.insert(0, TextPart(text=prefix))
        return augmented
    augmented.messages.append(Message(role=Role.USER, content=f"Retrieved context:\n{context}"))
    return augmented
