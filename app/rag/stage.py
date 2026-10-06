"""Pipeline stage that grounds chat requests in a RAG collection (`aigw.rag`)."""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Any

from app.core.errors import ErrorCode, GatewayError, InvalidRequestError, NotFoundError
from app.core.schemas import ChatRequest, RagOptions
from app.rag.access import UNRESTRICTED, RagAccess
from app.rag.retrieve import augment_request, build_context

if TYPE_CHECKING:
    from app.core.pipeline import RequestContext
    from app.rag.service import RagService

logger = logging.getLogger(__name__)


async def retrieve_and_augment(
    service: RagService,
    request: ChatRequest,
    options: RagOptions,
    access: RagAccess = UNRESTRICTED,
) -> tuple[ChatRequest, list[dict[str, Any]]]:
    """Search the collection and return the augmented request plus its sources.

    Unknown collections and bad options surface as client errors; anything
    else (Redis down, embedding provider failing) becomes `rag_unavailable`,
    because answering without the requested grounding would be silently wrong.
    """
    query = options.query or next(
        (m.text() for m in reversed(request.messages) if m.role.value == "user"), ""
    )
    try:
        chunks = await service.search(
            options.collection_id,
            query,
            top_k=options.top_k,
            min_score=options.min_score,
            diversity=options.diversity,
            filters=options.filters,
            search_mode=options.search_mode,
            rerank_model=options.rerank_model,
            access=access,
        )
    except (NotFoundError, InvalidRequestError):
        raise
    except Exception as exc:
        logger.warning("RAG retrieval failed", exc_info=True)
        raise GatewayError(
            ErrorCode.RAG_UNAVAILABLE,
            f"Retrieval from collection {options.collection_id!r} failed: {exc}",
            cause=exc,
        ) from exc
    context = build_context(chunks, max_tokens=options.max_context_tokens)
    augmented = augment_request(request, context, mode=options.mode)
    return augmented, [dataclasses.asdict(chunk) for chunk in chunks]


class RagStage:
    """Runs after auth (retrieval costs an embedding call) and before input
    guardrails and the cache, so retrieved text is screened and the cache keys
    on the grounded request."""

    name = "rag"

    async def process(self, ctx: RequestContext) -> None:
        options = ctx.request.rag
        if options is None:
            return None
        service: RagService = ctx.state.components["rag_service"]
        started = time.perf_counter()
        ctx.request, ctx.rag_sources = await retrieve_and_augment(
            service, ctx.request, options, RagAccess.for_context(ctx)
        )
        ctx.__dict__.setdefault("_rag_details", {}).update(
            {
                "collection_id": options.collection_id,
                "chunks": len(ctx.rag_sources),
                "mode": options.mode,
                "ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
        return None
