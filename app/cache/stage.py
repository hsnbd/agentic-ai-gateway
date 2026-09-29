"""Pipeline stages for semantic cache lookup and response storage."""

from __future__ import annotations

import logging
from typing import Literal

from app.cache.semantic import SemanticCache
from app.core.pipeline import RequestContext
from app.core.schemas import ChatResponse

logger = logging.getLogger(__name__)


class SemanticCacheStage:
    """Short-circuits the pipeline when a safe semantic match is found."""

    name = "cache"

    def __init__(self, cache: SemanticCache) -> None:
        self.cache = cache

    async def process(self, ctx: RequestContext) -> ChatResponse | None:
        eligible = self.cache._request_is_eligible(ctx)
        response = await self.cache.lookup(ctx)
        result: Literal["hit", "miss", "skip"] = (
            "hit"
            if response is not None
            else "miss"
            if eligible and self.cache.available
            else "skip"
        )
        try:
            from app.observability import metrics

            metrics.record_cache(result, ctx.cost_saved_usd)
        except ImportError:
            pass
        except Exception:
            logger.warning("Unable to record semantic cache metric", exc_info=True)
        return response


class CacheWriteStage:
    """Stores completed responses after other post-processing stages."""

    name = "cache_write"

    def __init__(self, cache: SemanticCache) -> None:
        self.cache = cache

    async def finalize(self, ctx: RequestContext, response: ChatResponse) -> ChatResponse:
        if not ctx.cache_hit:
            await self.cache.store(ctx, response)
        return response
