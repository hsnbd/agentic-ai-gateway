"""Pipeline stages for semantic cache lookup and response storage."""

from __future__ import annotations

from typing import Literal

from app.accounting.pricing import PriceTable
from app.cache.semantic import SemanticCache
from app.core.pipeline import RequestContext
from app.core.schemas import ChatResponse


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
        # Recorded once, by the observability stage, together with savings.
        ctx.cache_result = result
        return response


class CacheWriteStage:
    """Stores completed responses after other post-processing stages."""

    name = "cache_write"

    def __init__(self, cache: SemanticCache, prices: PriceTable | None = None) -> None:
        self.cache = cache
        self.prices = prices

    async def finalize(self, ctx: RequestContext, response: ChatResponse) -> ChatResponse:
        if not ctx.cache_hit:
            if response.cost_usd is None and self.prices is not None:
                # Accounting runs after this stage, so price the response now:
                # the stored cost is what a future hit reports as saved.
                deployment = ctx.routing.deployment if ctx.routing is not None else None
                response.cost_usd = self.prices.estimate_cost(
                    response.model,
                    response.provider or "unknown",
                    response.usage,
                    deployment,
                )
            if response.latency_ms is None:
                # Stored so a later hit can report how much time it saved.
                response.latency_ms = ctx.elapsed_ms()
            await self.cache.store(ctx, response)
        return response
