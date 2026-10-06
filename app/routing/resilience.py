"""Retry and fallback execution.

Two nested loops:

1. **Retry** — the same deployment is retried with exponential backoff and
   jitter while the error is classed retryable (429s, timeouts, 5xx).
2. **Fallback** — once a deployment is exhausted, move to the next one in the
   router's chain, provided the error is classed fallbackable.

Non-retryable, non-fallbackable errors (a malformed request, for example)
abort immediately: retrying them only wastes money and latency.
"""

from __future__ import annotations

import asyncio
import dataclasses
import random
import time
from collections.abc import AsyncIterator
from typing import Any

from app.core.errors import (
    AllProvidersFailedError,
    ErrorCode,
    GatewayError,
    NoHealthyDeploymentError,
    RateLimitExceededError,
)
from app.core.pipeline import RequestContext
from app.core.schemas import ChatResponse, StreamChunk
from app.observability import metrics
from app.providers.base import Deployment
from app.providers.registry import ProviderRegistry
from app.routing.breaker import CircuitBreaker
from app.routing.router import Router


class RetryPolicy:
    """Exponential backoff with full jitter, capped."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
        multiplier: float = 2.0,
        jitter: bool = True,
    ) -> None:
        self.max_attempts = max(1, max_attempts)
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.multiplier = multiplier
        self.jitter = jitter

    def backoff(self, attempt: int, retry_after: float | None = None) -> float:
        """Delay before attempt N (1-based). Honours a server Retry-After."""
        if retry_after is not None and retry_after > 0:
            return min(retry_after, self.max_backoff)

        delay = min(self.initial_backoff * (self.multiplier ** (attempt - 1)), self.max_backoff)
        if self.jitter:
            # Full jitter avoids synchronised retry storms across replicas.
            delay = random.uniform(0, delay)
        return delay


class ResilientExecutor:
    """Executes a chat request across a fallback chain with retries."""

    def __init__(
        self,
        registry: ProviderRegistry,
        router: Router,
        breaker: CircuitBreaker,
        policy: RetryPolicy | None = None,
        max_fallbacks: int = 3,
        redis: Any | None = None,
    ) -> None:
        self._registry = registry
        self._router = router
        self._breaker = breaker
        self._policy = policy or RetryPolicy()
        self._max_fallbacks = max_fallbacks
        #: Enables per-deployment `rpm_limit` / `tpm_limit`; None disables them.
        self._redis = redis

    # -- Non-streaming ----------------------------------------------------

    async def execute(self, ctx: RequestContext) -> ChatResponse:
        decision, chain = self._router.route(ctx)
        ctx.routing = decision

        last_error: GatewayError | None = None

        for index, deployment in enumerate(chain[: self._max_fallbacks + 1]):
            if index > 0:
                self._note_fallback(ctx, chain[index - 1], deployment, last_error)

            limited = await self._admit(ctx, deployment)
            if limited is not None:
                last_error = limited
                continue
            try:
                return await self._attempt_deployment(ctx, deployment)
            except GatewayError as exc:
                last_error = exc
                ctx.errors.append(exc)
                if not exc.fallbackable:
                    raise
                continue

        raise self._exhausted(ctx, last_error)

    async def _attempt_deployment(
        self, ctx: RequestContext, deployment: Deployment
    ) -> ChatResponse:
        provider = self._registry.provider_for(deployment)
        last_error: GatewayError | None = None

        for attempt in range(1, self._policy.max_attempts + 1):
            ctx.record_attempt(deployment.id)
            started = time.perf_counter()
            try:
                response = await provider.chat(ctx.request, deployment)
            except GatewayError as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                _record_attempt(ctx, deployment, "error", latency_ms, str(exc))
                last_error = exc
                self._breaker.record_failure(deployment.id, exc.code.value)
                metrics.record_retry(deployment.provider, exc.code.value)

                if not exc.retryable or attempt == self._policy.max_attempts:
                    raise
                delay = self._policy.backoff(attempt, exc.retry_after)
                _log_retry(ctx, deployment, exc, attempt, delay, latency_ms)
                await asyncio.sleep(delay)
                continue

            latency_ms = (time.perf_counter() - started) * 1000
            _record_attempt(ctx, deployment, "success", latency_ms, None)
            self._breaker.record_success(deployment.id, latency_ms)
            metrics.set_provider_health(deployment.provider, deployment.id, True)
            response.deployment_id = deployment.id
            response.provider = deployment.provider
            return response

        # The final attempt always returns or raises; this only satisfies the type checker.
        raise last_error or AllProvidersFailedError(  # pragma: no cover
            "Request failed with no recorded error"
        )

    # -- Streaming --------------------------------------------------------

    async def execute_stream(self, ctx: RequestContext) -> AsyncIterator[StreamChunk]:
        """Stream with fallback *before* the first chunk only.

        Once bytes have reached the client the response is committed: silently
        switching providers mid-stream would splice two different generations
        together. After first token we surface the error instead.
        """
        decision, chain = self._router.route(ctx)
        ctx.routing = decision

        last_error: GatewayError | None = None

        for index, deployment in enumerate(chain[: self._max_fallbacks + 1]):
            if index > 0:
                self._note_fallback(ctx, chain[index - 1], deployment, last_error)

            limited = await self._admit(ctx, deployment)
            if limited is not None:
                last_error = limited
                continue
            provider = self._registry.provider_for(deployment)
            ctx.record_attempt(deployment.id)
            started = time.perf_counter()
            emitted = False

            try:
                async for chunk in provider.stream(ctx.request, deployment):
                    if not emitted:
                        emitted = True
                        ctx.time_to_first_token_ms = (time.perf_counter() - started) * 1000
                        metrics.observe_time_to_first_token(
                            deployment.model_name,
                            deployment.provider,
                            ctx.time_to_first_token_ms / 1000,
                        )
                    chunk.provider = deployment.provider
                    chunk.deployment_id = deployment.id
                    yield chunk
            except GatewayError as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                _record_attempt(ctx, deployment, "error", latency_ms, str(exc))
                self._breaker.record_failure(deployment.id, exc.code.value)
                last_error = exc
                ctx.errors.append(exc)
                if emitted or not exc.fallbackable:
                    raise
                continue

            latency_ms = (time.perf_counter() - started) * 1000
            _record_attempt(ctx, deployment, "success", latency_ms, None)
            self._breaker.record_success(deployment.id, latency_ms)
            return

        raise self._exhausted(ctx, last_error)

    # -- Helpers ----------------------------------------------------------

    def _note_fallback(
        self,
        ctx: RequestContext,
        previous: Deployment,
        nxt: Deployment,
        error: GatewayError | None,
    ) -> None:
        ctx.fallback_used = True
        reason = error.code.value if error else ErrorCode.PROVIDER_ERROR.value
        metrics.record_fallback(previous.provider, nxt.provider, reason)
        # The request log, cost, and console must name the deployment that is
        # actually serving the request, not the one the router picked first.
        if ctx.routing is not None:
            ctx.routing = dataclasses.replace(
                ctx.routing,
                deployment=nxt,
                reason=f"fallback from {previous.id} after {reason}",
            )

    async def _admit(
        self, ctx: RequestContext, deployment: Deployment
    ) -> RateLimitExceededError | None:
        """Check the deployment's own rpm/tpm limits before calling it.

        A saturated deployment is skipped exactly like a failing one, so traffic
        spills over to the next deployment in the chain.
        """
        if self._redis is None or (deployment.rpm_limit is None and deployment.tpm_limit is None):
            return None
        from app.auth.ratelimit import SlidingWindowLimiter, TokenWindow
        from app.auth.stage import deployment_tpm_counter

        error: RateLimitExceededError | None = None
        if deployment.tpm_limit is not None:
            window = TokenWindow(self._redis)
            if await window.usage(deployment_tpm_counter(deployment.id)) >= deployment.tpm_limit:
                error = RateLimitExceededError(
                    f"Deployment {deployment.id} token rate limit reached",
                    retry_after=window.seconds_until_reset(),
                )
        if error is None and deployment.rpm_limit is not None:
            allowed, _, retry_after = await SlidingWindowLimiter(self._redis).check_and_consume(
                f"aigw:rl:dep:rpm:{deployment.id}", deployment.rpm_limit, 60
            )
            if not allowed:
                error = RateLimitExceededError(
                    f"Deployment {deployment.id} request rate limit reached",
                    retry_after=retry_after,
                )
        if error is not None:
            metrics.record_rate_limit("deployment")
            ctx.errors.append(error)
            if ctx.routing is not None:
                ctx.routing.candidates_rejected[deployment.id] = "rate limited"
        return error

    def _exhausted(self, ctx: RequestContext, last_error: GatewayError | None) -> GatewayError:
        if isinstance(last_error, RateLimitExceededError) and all(
            isinstance(error, RateLimitExceededError) for error in ctx.errors
        ):
            # Every deployment was saturated: that is back-pressure the caller
            # can act on, not a provider failure.
            return last_error
        if last_error is None:
            return NoHealthyDeploymentError(f"No deployment available for {ctx.request.model!r}")
        return AllProvidersFailedError(
            f"All providers failed for {ctx.request.model!r}: {last_error.message}",
            provider=last_error.provider,
            model=ctx.request.model,
            details={
                "attempted": ctx.attempted,
                "last_error_code": last_error.code.value,
                "errors": [str(e) for e in ctx.errors[-5:]],
            },
            cause=last_error,
        )


def _record_attempt(
    ctx: RequestContext,
    deployment: Deployment,
    outcome: str,
    latency_ms: float,
    error: str | None,
) -> None:
    details = ctx.__dict__.setdefault("_attempt_details", [])
    details.append(
        {
            "deployment_id": deployment.id,
            "provider": deployment.provider,
            "outcome": outcome,
            "latency_ms": round(latency_ms, 3),
            "error": error,
        }
    )


def _log_retry(
    ctx: RequestContext,
    deployment: Deployment,
    exc: GatewayError,
    attempt: int,
    delay: float,
    latency_ms: float,
) -> None:
    from app.observability.logging import get_logger

    get_logger(__name__).warning(
        "provider_retry",
        request_id=ctx.request_id,
        deployment=deployment.id,
        provider=deployment.provider,
        error_code=exc.code.value,
        attempt=attempt,
        backoff_seconds=round(delay, 3),
        latency_ms=round(latency_ms, 1),
    )
