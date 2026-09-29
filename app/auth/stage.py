"""Pipeline stage that authenticates virtual keys and enforces their policies."""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING

from app.accounting.tokens import estimate_request_tokens
from app.auth.keys import KeyService
from app.auth.quotas import QuotaService
from app.auth.ratelimit import ConcurrencyLimiter, SlidingWindowLimiter, TokenWindow
from app.core.errors import (
    AuthenticationError,
    PermissionDeniedError,
    RateLimitExceededError,
)
from app.observability import metrics

if TYPE_CHECKING:
    from app.core.pipeline import RequestContext


def key_tpm_counter(key_id: str) -> str:
    return f"aigw:tpm:key:{key_id}"


def deployment_tpm_counter(deployment_id: str) -> str:
    return f"aigw:tpm:dep:{deployment_id}"


class AuthStage:
    name = "auth"

    async def process(self, ctx: RequestContext) -> None:
        raw_key = ctx.request.metadata.get("api_key")
        if not isinstance(raw_key, str) or not raw_key:
            raise AuthenticationError()

        settings = ctx.state.settings
        if hmac.compare_digest(raw_key, settings.master_key.get_secret_value()):
            ctx.virtual_key = None
            ctx.key_id = "master"
            ctx.end_user = ctx.request.user
            return None

        key = await KeyService(ctx.state.db, ctx.state.redis).lookup(raw_key)
        if key is None or not key.is_valid():
            raise AuthenticationError()
        if not key.permits_model(ctx.request.model):
            raise PermissionDeniedError(
                "This virtual key is not permitted to use the requested model"
            )

        if not key.permits_route(ctx.route):
            raise PermissionDeniedError(f"This virtual key is not permitted to call {ctx.route}")

        quotas = QuotaService(ctx.state.db, ctx.state.redis)
        await quotas.reset_if_due(key)
        await quotas.check_budget(key)

        redis = ctx.state.redis
        if key.rpm_limit is not None:
            allowed, _, retry_after = await SlidingWindowLimiter(redis).check_and_consume(
                f"aigw:rl:rpm:{key.id}", key.rpm_limit, 60
            )
            if not allowed:
                metrics.record_rate_limit("key_rpm")
                raise RateLimitExceededError(
                    "Virtual key request rate limit exceeded", retry_after=retry_after
                )

        if key.tpm_limit is not None:
            # Admission uses the prompt estimate; the observability stage adds
            # the actual prompt and completion tokens once the response exists.
            window = TokenWindow(redis)
            used = await window.usage(key_tpm_counter(key.id))
            if used + estimate_request_tokens(ctx.request) > key.tpm_limit:
                metrics.record_rate_limit("key_tpm")
                raise RateLimitExceededError(
                    "Virtual key token rate limit exceeded",
                    retry_after=window.seconds_until_reset(),
                )

        if key.max_parallel_requests is not None:
            limiter = ConcurrencyLimiter(redis)
            slot = f"aigw:par:key:{key.id}"
            if not await limiter.acquire(slot, key.max_parallel_requests):
                metrics.record_rate_limit("key_parallel")
                raise RateLimitExceededError(
                    "Virtual key has too many requests in flight", retry_after=1
                )
            ctx.cleanups.append(lambda: limiter.release(slot))

        ctx.virtual_key = key  # type: ignore[assignment]
        ctx.key_id = key.id
        ctx.team_id = key.team_id
        ctx.end_user = ctx.request.user
        if ctx.request.guardrail_policy is None and key.guardrail_policy is not None:
            ctx.request.guardrail_policy = key.guardrail_policy
        return None
