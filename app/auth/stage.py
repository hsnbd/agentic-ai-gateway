"""Pipeline stage that authenticates virtual keys and enforces their policies."""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING

from app.auth.keys import KeyService
from app.auth.quotas import QuotaService
from app.auth.ratelimit import SlidingWindowLimiter
from app.core.errors import (
    AuthenticationError,
    PermissionDeniedError,
    RateLimitExceededError,
)

if TYPE_CHECKING:
    from app.core.pipeline import RequestContext


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

        quotas = QuotaService(ctx.state.db, ctx.state.redis)
        await quotas.reset_if_due(key)
        await quotas.check_budget(key)

        if key.rpm_limit is not None:
            allowed, _, retry_after = await SlidingWindowLimiter(ctx.state.redis).check_and_consume(
                f"aigw:rl:rpm:{key.id}", key.rpm_limit, 60
            )
            if not allowed:
                raise RateLimitExceededError(
                    "Virtual key request rate limit exceeded", retry_after=retry_after
                )

        ctx.virtual_key = key  # type: ignore[assignment]
        ctx.key_id = key.id
        ctx.team_id = key.team_id
        ctx.end_user = ctx.request.user
        if ctx.request.guardrail_policy is None and key.guardrail_policy is not None:
            ctx.request.guardrail_policy = key.guardrail_policy
        return None
