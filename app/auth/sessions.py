"""Console session revocation: logout, refresh-token rotation, and "sign out everywhere".

JWTs are stateless, so revocation is a Redis denylist:

* one key per revoked token id (`jti`), expiring when the token would have;
* one "not before" timestamp per user, so a password change or a role or
  status change invalidates every token that user already holds.

Checks fail open when Redis is unreachable (logged), matching the rest of the
gateway's Redis-backed controls: an outage should not lock operators out.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_REVOKED = "aigw:console:revoked:"
_NOT_BEFORE = "aigw:console:not_before:"


class TokenRevocation:
    def __init__(self, redis: Any, *, user_ttl_seconds: int = 7 * 24 * 3600) -> None:
        self.redis = redis
        #: How long a user-wide revocation must be remembered: the refresh TTL.
        self.user_ttl_seconds = user_ttl_seconds

    async def revoke(self, claims: dict[str, Any]) -> None:
        jti, exp = claims.get("jti"), claims.get("exp")
        if self.redis is None or not jti or not isinstance(exp, int | float):
            return
        ttl = max(1, int(exp - time.time()))
        try:
            await self.redis.set(f"{_REVOKED}{jti}", "1", ex=ttl)
        except Exception:
            logger.warning("Could not revoke console token", exc_info=True)

    async def revoke_user(self, user_id: str) -> None:
        if self.redis is None:
            return
        try:
            await self.redis.set(
                f"{_NOT_BEFORE}{user_id}", str(_now_ms()), ex=self.user_ttl_seconds
            )
        except Exception:
            logger.warning("Could not revoke console sessions for user", exc_info=True)

    async def is_revoked(self, claims: dict[str, Any]) -> bool:
        if self.redis is None:
            return False
        jti, user_id = claims.get("jti"), claims.get("sub")
        try:
            revoked, not_before = await self.redis.mget(
                f"{_REVOKED}{jti}", f"{_NOT_BEFORE}{user_id}"
            )
        except Exception:
            logger.warning("Console token revocation check unavailable", exc_info=True)
            return False
        if revoked is not None:
            return True
        if not_before is not None:
            issued_ms = claims.get("iat_ms") or int(claims.get("iat", 0)) * 1000
            return int(issued_ms) < int(not_before)
        return False


def _now_ms() -> int:
    return int(time.time() * 1000)
