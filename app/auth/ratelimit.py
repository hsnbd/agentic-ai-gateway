"""Redis-backed rate limiters: requests per window, tokens per minute, concurrency.

Every limiter fails open. Redis is a dependency of rate limiting, not of
serving traffic, so an outage degrades to "unlimited" with a warning rather
than to "every request rejected".
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_CONSUME_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local used = redis.call('ZCARD', key)
if used + cost > limit then
    -- The caller may retry once the oldest entry leaves the window.
    local retry = window
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    if oldest[2] then
        retry = math.max(1, math.ceil(tonumber(oldest[2]) + window - now))
    end
    return {0, math.max(0, limit - used), retry}
end
for i = 1, cost do
    redis.call('ZADD', key, now, ARGV[4 + i])
end
redis.call('EXPIRE', key, math.max(1, window))
return {1, math.max(0, limit - used - cost), 0}
"""


class SlidingWindowLimiter:
    def __init__(self, redis: Any) -> None:
        self.redis = redis

    async def check_and_consume(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        cost: int = 1,
    ) -> tuple[bool, int, float]:
        now = time.time()
        cost = max(1, cost)
        members = [f"{now:.9f}:{uuid.uuid4().hex}:{index}" for index in range(cost)]
        try:
            pipeline = self.redis.pipeline(transaction=True)
            pipeline.execute_command(
                "EVAL",
                _CONSUME_SCRIPT,
                1,
                key,
                now,
                window_seconds,
                limit,
                cost,
                *members,
            )
            result = (await pipeline.execute())[0]
            return bool(int(result[0])), int(result[1]), float(result[2])
        except Exception:
            logger.warning("Rate limiter unavailable; allowing request", exc_info=True)
            return True, max(0, limit), 0.0

    async def peek(self, key: str, limit: int, window_seconds: int) -> tuple[int, float]:
        now = time.time()
        try:
            pipeline = self.redis.pipeline(transaction=True)
            pipeline.zremrangebyscore(key, "-inf", now - window_seconds)
            pipeline.zcard(key)
            results = await pipeline.execute()
            used = int(results[-1])
            return used, float(max(0, limit - used))
        except Exception:
            logger.warning("Rate limiter usage lookup failed", exc_info=True)
            return 0, float(max(0, limit))


class TokenWindow:
    """Tokens used in the last minute, from two per-minute counters.

    Usage is the current minute's count plus the previous minute's, weighted by
    how much of it still overlaps the sliding 60-second window. That is the
    usual sliding-window-counter approximation: O(1) memory per key, unlike a
    sorted set holding one member per token.
    """

    window_seconds = 60

    def __init__(self, redis: Any) -> None:
        self.redis = redis

    def _keys(self, key: str, now: float) -> tuple[str, str, float]:
        minute = int(now // self.window_seconds)
        elapsed = (now % self.window_seconds) / self.window_seconds
        return f"{key}:{minute}", f"{key}:{minute - 1}", elapsed

    async def usage(self, key: str) -> float:
        current, previous, elapsed = self._keys(key, time.time())
        try:
            values = await self.redis.mget(current, previous)
        except Exception:
            logger.warning("Token limiter unavailable; allowing request", exc_info=True)
            return 0.0
        now_count, prev_count = (float(value or 0) for value in values)
        return now_count + prev_count * (1 - elapsed)

    async def add(self, key: str, tokens: int) -> None:
        if tokens <= 0:
            return
        current, _, _ = self._keys(key, time.time())
        try:
            pipeline = self.redis.pipeline(transaction=True)
            pipeline.incrby(current, tokens)
            pipeline.expire(current, self.window_seconds * 2)
            await pipeline.execute()
        except Exception:
            logger.warning("Token limiter update failed", exc_info=True)

    def seconds_until_reset(self) -> float:
        return self.window_seconds - (time.time() % self.window_seconds)


class ConcurrencyLimiter:
    """Caps in-flight requests per key with a Redis counter.

    The TTL is a safety net: if a process dies holding slots, they free
    themselves once no request has touched the key for `ttl_seconds`.
    """

    def __init__(self, redis: Any, ttl_seconds: int = 300) -> None:
        self.redis = redis
        self.ttl_seconds = ttl_seconds

    async def acquire(self, key: str, limit: int) -> bool:
        try:
            pipeline = self.redis.pipeline(transaction=True)
            pipeline.incr(key)
            pipeline.expire(key, self.ttl_seconds)
            in_flight = int((await pipeline.execute())[0])
        except Exception:
            logger.warning("Concurrency limiter unavailable; allowing request", exc_info=True)
            return True
        if in_flight > limit:
            await self.release(key)
            return False
        return True

    async def release(self, key: str) -> None:
        try:
            if int(await self.redis.decr(key)) < 0:
                await self.redis.set(key, 0, ex=self.ttl_seconds)
        except Exception:
            logger.warning("Concurrency limiter release failed", exc_info=True)
