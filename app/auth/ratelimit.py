"""Redis-backed sliding-window request and token limiter."""

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
    return {0, math.max(0, limit - used), window}
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
