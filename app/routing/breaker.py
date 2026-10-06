"""Circuit breaker and health tracking.

A breaker is kept per deployment. Consecutive failures trip it open, which
takes the deployment out of routing until a cooldown elapses, after which a
single probe request decides whether to close it again.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class DeploymentHealth:
    """Rolling health for one deployment."""

    deployment_id: str
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    opened_at: float | None = None
    last_error: str | None = None
    #: Exponentially weighted moving average of latency, in milliseconds.
    ewma_latency_ms: float = 0.0
    _latency_samples: int = field(default=0, repr=False)

    def record_latency(self, latency_ms: float) -> None:
        # Warm up with a plain mean so early samples are not dominated by 0.0.
        if self._latency_samples < 5:
            total = self.ewma_latency_ms * self._latency_samples + latency_ms
            self._latency_samples += 1
            self.ewma_latency_ms = total / self._latency_samples
        else:
            alpha = 0.2
            self.ewma_latency_ms = alpha * latency_ms + (1 - alpha) * self.ewma_latency_ms

    @property
    def failure_rate(self) -> float:
        return self.total_failures / self.total_requests if self.total_requests else 0.0


class CircuitBreaker:
    """In-process breaker registry.

    Deliberately process-local: it reacts within milliseconds and needs no
    network round trip. Cross-replica health converges because every replica
    observes the same upstream failures.
    """

    def __init__(self, threshold: int = 5, cooldown_seconds: float = 30.0) -> None:
        self._threshold = threshold
        self._cooldown = cooldown_seconds
        self._health: dict[str, DeploymentHealth] = {}

    def health(self, deployment_id: str) -> DeploymentHealth:
        health = self._health.get(deployment_id)
        if health is None:
            health = DeploymentHealth(deployment_id=deployment_id)
            self._health[deployment_id] = health
        return health

    def is_available(self, deployment_id: str, now: float | None = None) -> bool:
        health = self.health(deployment_id)
        if health.state is BreakerState.CLOSED:
            return True

        now = now if now is not None else time.monotonic()
        if health.state is BreakerState.OPEN:
            if health.opened_at is not None and now - health.opened_at >= self._cooldown:
                # Cooldown elapsed: allow exactly one probe through.
                health.state = BreakerState.HALF_OPEN
                return True
            return False

        # HALF_OPEN: the probe is already in flight, so hold everyone else back.
        return False

    def record_success(self, deployment_id: str, latency_ms: float | None = None) -> None:
        health = self.health(deployment_id)
        health.total_requests += 1
        health.consecutive_failures = 0
        health.state = BreakerState.CLOSED
        health.opened_at = None
        health.last_error = None
        if latency_ms is not None:
            health.record_latency(latency_ms)

    def record_failure(
        self, deployment_id: str, error: str | None = None, now: float | None = None
    ) -> None:
        health = self.health(deployment_id)
        health.total_requests += 1
        health.total_failures += 1
        health.consecutive_failures += 1
        health.last_error = error

        # A failed half-open probe re-opens immediately, without waiting for
        # the threshold to be reached again.
        if health.state is BreakerState.HALF_OPEN or health.consecutive_failures >= self._threshold:
            health.state = BreakerState.OPEN
            health.opened_at = now if now is not None else time.monotonic()

    def reset(self, deployment_id: str) -> None:
        self._health.pop(deployment_id, None)

    def snapshot(self) -> dict[str, DeploymentHealth]:
        return dict(self._health)
