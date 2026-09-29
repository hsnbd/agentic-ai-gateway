"""Composition root.

Assembles the request pipeline from whichever subsystems are configured and
importable. Stage order is the contract of the whole gateway, so it is
declared once, here, rather than being spread across modules.

Each subsystem is constructed explicitly, because they legitimately need
different dependencies (a policy registry, a Redis handle, a price table).
Optional subsystems are wired defensively: a gateway without Redis should
still serve traffic, just without caching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.pipeline import Pipeline, PostStage, Stage
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.core.state import GatewayState

logger = get_logger(__name__)


def build_pipeline(state: GatewayState) -> Pipeline:
    """Wire the stages in execution order.

    Pre-stages (any may short-circuit):
      1. auth        — identify the key, enforce rate limits and budgets
      2. guardrails  — inspect the prompt before it costs anything
      3. cache       — a hit here skips the provider call entirely

    Then the resilient executor performs routing, retries, and fallbacks.

    Post-stages (each sees the finished response):
      1. guardrails_out — inspect what the model produced
      2. cache_write    — store the result for next time
      3. observability  — metrics, cost accounting, and the request log

    Ordering rationale: auth precedes everything because unauthenticated work
    should cost nothing. Guardrails precede the cache so a blocked prompt is
    never even looked up. Observability is last so it records the final,
    post-guardrail response and the timings of every stage before it.
    """
    pre: list[Stage] = []
    post: list[PostStage] = []

    auth = _build_auth(state)
    if auth is not None:
        pre.append(auth)

    guard_in, guard_out = _build_guardrails(state)
    if guard_in is not None:
        pre.append(guard_in)

    cache_read, cache_write = _build_cache(state)
    if cache_read is not None:
        pre.append(cache_read)

    if guard_out is not None:
        post.append(guard_out)
    if cache_write is not None:
        post.append(cache_write)

    observability = _build_observability(state)
    if observability is not None:
        post.append(observability)

    executor = _build_executor(state)

    logger.info(
        "pipeline_built",
        pre_stages=[s.name for s in pre],
        post_stages=[s.name for s in post],
    )
    return Pipeline(pre_stages=pre, executor=executor, post_stages=post)


def _build_auth(state: GatewayState) -> Stage | None:
    """Auth is mandatory.

    Serving unauthenticated traffic silently would be worse than refusing to
    start, so an import failure here is fatal rather than degraded.
    """
    try:
        from app.auth.stage import AuthStage
    except ImportError as exc:
        raise RuntimeError(f"Required auth stage is unavailable: {exc}") from exc
    return AuthStage()


def _build_guardrails(state: GatewayState) -> tuple[Stage | None, PostStage | None]:
    if not state.settings.guardrails_enabled:
        logger.info("stage_disabled", stage="guardrails")
        return None, None

    try:
        from app.guardrails.registry import GuardrailRegistry
        from app.guardrails.stage import InputGuardrailStage, OutputGuardrailStage

        path = state.settings.guardrails_config_path
        registry = GuardrailRegistry.load(path)
    except Exception as exc:
        logger.warning("stage_unavailable", stage="guardrails", reason=str(exc))
        return None, None

    state.components["guardrails"] = registry
    return InputGuardrailStage(registry), OutputGuardrailStage(registry)


def _build_cache(state: GatewayState) -> tuple[Stage | None, PostStage | None]:
    if not state.settings.cache_enabled:
        logger.info("stage_disabled", stage="cache")
        return None, None
    if state.redis is None:
        logger.warning("stage_unavailable", stage="cache", reason="no redis connection")
        return None, None

    try:
        from app.cache.embedder import CacheEmbedder
        from app.cache.semantic import SemanticCache
        from app.cache.stage import CacheWriteStage, SemanticCacheStage

        embedder = CacheEmbedder(state.registry, state.settings)
        cache = SemanticCache(state.redis, embedder, state.settings)
    except Exception as exc:
        logger.warning("stage_unavailable", stage="cache", reason=str(exc))
        return None, None

    state.components["cache"] = cache
    return SemanticCacheStage(cache), CacheWriteStage(cache)


def _build_observability(state: GatewayState) -> PostStage | None:
    """Metrics always work; only cost accounting needs the price table and DB."""
    try:
        from app.observability.stage import ObservabilityStage
    except ImportError as exc:
        logger.warning("stage_unavailable", stage="observability", reason=str(exc))
        return None

    try:
        from app.accounting.pricing import PriceTable
        from app.accounting.usage import UsageService

        prices = PriceTable(state.settings.pricing_config_path)
        usage = UsageService(state.db, state.redis, prices)
    except Exception as exc:
        # Still record metrics; just lose per-request cost rows.
        logger.warning("accounting_degraded", reason=str(exc))
        return ObservabilityStage()

    state.components["usage"] = usage
    state.components["prices"] = prices
    return ObservabilityStage(usage)


def _build_executor(state: GatewayState) -> Any:
    from app.routing.breaker import CircuitBreaker
    from app.routing.resilience import ResilientExecutor, RetryPolicy
    from app.routing.router import Router

    settings = state.settings
    breaker = CircuitBreaker(
        threshold=settings.circuit_breaker_threshold,
        cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
    )
    router = Router(
        state.registry,
        breaker,
        default_strategy=settings.routing_strategy,
    )
    policy = RetryPolicy(
        max_attempts=settings.max_retries,
        initial_backoff=settings.retry_base_delay_seconds,
        max_backoff=settings.retry_max_delay_seconds,
    )

    # Exposed so health endpoints and the console can read breaker state.
    state.breaker = breaker
    state.router = router

    return ResilientExecutor(
        state.registry,
        router,
        breaker,
        policy,
        max_fallbacks=settings.max_fallbacks,
    )
