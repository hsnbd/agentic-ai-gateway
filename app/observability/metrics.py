from __future__ import annotations

from contextlib import AbstractContextManager, suppress
from threading import Lock
from typing import cast

from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from prometheus_client.metrics import MetricWrapperBase


def _registered_metric(
    metric_type: type[MetricWrapperBase],
    name: str,
    documentation: str,
    labelnames: tuple[str, ...] = (),
    buckets: tuple[float, ...] | None = None,
) -> MetricWrapperBase:
    try:
        if metric_type is Counter:
            return Counter(name, documentation, labelnames=labelnames, registry=REGISTRY)
        if metric_type is Gauge:
            return Gauge(name, documentation, labelnames=labelnames, registry=REGISTRY)
        if buckets is None:
            return Histogram(name, documentation, labelnames=labelnames, registry=REGISTRY)
        return Histogram(
            name,
            documentation,
            labelnames=labelnames,
            buckets=buckets,
            registry=REGISTRY,
        )
    except ValueError:
        registered = REGISTRY._names_to_collectors.get(name)
        if registered is None:
            raise
        if not isinstance(registered, metric_type):
            raise RuntimeError(
                f"Registered Prometheus metric {name!r} has an unexpected type"
            ) from None
        return cast(MetricWrapperBase, registered)


REQUESTS = cast(
    Counter,
    _registered_metric(
        Counter,
        "aigw_requests_total",
        "Gateway requests",
        ("model", "provider", "status", "dialect"),
    ),
)
ERRORS = cast(
    Counter,
    _registered_metric(
        Counter, "aigw_errors_total", "Gateway errors", ("model", "provider", "error_code")
    ),
)
REQUEST_DURATION = cast(
    Histogram,
    _registered_metric(
        Histogram,
        "aigw_request_duration_seconds",
        "Gateway request duration in seconds",
        ("model", "provider"),
        (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
    ),
)
TIME_TO_FIRST_TOKEN = cast(
    Histogram,
    _registered_metric(
        Histogram,
        "aigw_time_to_first_token_seconds",
        "Time to first generated token in seconds",
        ("model", "provider"),
        (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    ),
)
TOKENS = cast(
    Counter,
    _registered_metric(
        Counter, "aigw_tokens_total", "Tokens processed", ("model", "provider", "kind")
    ),
)
COST = cast(
    Counter,
    _registered_metric(
        Counter,
        "aigw_cost_usd_total",
        "Estimated model cost in USD",
        ("model", "provider", "virtual_key"),
    ),
)
CACHE_LOOKUPS = cast(
    Counter,
    _registered_metric(Counter, "aigw_cache_lookups_total", "Cache lookup results", ("result",)),
)
CACHE_HIT_RATIO = cast(
    Gauge,
    _registered_metric(Gauge, "aigw_cache_hit_ratio", "Ratio of cache lookups that hit"),
)
CACHE_COST_SAVED = cast(
    Counter,
    _registered_metric(
        Counter, "aigw_cache_cost_saved_usd_total", "Estimated cost avoided by cache hits"
    ),
)
FALLBACKS = cast(
    Counter,
    _registered_metric(
        Counter,
        "aigw_fallbacks_total",
        "Provider fallback events",
        ("from_provider", "to_provider", "reason"),
    ),
)
RETRIES = cast(
    Counter,
    _registered_metric(
        Counter, "aigw_retries_total", "Provider retry events", ("provider", "error_code")
    ),
)
PROVIDER_HEALTH = cast(
    Gauge,
    _registered_metric(
        Gauge, "aigw_provider_healthy", "Provider deployment health", ("provider", "deployment")
    ),
)
GUARDRAIL_ACTIONS = cast(
    Counter,
    _registered_metric(
        Counter, "aigw_guardrail_actions_total", "Guardrail actions", ("policy", "phase", "action")
    ),
)
ACTIVE_REQUESTS = cast(
    Gauge,
    _registered_metric(Gauge, "aigw_active_requests", "Active requests", ("model",)),
)
RATE_LIMIT_HITS = cast(
    Counter,
    _registered_metric(Counter, "aigw_rate_limit_hits_total", "Rate limit hits", ("scope",)),
)


def _label(value: str | None, default: str = "unknown", max_length: int = 128) -> str:
    return (value or default)[:max_length]


def _key_label(key_id: str | None) -> str:
    """Only accept an internal key identifier, never a raw client credential."""
    return _label(key_id, "anonymous", 64)


def record_request(
    model: str,
    provider: str,
    status: str,
    dialect: str,
    duration_seconds: float,
) -> None:
    try:
        REQUESTS.labels(_label(model), _label(provider), _label(status), _label(dialect)).inc()
        REQUEST_DURATION.labels(_label(model), _label(provider)).observe(max(duration_seconds, 0.0))
    except Exception:
        return


def record_error(model: str, provider: str, error_code: str) -> None:
    try:
        ERRORS.labels(_label(model), _label(provider), _label(error_code)).inc()
    except Exception:
        return


def record_tokens(
    model: str,
    provider: str,
    prompt: int,
    completion: int,
    cached: int = 0,
) -> None:
    try:
        for kind, count in (("prompt", prompt), ("completion", completion), ("cached", cached)):
            if count > 0:
                TOKENS.labels(_label(model), _label(provider), kind).inc(count)
    except Exception:
        return


def record_cost(model: str, provider: str, virtual_key: str | None, usd: float) -> None:
    try:
        if usd > 0:
            COST.labels(_label(model), _label(provider), _key_label(virtual_key)).inc(usd)
    except Exception:
        return


_cache_counts_lock = Lock()
_cache_hits = int(CACHE_LOOKUPS.labels("hit")._value.get())
_cache_misses = int(CACHE_LOOKUPS.labels("miss")._value.get())


def record_cache(result: str, cost_saved_usd: float = 0.0) -> None:
    global _cache_hits, _cache_misses
    try:
        outcome = result if result in {"hit", "miss", "skip"} else "skip"
        CACHE_LOOKUPS.labels(outcome).inc()
        with _cache_counts_lock:
            if outcome == "hit":
                _cache_hits += 1
            elif outcome == "miss":
                _cache_misses += 1
            lookup_count = _cache_hits + _cache_misses
            CACHE_HIT_RATIO.set(_cache_hits / lookup_count if lookup_count else 0.0)
        if outcome == "hit" and cost_saved_usd > 0:
            CACHE_COST_SAVED.inc(cost_saved_usd)
    except Exception:
        return


def record_ttft(model: str, provider: str, seconds: float) -> None:
    try:
        TIME_TO_FIRST_TOKEN.labels(_label(model), _label(provider)).observe(max(seconds, 0.0))
    except Exception:
        return


def observe_time_to_first_token(model: str, provider: str, seconds: float) -> None:
    record_ttft(model, provider, seconds)


def record_fallback(from_provider: str, to_provider: str, reason: str) -> None:
    try:
        FALLBACKS.labels(_label(from_provider), _label(to_provider), _label(reason)).inc()
    except Exception:
        return


def record_retry(provider: str, error_code: str) -> None:
    try:
        RETRIES.labels(_label(provider), _label(error_code)).inc()
    except Exception:
        return


def set_provider_health(provider: str, deployment: str, healthy: bool) -> None:
    try:
        PROVIDER_HEALTH.labels(_label(provider), _label(deployment)).set(1.0 if healthy else 0.0)
    except Exception:
        return


def record_guardrail(policy: str, phase: str, action: str) -> None:
    try:
        GUARDRAIL_ACTIONS.labels(_label(policy), _label(phase), _label(action)).inc()
    except Exception:
        return


def set_active_requests(model: str, count: int) -> None:
    try:
        ACTIVE_REQUESTS.labels(_label(model)).set(max(count, 0))
    except Exception:
        return


def record_rate_limit(scope: str) -> None:
    try:
        RATE_LIMIT_HITS.labels(_label(scope)).inc()
    except Exception:
        return


def record_rate_limit_hit(scope: str) -> None:
    record_rate_limit(scope)


class _ActiveRequest(AbstractContextManager[None]):
    def __init__(self, model: str) -> None:
        self.model = _label(model)

    def __enter__(self) -> None:
        with suppress(Exception):
            ACTIVE_REQUESTS.labels(self.model).inc()
        return None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        with suppress(Exception):
            ACTIVE_REQUESTS.labels(self.model).dec()


def track_active(model: str) -> AbstractContextManager[None]:
    return _ActiveRequest(model)


# Named aliases mirror Prometheus exposition names for discovery and integrations.
aigw_requests_total = REQUESTS
aigw_errors_total = ERRORS
aigw_request_duration_seconds = REQUEST_DURATION
aigw_time_to_first_token_seconds = TIME_TO_FIRST_TOKEN
aigw_tokens_total = TOKENS
aigw_cost_usd_total = COST
aigw_cache_lookups_total = CACHE_LOOKUPS
aigw_cache_hit_ratio = CACHE_HIT_RATIO
aigw_cache_cost_saved_usd_total = CACHE_COST_SAVED
aigw_fallbacks_total = FALLBACKS
aigw_retries_total = RETRIES
aigw_provider_healthy = PROVIDER_HEALTH
aigw_guardrail_actions_total = GUARDRAIL_ACTIONS
aigw_active_requests = ACTIVE_REQUESTS
aigw_rate_limit_hits_total = RATE_LIMIT_HITS
