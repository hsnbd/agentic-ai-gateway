"""Failure-tolerance of metrics, tracing, logging setup, and the observability stage."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from prometheus_client import Counter, Gauge, Histogram

from app.accounting.pricing import PriceTable
from app.accounting.usage import UsageService
from app.config.settings import Settings
from app.core.errors import ErrorCode, ProviderError
from app.core.pipeline import RequestContext, RoutingDecision
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    Choice,
    FinishReason,
    Message,
    Role,
    Usage,
)
from app.observability import logging as logging_module
from app.observability import metrics, tracing
from app.observability.stage import ObservabilityStage, _charge_spend, _count_tokens
from app.providers.base import Deployment
from tests.unit.test_auth_edges import CounterRedis

# -- Metrics ----------------------------------------------------------------


class _Broken:
    def labels(self, *args: Any) -> Any:
        raise RuntimeError("metric backend broken")

    def inc(self, *args: Any) -> None:
        raise RuntimeError("metric backend broken")

    def set(self, *args: Any) -> None:
        raise RuntimeError("metric backend broken")


RECORDERS: list[tuple[str, Callable[[], None]]] = [
    ("REQUESTS", lambda: metrics.record_request("m", "p", "success", "openai", 1.0)),
    ("ERRORS", lambda: metrics.record_error("m", "p", "e")),
    ("TOKENS", lambda: metrics.record_tokens("m", "p", 1, 1, 1)),
    ("COST", lambda: metrics.record_cost("m", "p", "k", 1.0)),
    ("CACHE_LOOKUPS", lambda: metrics.record_cache("hit", 1.0)),
    ("TIME_TO_FIRST_TOKEN", lambda: metrics.observe_time_to_first_token("m", "p", 0.1)),
    ("FALLBACKS", lambda: metrics.record_fallback("a", "b", "r")),
    ("RETRIES", lambda: metrics.record_retry("p", "e")),
    ("PROVIDER_HEALTH", lambda: metrics.set_provider_health("p", "d", False)),
    ("GUARDRAIL_ACTIONS", lambda: metrics.record_guardrail("p", "input", "block")),
    ("ACTIVE_REQUESTS", lambda: metrics.set_active_requests("m", -1)),
    ("RATE_LIMIT_HITS", lambda: metrics.record_rate_limit_hit("key")),
]


@pytest.mark.parametrize(("metric", "record"), RECORDERS, ids=[name for name, _ in RECORDERS])
def test_metric_recorders_swallow_backend_errors(
    monkeypatch: pytest.MonkeyPatch, metric: str, record: Callable[[], None]
) -> None:
    monkeypatch.setattr(metrics, metric, _Broken())
    record()


def test_track_active_survives_broken_gauge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "ACTIVE_REQUESTS", _Broken())
    with metrics.track_active("m"):
        pass


def test_zero_counts_are_not_recorded() -> None:
    before = metrics.CACHE_COST_SAVED._value.get()
    metrics.record_tokens("zero-model", "p", 0, 0, 0)
    metrics.record_cost("zero-model", "p", None, 0.0)
    metrics.record_cache("unexpected")
    metrics.record_cache("hit", 0.0)
    assert metrics.CACHE_COST_SAVED._value.get() == before


def test_registered_metric_reuses_and_validates_existing_collectors() -> None:
    histogram = metrics._registered_metric(Histogram, "aigw_test_plain_histogram", "doc")
    assert metrics._registered_metric(Histogram, "aigw_test_plain_histogram", "doc") is histogram
    gauge = metrics._registered_metric(Gauge, "aigw_test_gauge", "doc")
    with pytest.raises(RuntimeError, match="unexpected type"):
        metrics._registered_metric(Counter, "aigw_test_gauge", "doc")
    assert gauge is not None
    # A ValueError that is not a duplicate registration is re-raised.
    with pytest.raises(ValueError, match="sorted"):
        metrics._registered_metric(Histogram, "aigw_test_unsorted", "doc", (), (2.0, 1.0))


# -- Tracing ----------------------------------------------------------------


class _Manager:
    def __init__(self, *, fail_enter: bool = False, fail_exit: bool = False) -> None:
        self.fail_enter = fail_enter
        self.fail_exit = fail_exit
        self.exits: list[Any] = []

    def __enter__(self) -> str:
        if self.fail_enter:
            raise RuntimeError("enter failed")
        return "span"

    def __exit__(self, *exc: Any) -> None:
        self.exits.append(exc[0])
        if self.fail_exit:
            raise RuntimeError("exit failed")


class _Tracer:
    def __init__(self, manager: _Manager) -> None:
        self.manager = manager

    def start_as_current_span(self, name: str, attributes: dict[str, Any]) -> _Manager:
        return self.manager


@contextmanager
def _tracer(monkeypatch: pytest.MonkeyPatch, manager: _Manager) -> Any:
    monkeypatch.setattr(tracing, "_enabled", True)
    monkeypatch.setattr(tracing, "_tracer", _Tracer(manager))
    yield manager


@pytest.mark.parametrize("fail_exit", [False, True])
async def test_span_success_path(monkeypatch: pytest.MonkeyPatch, fail_exit: bool) -> None:
    with _tracer(monkeypatch, _Manager(fail_exit=fail_exit)) as manager:
        async with tracing.span("work", a=1) as current:
            assert current == "span"
    assert manager.exits == [None]


@pytest.mark.parametrize("fail_exit", [False, True])
async def test_span_error_path_reraises(monkeypatch: pytest.MonkeyPatch, fail_exit: bool) -> None:
    with _tracer(monkeypatch, _Manager(fail_exit=fail_exit)) as manager, pytest.raises(ValueError):
        async with tracing.span("work"):
            raise ValueError("inside span")
    assert manager.exits == [ValueError]


async def test_span_enter_failure_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    with _tracer(monkeypatch, _Manager(fail_enter=True)):
        async with tracing.span("work") as current:
            assert current is None


def test_configure_tracing_with_console_and_otlp_exporters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import FastAPI
    from opentelemetry import trace

    monkeypatch.setattr(trace, "set_tracer_provider", lambda provider: None)
    for endpoint in (None, "http://localhost:4318/v1/traces"):
        tracing.configure_tracing(Settings(tracing_enabled=True, otlp_endpoint=endpoint), FastAPI())
        assert tracing._enabled
    # No span is active, so there is no trace id to report.
    assert tracing.current_trace_id() is None
    tracing.configure_tracing(Settings(tracing_enabled=False), FastAPI())


def test_configure_tracing_failure_disables_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("instrumentation failed")

    monkeypatch.setattr(FastAPIInstrumentor, "instrument_app", explode)
    monkeypatch.setattr("opentelemetry.trace.set_tracer_provider", lambda provider: None)
    tracing.configure_tracing(Settings(tracing_enabled=True), object())
    assert not tracing._enabled


def test_current_trace_id_reads_active_span(monkeypatch: pytest.MonkeyPatch) -> None:
    from opentelemetry import trace

    context = SimpleNamespace(is_valid=True, trace_id=0xABC)
    monkeypatch.setattr(tracing, "_enabled", True)
    monkeypatch.setattr(
        trace, "get_current_span", lambda: SimpleNamespace(get_span_context=lambda: context)
    )
    assert tracing.current_trace_id() == f"{0xABC:032x}"

    def explode() -> Any:
        raise RuntimeError("no context")

    monkeypatch.setattr(trace, "get_current_span", explode)
    assert tracing.current_trace_id() is None


# -- Logging ----------------------------------------------------------------


@pytest.mark.parametrize("log_format", ["json", "console"])
def test_configure_logging_installs_redacting_handler(
    log_format: str, capsys: pytest.CaptureFixture[str]
) -> None:
    root = logging.getLogger()
    previous_handlers, previous_level = list(root.handlers), root.level
    try:
        logging_module.configure_logging(Settings(log_format=log_format, log_level="debug"))
        assert root.level == logging.DEBUG
        logging.getLogger("aigw.test").warning("token sk-1234567890abcdef leaked")
        err = capsys.readouterr().err
        assert "sk-1234567890abcdef" not in err
        assert "[REDACTED]" in err
    finally:
        root.handlers[:] = previous_handlers
        root.setLevel(previous_level)


def test_redact_mapping_handles_tuples_and_scalars() -> None:
    cleaned = logging_module.redact_mapping({"items": ("sk-1234567890abc", 3)})
    assert cleaned["items"] == ("[REDACTED]", 3)


def test_request_context_binding_round_trip() -> None:
    import structlog

    tokens = logging_module.bind_request_context("req-1", "trace-1")
    assert structlog.contextvars.get_contextvars()["request_id"] == "req-1"
    logging_module.reset_request_context(tokens)
    assert "request_id" not in structlog.contextvars.get_contextvars()


# -- Observability stage ----------------------------------------------------


class _Usage:
    def __init__(self) -> None:
        self.recorded: list[ChatResponse] = []
        self.errors: list[Exception] = []

    async def record(self, ctx: RequestContext, response: ChatResponse) -> None:
        self.recorded.append(response)

    async def record_error(self, ctx: RequestContext, error: Exception) -> None:
        self.errors.append(error)


def _state(**settings: Any) -> SimpleNamespace:
    return SimpleNamespace(
        settings=Settings(**settings),
        components={},
        db=None,
        redis=CounterRedis(),
    )


def _ctx(state: SimpleNamespace, **fields: Any) -> RequestContext:
    request = ChatRequest(model="m", messages=[Message(role=Role.USER, content="hi")])
    return RequestContext(request=request, state=state, **fields)  # type: ignore[arg-type]


def _response(finish: FinishReason = FinishReason.STOP) -> ChatResponse:
    return ChatResponse(
        model="m",
        choices=[Choice(message=Message(role=Role.ASSISTANT, content="ok"), finish_reason=finish)],
        usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        cost_usd=0.0,
    )


def test_usage_service_is_resolved_lazily_and_cached() -> None:
    state = _state(pricing_config_path="config/pricing.yaml")
    stage = ObservabilityStage()
    service = stage._usage_service(_ctx(state))
    assert isinstance(service, UsageService)
    assert stage._usage_service(_ctx(state)) is service
    injected = UsageService(None, None, PriceTable(Path("config/pricing.yaml")))
    assert ObservabilityStage(injected)._usage_service(_ctx(state)) is injected


async def test_error_responses_flagged_guardrails_and_cache_results_are_metered() -> None:
    usage = _Usage()
    state = _state()
    ctx = _ctx(state, key_id="k", cache_result="miss", guardrail_flagged=True)
    ctx.errors.append(ProviderError(ErrorCode.PROVIDER_ERROR, "x"))
    before = metrics.ERRORS.labels("m", "unknown", "provider_error")._value.get()
    await ObservabilityStage(usage).finalize(ctx, _response(FinishReason.ERROR))
    after = metrics.ERRORS.labels("m", "unknown", "provider_error")._value.get()
    assert after == before + 1

    plain_error = _ctx(state)
    await ObservabilityStage(usage).finalize(plain_error, _response(FinishReason.ERROR))
    assert len(usage.recorded) == 2


async def test_metrics_can_be_disabled() -> None:
    usage = _Usage()
    state = _state(metrics_enabled=False)
    ctx = _ctx(state, key_id="k")
    await ObservabilityStage(usage).finalize(ctx, _response())
    await ObservabilityStage(usage).on_failure(ctx, RuntimeError("x"))
    assert len(usage.errors) == 1


async def test_on_failure_uses_the_routed_provider() -> None:
    usage = _Usage()
    ctx = _ctx(_state(), key_id="k")
    ctx.routing = RoutingDecision(
        deployment=Deployment(id="d", model_name="m", provider="openai", provider_model="x"),
        strategy="s",
        reason="r",
    )
    before = metrics.ERRORS.labels("m", "openai", "RuntimeError")._value.get()
    await ObservabilityStage(usage).on_failure(ctx, RuntimeError("x"))
    assert metrics.ERRORS.labels("m", "openai", "RuntimeError")._value.get() == before + 1


async def test_token_windows_are_fed_for_limited_keys_and_deployments() -> None:
    state = _state()
    ctx = _ctx(state)
    await _count_tokens(ctx, 10)  # no key and no routing: nothing to count
    assert state.redis.counters == {}

    ctx.virtual_key = SimpleNamespace(id="k", tpm_limit=100)  # type: ignore[assignment]
    ctx.routing = RoutingDecision(
        deployment=Deployment(
            id="d", model_name="m", provider="p", provider_model="x", tpm_limit=100
        ),
        strategy="s",
        reason="r",
    )
    await _count_tokens(ctx, 10)
    assert sorted(key.split(":")[3] for key in state.redis.counters) == ["d", "k"]

    unlimited = _ctx(state)
    unlimited.virtual_key = SimpleNamespace(id="u", tpm_limit=None)  # type: ignore[assignment]
    unlimited.routing = RoutingDecision(
        deployment=Deployment(id="u", model_name="m", provider="p", provider_model="x"),
        strategy="s",
        reason="r",
    )
    counted = dict(state.redis.counters)
    await _count_tokens(unlimited, 10)
    await _count_tokens(ctx, 0)
    assert state.redis.counters == counted
    state.redis = None
    await _count_tokens(ctx, 10)  # no Redis: silently skipped


async def test_spend_is_charged_only_for_virtual_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    charged: list[tuple[str, str | None, float]] = []

    class _Quotas:
        def __init__(self, db: Any, redis: Any) -> None:
            pass

        async def record_spend(self, key_id: str, team_id: str | None, amount: float) -> None:
            if key_id == "broken":
                raise RuntimeError("db down")
            charged.append((key_id, team_id, amount))

    monkeypatch.setattr("app.auth.quotas.QuotaService", _Quotas)
    state = _state()
    await _charge_spend(_ctx(state), 1.0)
    await _charge_spend(_ctx(state, key_id="master"), 1.0)
    await _charge_spend(_ctx(state, key_id="k"), 0.0)
    await _charge_spend(_ctx(state, key_id="k", team_id="t"), 1.5)
    await _charge_spend(_ctx(state, key_id="broken"), 1.0)
    assert charged == [("k", "t", 1.5)]


async def test_successful_cache_hit_skips_token_windows() -> None:
    usage = _Usage()
    state = _state()
    ctx = _ctx(state, key_id="k", cache_hit=True, cache_result="hit")
    ctx.virtual_key = SimpleNamespace(id="k", tpm_limit=100)  # type: ignore[assignment]
    await ObservabilityStage(usage).finalize(ctx, _response())
    assert state.redis.counters == {}
    assert len(usage.recorded) == 1
