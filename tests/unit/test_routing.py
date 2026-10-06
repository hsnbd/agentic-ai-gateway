"""Routing, circuit breaker, and retry/fallback behaviour."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar

import pytest

from app.core.errors import (
    AllProvidersFailedError,
    ErrorCode,
    GatewayError,
    InvalidRequestError,
    NoHealthyDeploymentError,
    ProviderError,
)
from app.core.pipeline import RequestContext
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    Choice,
    FunctionDef,
    Message,
    Role,
    StreamChunk,
    ToolDef,
    Usage,
)
from app.providers.base import Capabilities, Deployment, Pricing
from app.routing.breaker import BreakerState, CircuitBreaker
from app.routing.resilience import ResilientExecutor, RetryPolicy
from app.routing.router import Router
from app.routing.strategies import (
    ConditionalStrategy,
    LeastCostStrategy,
    LowestLatencyStrategy,
    PriorityStrategy,
    WeightedStrategy,
    get_strategy,
)

# --------------------------------------------------------------------------
# Fixtures / fakes
# --------------------------------------------------------------------------


def make_deployment(
    dep_id: str,
    *,
    provider: str = "openai",
    model_name: str = "gpt-4o",
    input_price: float = 1.0,
    output_price: float = 2.0,
    priority: int = 0,
    weight: int = 1,
    context: int = 128000,
    tools: bool = True,
    tags: list[str] | None = None,
) -> Deployment:
    return Deployment(
        id=dep_id,
        model_name=model_name,
        provider=provider,
        provider_model=model_name,
        capabilities=Capabilities(
            tools=tools, streaming=True, max_context_tokens=context, vision=True
        ),
        pricing=Pricing(input_per_mtok=input_price, output_per_mtok=output_price),
        priority=priority,
        weight=weight,
        tags=tags or [],
    )


def make_request(**kw: Any) -> ChatRequest:
    kw.setdefault("model", "gpt-4o")
    kw.setdefault("messages", [Message(role=Role.USER, content="hello")])
    return ChatRequest(**kw)


def make_response(deployment: Deployment) -> ChatResponse:
    return ChatResponse(
        model=deployment.model_name,
        provider=deployment.provider,
        choices=[Choice(index=0, message=Message(role=Role.ASSISTANT, content="hi"))],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class FakeProvider:
    """Replays a scripted sequence of outcomes per call."""

    def __init__(self, name: str, outcomes: list[Any]) -> None:
        self.name = name
        self._outcomes = list(outcomes)
        self.calls = 0

    def _next(self) -> Any:
        self.calls += 1
        if self._outcomes:
            return self._outcomes.pop(0)
        return "ok"

    async def chat(self, request: ChatRequest, deployment: Deployment) -> ChatResponse:
        outcome = self._next()
        if isinstance(outcome, Exception):
            raise outcome
        return make_response(deployment)

    async def stream(
        self, request: ChatRequest, deployment: Deployment
    ) -> AsyncIterator[StreamChunk]:
        outcome = self._next()
        if isinstance(outcome, Exception):
            raise outcome
        yield StreamChunk(model=deployment.model_name, content="hi")
        yield StreamChunk(model=deployment.model_name, finish_reason=None)


class FakeRegistry:
    def __init__(self, deployments: list[Deployment], providers: dict[str, FakeProvider]) -> None:
        self._deployments = deployments
        self._providers = providers

    def deployments_for(self, model: str, *, include_disabled: bool = False) -> list[Deployment]:
        found = [d for d in self._deployments if d.model_name == model]
        if not found:
            from app.core.errors import NotFoundError

            raise NotFoundError(f"Model {model!r} is not configured on this gateway")
        return found

    def provider_for(self, deployment: Deployment) -> FakeProvider:
        return self._providers[deployment.provider]


class FakeState:
    pass


def make_ctx(request: ChatRequest | None = None, **kw: Any) -> RequestContext:
    return RequestContext(request=request or make_request(), state=FakeState(), **kw)  # type: ignore[arg-type]


def retryable(code: ErrorCode = ErrorCode.PROVIDER_TIMEOUT) -> ProviderError:
    return ProviderError(code, "boom", provider="openai")


# --------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------


def test_breaker_opens_after_threshold_failures() -> None:
    breaker = CircuitBreaker(threshold=3, cooldown_seconds=30)

    for _ in range(2):
        breaker.record_failure("d1")
    assert breaker.is_available("d1")
    assert breaker.health("d1").state is BreakerState.CLOSED

    breaker.record_failure("d1")
    assert breaker.health("d1").state is BreakerState.OPEN
    assert not breaker.is_available("d1")


def test_breaker_half_opens_after_cooldown_and_closes_on_success() -> None:
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=10)
    breaker.record_failure("d1", now=100.0)
    assert not breaker.is_available("d1", now=105.0)

    # Cooldown elapsed: exactly one probe is admitted.
    assert breaker.is_available("d1", now=111.0)
    assert breaker.health("d1").state is BreakerState.HALF_OPEN
    assert not breaker.is_available("d1", now=111.0)

    breaker.record_success("d1", latency_ms=50)
    assert breaker.health("d1").state is BreakerState.CLOSED
    assert breaker.is_available("d1")


def test_failed_half_open_probe_reopens_immediately() -> None:
    breaker = CircuitBreaker(threshold=5, cooldown_seconds=10)
    breaker.record_failure("d1", now=100.0)
    breaker.health("d1").state = BreakerState.HALF_OPEN

    breaker.record_failure("d1", now=120.0)
    assert breaker.health("d1").state is BreakerState.OPEN
    assert breaker.health("d1").consecutive_failures == 2


def test_success_resets_consecutive_failures() -> None:
    breaker = CircuitBreaker(threshold=3)
    breaker.record_failure("d1")
    breaker.record_failure("d1")
    breaker.record_success("d1", latency_ms=10)
    assert breaker.health("d1").consecutive_failures == 0
    assert breaker.health("d1").total_requests == 3
    assert breaker.health("d1").failure_rate == pytest.approx(2 / 3)


def test_latency_ewma_tracks_recent_samples() -> None:
    breaker = CircuitBreaker()
    for _ in range(5):
        breaker.record_success("d1", latency_ms=100)
    assert breaker.health("d1").ewma_latency_ms == pytest.approx(100.0)

    breaker.record_success("d1", latency_ms=600)
    # Weighted toward history, so one spike must not dominate.
    assert 100 < breaker.health("d1").ewma_latency_ms < 250


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


def test_least_cost_picks_cheapest() -> None:
    cheap = make_deployment("cheap", input_price=0.15, output_price=0.6)
    pricey = make_deployment("pricey", input_price=5.0, output_price=15.0)
    chosen, reason = LeastCostStrategy().select([pricey, cheap], make_request(), CircuitBreaker())
    assert chosen.id == "cheap"
    assert "cheapest" in reason


def test_least_cost_order_is_ascending() -> None:
    a = make_deployment("a", input_price=3.0)
    b = make_deployment("b", input_price=1.0)
    c = make_deployment("c", input_price=2.0)
    order = LeastCostStrategy().order([a, b, c], make_request(), CircuitBreaker())
    assert [d.id for d in order] == ["b", "c", "a"]


def test_lowest_latency_probes_unmeasured_first() -> None:
    fast = make_deployment("fast")
    fresh = make_deployment("fresh")
    breaker = CircuitBreaker()
    breaker.record_success("fast", latency_ms=20)

    chosen, reason = LowestLatencyStrategy().select([fast, fresh], make_request(), breaker)
    assert chosen.id == "fresh"
    assert "no latency samples" in reason


def test_lowest_latency_prefers_faster_once_measured() -> None:
    fast = make_deployment("fast")
    slow = make_deployment("slow")
    breaker = CircuitBreaker()
    breaker.record_success("fast", latency_ms=20)
    breaker.record_success("slow", latency_ms=900)

    chosen, _ = LowestLatencyStrategy().select([slow, fast], make_request(), breaker)
    assert chosen.id == "fast"


def test_priority_strategy_prefers_the_lowest_number() -> None:
    """`priority: 1` must mean "first choice", as it does in every other
    fallback list users have configured."""
    first = make_deployment("first", priority=1, weight=1)
    second = make_deployment("second", priority=5, weight=9)
    chosen, _ = PriorityStrategy().select([first, second], make_request(), CircuitBreaker())
    assert chosen.id == "first"

    order = PriorityStrategy().order([first, second], make_request(), CircuitBreaker())
    assert [d.id for d in order] == ["first", "second"]


def test_priority_ties_break_on_weight() -> None:
    light = make_deployment("light", priority=1, weight=1)
    heavy = make_deployment("heavy", priority=1, weight=9)
    order = PriorityStrategy().order([light, heavy], make_request(), CircuitBreaker())
    assert [d.id for d in order] == ["heavy", "light"]


def test_weighted_strategy_respects_distribution() -> None:
    heavy = make_deployment("heavy", weight=95)
    light = make_deployment("light", weight=5)
    strategy = WeightedStrategy()
    breaker = CircuitBreaker()

    picks = [strategy.select([heavy, light], make_request(), breaker)[0].id for _ in range(500)]
    assert picks.count("heavy") > picks.count("light") * 3


def test_weighted_order_contains_every_candidate_once() -> None:
    deployments = [make_deployment(f"d{i}", weight=i + 1) for i in range(4)]
    order = WeightedStrategy().order(deployments, make_request(), CircuitBreaker())
    assert sorted(d.id for d in order) == ["d0", "d1", "d2", "d3"]


def test_conditional_routes_long_prompts_to_large_context() -> None:
    small = make_deployment("small", context=8000, input_price=0.1)
    large = make_deployment("large", context=200000, input_price=5.0)
    long_prompt = "word " * 20000  # ~25k estimated tokens

    chosen, reason = ConditionalStrategy().select(
        [small, large],
        make_request(messages=[Message(role=Role.USER, content=long_prompt)]),
        CircuitBreaker(),
    )
    assert chosen.id == "large"
    assert "context" in reason


def test_conditional_routes_tool_requests_to_tool_capable() -> None:
    no_tools = make_deployment("no_tools", tools=False, input_price=0.01)
    with_tools = make_deployment("with_tools", tools=True, input_price=5.0)

    request = make_request(
        tools=[ToolDef(function=FunctionDef(name="get_weather", parameters={"type": "object"}))]
    )
    chosen, reason = ConditionalStrategy().select([no_tools, with_tools], request, CircuitBreaker())
    assert chosen.id == "with_tools"
    assert "tool" in reason


def test_conditional_prefers_cheap_tagged_for_short_prompts() -> None:
    cheap = make_deployment("cheap", tags=["cheap"], input_price=3.0)
    other = make_deployment("other", input_price=1.0)
    chosen, reason = ConditionalStrategy().select([other, cheap], make_request(), CircuitBreaker())
    assert chosen.id == "cheap"
    assert "cheap" in reason


def test_conditional_prefers_deployments_matching_request_tags() -> None:
    frontier = make_deployment("frontier", tags=["frontier"], input_price=10.0)
    eu = make_deployment("eu", tags=["eu", "cheap"], input_price=2.0)
    eu_backup = make_deployment("eu_backup", tags=["eu"], input_price=3.0)
    plain = make_deployment("plain", input_price=0.1)
    candidates = [plain, frontier, eu_backup, eu]

    chosen, reason = ConditionalStrategy().select(
        candidates, make_request(tags=["eu"]), CircuitBreaker()
    )
    assert chosen.id == "eu", "cheapest deployment sharing a tag with the request"
    assert "eu" in reason

    order = ConditionalStrategy().order(candidates, make_request(tags=["eu"]), CircuitBreaker())
    assert [d.id for d in order][:2] == ["eu", "eu_backup"], "other matches fall back first"

    unmatched, _ = ConditionalStrategy().select(
        candidates, make_request(tags=["nowhere"]), CircuitBreaker()
    )
    assert unmatched.id == "eu", "no tag match: the usual heuristics apply (cheap tag)"


def test_get_strategy_resolves_aliases_and_defaults() -> None:
    assert get_strategy("cheapest").name == "least-cost"
    assert get_strategy("fastest").name == "lowest-latency"
    assert get_strategy(None).name == "priority"
    assert get_strategy("nonsense").name == "priority"


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------


def test_router_filters_by_capability() -> None:
    no_tools = make_deployment("no_tools", tools=False)
    with_tools = make_deployment("with_tools", tools=True)
    registry = FakeRegistry([no_tools, with_tools], {})
    router = Router(registry, CircuitBreaker())  # type: ignore[arg-type]

    ctx = make_ctx(
        make_request(tools=[ToolDef(function=FunctionDef(name="f", parameters={"type": "object"}))])
    )
    decision, chain = router.route(ctx)
    assert [d.id for d in chain] == ["with_tools"]
    assert "no_tools" in decision.candidates_rejected


def test_router_raises_when_no_deployment_is_capable() -> None:
    registry = FakeRegistry([make_deployment("d1", tools=False)], {})
    router = Router(registry, CircuitBreaker())  # type: ignore[arg-type]
    ctx = make_ctx(
        make_request(tools=[ToolDef(function=FunctionDef(name="f", parameters={"type": "object"}))])
    )
    with pytest.raises(NoHealthyDeploymentError):
        router.route(ctx)


def test_router_skips_broken_deployments() -> None:
    healthy = make_deployment("healthy")
    broken = make_deployment("broken")
    breaker = CircuitBreaker(threshold=1)
    breaker.record_failure("broken")

    router = Router(FakeRegistry([broken, healthy], {}), breaker)  # type: ignore[arg-type]
    _, chain = router.route(make_ctx())
    assert [d.id for d in chain] == ["healthy"]


def test_router_serves_traffic_when_everything_is_broken() -> None:
    """A total outage must not become a permanent self-inflicted refusal."""
    d1 = make_deployment("d1")
    breaker = CircuitBreaker(threshold=1)
    breaker.record_failure("d1")

    router = Router(FakeRegistry([d1], {}), breaker)  # type: ignore[arg-type]
    decision, chain = router.route(make_ctx())
    assert [d.id for d in chain] == ["d1"]
    assert "circuit breaker open" in decision.candidates_rejected["d1"]


def test_router_appends_explicit_fallback_models() -> None:
    primary = make_deployment("primary", model_name="gpt-4o")
    backup = make_deployment("backup", model_name="claude-sonnet-4", provider="anthropic")
    router = Router(FakeRegistry([primary, backup], {}), CircuitBreaker())  # type: ignore[arg-type]

    ctx = make_ctx(make_request(fallbacks=["claude-sonnet-4"], routing_strategy="priority"))
    _, chain = router.route(ctx)
    assert {d.id for d in chain} == {"primary", "backup"}


def test_router_ignores_unknown_fallback_models() -> None:
    primary = make_deployment("primary")
    router = Router(FakeRegistry([primary], {}), CircuitBreaker())  # type: ignore[arg-type]

    ctx = make_ctx(make_request(fallbacks=["does-not-exist"]))
    decision, chain = router.route(ctx)
    assert [d.id for d in chain] == ["primary"]
    assert "does-not-exist" in decision.candidates_rejected


def test_router_enforces_virtual_key_allowlist() -> None:
    allowed = make_deployment("allowed", model_name="gpt-4o")
    router = Router(FakeRegistry([allowed], {}), CircuitBreaker())  # type: ignore[arg-type]

    class Key:
        allowed_models: ClassVar[list[str]] = ["claude-sonnet-4"]
        blocked_models: ClassVar[list[str]] = []

    ctx = make_ctx()
    ctx.virtual_key = Key()  # type: ignore[assignment]
    with pytest.raises(NoHealthyDeploymentError, match="not permitted"):
        router.route(ctx)


def test_router_honours_request_strategy_override() -> None:
    cheap = make_deployment("cheap", input_price=0.1, priority=0)
    pricey = make_deployment("pricey", input_price=9.0, priority=10)
    router = Router(FakeRegistry([pricey, cheap], {}), CircuitBreaker(), "priority")  # type: ignore[arg-type]

    decision, _ = router.route(make_ctx(make_request(routing_strategy="least-cost")))
    assert decision.deployment.id == "cheap"
    assert decision.strategy == "least-cost"


# --------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------


def test_backoff_grows_exponentially_and_is_capped() -> None:
    policy = RetryPolicy(initial_backoff=1.0, multiplier=2.0, max_backoff=8.0, jitter=False)
    assert policy.backoff(1) == 1.0
    assert policy.backoff(2) == 2.0
    assert policy.backoff(3) == 4.0
    assert policy.backoff(10) == 8.0


def test_backoff_honours_retry_after() -> None:
    policy = RetryPolicy(initial_backoff=1.0, max_backoff=30.0, jitter=False)
    assert policy.backoff(1, retry_after=12.0) == 12.0
    # ...but never beyond our own ceiling.
    assert policy.backoff(1, retry_after=999.0) == 30.0


def test_jitter_stays_within_bounds() -> None:
    policy = RetryPolicy(initial_backoff=4.0, multiplier=1.0, max_backoff=4.0, jitter=True)
    delays = [policy.backoff(1) for _ in range(200)]
    assert all(0 <= d <= 4.0 for d in delays)
    assert len(set(delays)) > 1


# --------------------------------------------------------------------------
# Executor: retries and fallbacks
# --------------------------------------------------------------------------


def build_executor(
    deployments: list[Deployment],
    providers: dict[str, FakeProvider],
    *,
    breaker: CircuitBreaker | None = None,
    policy: RetryPolicy | None = None,
) -> ResilientExecutor:
    breaker = breaker or CircuitBreaker()
    registry = FakeRegistry(deployments, providers)
    router = Router(registry, breaker)  # type: ignore[arg-type]
    return ResilientExecutor(
        registry,  # type: ignore[arg-type]
        router,
        breaker,
        policy or RetryPolicy(max_attempts=3, initial_backoff=0.0, jitter=False),
    )


async def test_successful_call_needs_one_attempt() -> None:
    d1 = make_deployment("d1")
    provider = FakeProvider("openai", [])
    executor = build_executor([d1], {"openai": provider})

    ctx = make_ctx()
    response = await executor.execute(ctx)

    assert response.deployment_id == "d1"
    assert ctx.attempt_count == 1
    assert ctx.fallback_used is False


async def test_retries_same_deployment_on_retryable_error() -> None:
    d1 = make_deployment("d1")
    provider = FakeProvider("openai", [retryable(), retryable()])
    executor = build_executor([d1], {"openai": provider})

    ctx = make_ctx()
    response = await executor.execute(ctx)

    assert provider.calls == 3
    assert ctx.attempt_count == 3
    assert ctx.attempted == ["d1", "d1", "d1"]
    assert response.deployment_id == "d1"
    # Retrying the same deployment is not a fallback.
    assert ctx.routing is not None


async def test_non_retryable_error_aborts_immediately() -> None:
    d1 = make_deployment("d1")
    provider = FakeProvider("openai", [InvalidRequestError("bad tool schema")])
    executor = build_executor([d1], {"openai": provider})

    with pytest.raises(InvalidRequestError):
        await executor.execute(make_ctx())
    assert provider.calls == 1


async def test_falls_back_to_next_deployment_after_retries_exhaust() -> None:
    primary = make_deployment("primary", provider="openai", priority=1)
    backup = make_deployment("backup", provider="anthropic", priority=2)
    failing = FakeProvider("openai", [retryable(), retryable(), retryable()])
    working = FakeProvider("anthropic", [])

    executor = build_executor([primary, backup], {"openai": failing, "anthropic": working})
    ctx = make_ctx()
    response = await executor.execute(ctx)

    assert failing.calls == 3
    assert working.calls == 1
    assert response.deployment_id == "backup"
    assert ctx.fallback_used is True
    assert ctx.attempted == ["primary", "primary", "primary", "backup"]


async def test_context_length_error_falls_back_without_retrying() -> None:
    """Retrying an oversized prompt is pointless; a bigger model is the fix."""
    small = make_deployment("small", provider="openai", priority=1)
    large = make_deployment("large", provider="anthropic", priority=2)
    overflow = ProviderError(
        ErrorCode.CONTEXT_LENGTH_EXCEEDED, "maximum context length", provider="openai"
    )
    failing = FakeProvider("openai", [overflow])
    working = FakeProvider("anthropic", [])

    executor = build_executor([small, large], {"openai": failing, "anthropic": working})
    ctx = make_ctx()
    response = await executor.execute(ctx)

    assert failing.calls == 1
    assert response.deployment_id == "large"
    assert ctx.fallback_used is True


async def test_all_providers_failed_carries_diagnostics() -> None:
    d1 = make_deployment("d1", provider="openai", priority=1)
    d2 = make_deployment("d2", provider="anthropic", priority=2)
    executor = build_executor(
        [d1, d2],
        {
            "openai": FakeProvider("openai", [retryable()] * 3),
            "anthropic": FakeProvider("anthropic", [retryable()] * 3),
        },
    )

    ctx = make_ctx()
    with pytest.raises(AllProvidersFailedError) as excinfo:
        await executor.execute(ctx)

    error = excinfo.value
    assert error.details["last_error_code"] == ErrorCode.PROVIDER_TIMEOUT.value
    assert set(error.details["attempted"]) == {"d1", "d2"}


async def test_breaker_opens_after_repeated_executor_failures() -> None:
    d1 = make_deployment("d1")
    breaker = CircuitBreaker(threshold=3, cooldown_seconds=60)
    executor = build_executor(
        [d1], {"openai": FakeProvider("openai", [retryable()] * 3)}, breaker=breaker
    )

    with pytest.raises(GatewayError):
        await executor.execute(make_ctx())

    assert breaker.health("d1").state is BreakerState.OPEN


async def test_max_fallbacks_bounds_the_chain() -> None:
    deployments = [make_deployment(f"d{i}", provider=f"p{i}", priority=i) for i in range(6)]
    providers = {f"p{i}": FakeProvider(f"p{i}", [retryable()] * 3) for i in range(6)}
    breaker = CircuitBreaker()
    registry = FakeRegistry(deployments, providers)
    executor = ResilientExecutor(
        registry,  # type: ignore[arg-type]
        Router(registry, breaker),  # type: ignore[arg-type]
        breaker,
        RetryPolicy(max_attempts=1, initial_backoff=0.0, jitter=False),
        max_fallbacks=2,
    )

    ctx = make_ctx()
    with pytest.raises(AllProvidersFailedError):
        await executor.execute(ctx)

    # 1 primary + 2 fallbacks, not all six.
    assert len(set(ctx.attempted)) == 3


# --------------------------------------------------------------------------
# Executor: streaming
# --------------------------------------------------------------------------


async def test_stream_falls_back_before_first_chunk() -> None:
    primary = make_deployment("primary", provider="openai", priority=1)
    backup = make_deployment("backup", provider="anthropic", priority=2)
    executor = build_executor(
        [primary, backup],
        {
            "openai": FakeProvider("openai", [retryable()]),
            "anthropic": FakeProvider("anthropic", []),
        },
    )

    ctx = make_ctx(make_request(stream=True))
    chunks = [c async for c in executor.execute_stream(ctx)]

    assert [c.deployment_id for c in chunks] == ["backup", "backup"]
    assert ctx.fallback_used is True
    assert ctx.time_to_first_token_ms is not None


async def test_stream_error_after_first_chunk_is_not_masked() -> None:
    """Switching providers mid-stream would splice two generations together."""

    class MidStreamFailure(FakeProvider):
        async def stream(
            self, request: ChatRequest, deployment: Deployment
        ) -> AsyncIterator[StreamChunk]:
            self.calls += 1
            yield StreamChunk(model=deployment.model_name, content="par")
            raise retryable()

    primary = make_deployment("primary", provider="openai", priority=1)
    backup = make_deployment("backup", provider="anthropic", priority=2)
    backup_provider = FakeProvider("anthropic", [])
    executor = build_executor(
        [primary, backup],
        {"openai": MidStreamFailure("openai", []), "anthropic": backup_provider},
    )

    ctx = make_ctx(make_request(stream=True))
    received: list[StreamChunk] = []
    with pytest.raises(ProviderError):
        async for chunk in executor.execute_stream(ctx):
            received.append(chunk)

    assert len(received) == 1
    assert backup_provider.calls == 0
