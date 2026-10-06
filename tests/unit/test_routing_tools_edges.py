"""Edge cases for routing, resilience, and the MCP tool loop."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.errors import (
    AllProvidersFailedError,
    ErrorCode,
    InvalidRequestError,
    NoHealthyDeploymentError,
    ProviderError,
    RateLimitExceededError,
)
from app.core.pipeline import RequestContext
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    Choice,
    FinishReason,
    FunctionDef,
    Message,
    Role,
    StreamChunk,
    ToolCall,
    ToolDef,
)
from app.providers.base import Capabilities, Deployment
from app.routing import strategies
from app.routing.breaker import CircuitBreaker
from app.routing.resilience import ResilientExecutor, RetryPolicy
from app.routing.router import Router
from app.routing.strategies import (
    ConditionalStrategy,
    LowestLatencyStrategy,
    WeightedStrategy,
    list_strategies,
)
from app.tools.executor import ToolExecutor, _normalized_arguments, run_tool_loop
from app.tools.validation import validate_arguments
from tests.unit.test_auth_edges import CounterRedis


def _dep(id: str, **fields: Any) -> Deployment:
    return Deployment(id=id, model_name="m", provider="p", provider_model=id, **fields)


def _ctx(**request: Any) -> RequestContext:
    chat = ChatRequest(model="m", messages=[Message(role=Role.USER, content="hi")], **request)
    return RequestContext(request=chat, state=SimpleNamespace())  # type: ignore[arg-type]


class _Registry:
    def __init__(self, deployments: list[Deployment], provider: Any = None) -> None:
        self.deployments = deployments
        self.provider = provider

    def deployments_for(self, model: str) -> list[Deployment]:
        return list(self.deployments)

    def provider_for(self, deployment: Deployment) -> Any:
        return self.provider


# -- Router -----------------------------------------------------------------


def test_router_keeps_chain_consistent_with_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = _dep("a"), _dep("b")
    calls = 0

    def choices(population: list[Deployment], weights: Any = None, k: int = 1) -> list[Any]:
        nonlocal calls
        calls += 1
        # order() samples first and picks "a"; select() then picks "b".
        return [population[0] if calls == 1 else population[-1]]

    monkeypatch.setattr(strategies.random, "choices", choices)
    router = Router(_Registry([a, b]), CircuitBreaker(), default_strategy="weighted")  # type: ignore[arg-type]
    decision, chain = router.route(_ctx())
    assert decision.deployment.id == chain[0].id


def test_router_applies_key_blocklist() -> None:
    router = Router(_Registry([_dep("a"), _dep("b")]), CircuitBreaker())  # type: ignore[arg-type]
    ctx = _ctx()
    ctx.virtual_key = SimpleNamespace(allowed_models=[], blocked_models=["a"])  # type: ignore[assignment]
    decision, chain = router.route(ctx)
    assert [d.id for d in chain] == ["b"]
    assert decision.candidates_rejected["a"] == "model is blocked for this virtual key"


# -- Strategies -------------------------------------------------------------


def test_default_order_and_strategy_listing() -> None:
    class First(strategies.RoutingStrategy):
        name = "first"

        def select(self, candidates: list[Deployment], request: Any, breaker: Any) -> Any:
            return candidates[-1], "last"

    ordered = First().order([_dep("a"), _dep("b")], _ctx().request, CircuitBreaker())
    assert [d.id for d in ordered] == ["b", "a"]
    assert "priority" in list_strategies()


def test_weighted_strategy_with_zero_weights_is_uniform() -> None:
    chosen, reason = WeightedStrategy().select(
        [_dep("a", weight=0), _dep("b", weight=0)], _ctx().request, CircuitBreaker()
    )
    assert chosen.id in {"a", "b"}
    assert "uniformly" in reason


def test_lowest_latency_reports_measured_latency() -> None:
    breaker = CircuitBreaker()
    breaker.record_success("a", 50.0)
    breaker.record_success("b")
    chosen, reason = LowestLatencyStrategy().select([_dep("a")], _ctx().request, breaker)
    assert "EWMA" in reason and chosen.id == "a"


def test_conditional_strategy_fallthroughs() -> None:
    long_prompt = _ctx()
    long_prompt.request.messages[0].content = "x" * 80_000  # ~20k tokens
    small = [_dep("small", capabilities=Capabilities(max_context_tokens=8000))]
    _, reason = ConditionalStrategy().select(small, long_prompt.request, CircuitBreaker())
    assert "cheapest" in reason  # no large-context deployment, so least-cost

    tool = ToolDef(function=FunctionDef(name="t"))
    tooled = _ctx(tools=[tool])
    _, reason = ConditionalStrategy().select(
        [_dep("plain", capabilities=Capabilities(tools=False))], tooled.request, CircuitBreaker()
    )
    assert "cheapest" in reason


# -- Breaker ----------------------------------------------------------------


def test_breaker_reset_forgets_health() -> None:
    breaker = CircuitBreaker(threshold=1)
    breaker.record_failure("a", "boom")
    assert not breaker.is_available("a")
    breaker.reset("a")
    assert "a" not in breaker.snapshot()
    assert breaker.is_available("a")


# -- Resilient executor -----------------------------------------------------


class _Provider:
    def __init__(self, chunks: list[StreamChunk] | None = None) -> None:
        self.chunks = chunks or [StreamChunk(model="m", content="ok")]

    async def chat(self, request: ChatRequest, deployment: Deployment) -> ChatResponse:
        return ChatResponse(
            model="m", choices=[Choice(message=Message(role=Role.ASSISTANT, content="ok"))]
        )

    async def stream(
        self, request: ChatRequest, deployment: Deployment
    ) -> AsyncIterator[StreamChunk]:
        for chunk in self.chunks:
            yield chunk


def _executor(deployments: list[Deployment], redis: Any = None, **kwargs: Any) -> ResilientExecutor:
    registry = _Registry(deployments, _Provider())
    breaker = CircuitBreaker()
    return ResilientExecutor(
        registry,  # type: ignore[arg-type]
        Router(registry, breaker),  # type: ignore[arg-type]
        breaker,
        RetryPolicy(max_attempts=1, jitter=False),
        redis=redis,
        **kwargs,
    )


async def test_streams_skip_rate_limited_deployments() -> None:
    redis = CounterRedis()
    saturated = _dep("saturated", rpm_limit=1, priority=0)
    spare = _dep("spare", priority=1)
    executor = _executor([saturated, spare], redis)
    await executor._admit(_ctx(), saturated)  # uses the only slot
    ctx = _ctx()
    chunks = [chunk async for chunk in executor.execute_stream(ctx)]
    assert chunks[0].deployment_id == "spare"
    assert ctx.routing is not None and ctx.routing.deployment.id == "spare"


async def test_admit_without_routing_decision_still_records_the_error() -> None:
    redis = CounterRedis()
    limited = _dep("limited", rpm_limit=1)
    executor = _executor([limited], redis)
    ctx = _ctx()
    assert await executor._admit(ctx, limited) is None
    error = await executor._admit(ctx, limited)
    assert isinstance(error, RateLimitExceededError)
    assert ctx.errors == [error]


def test_note_fallback_without_routing_decision() -> None:
    executor = _executor([_dep("a")])
    ctx = _ctx()
    executor._note_fallback(ctx, _dep("a"), _dep("b"), None)
    assert ctx.fallback_used and ctx.routing is None


async def test_empty_chain_reports_no_healthy_deployment() -> None:
    executor = _executor([_dep("a")], max_fallbacks=-1)
    with pytest.raises(NoHealthyDeploymentError):
        await executor.execute(_ctx())


def test_exhausted_mixes_rate_limits_with_other_errors() -> None:
    executor = _executor([_dep("a")])
    ctx = _ctx()
    ctx.errors = [ProviderError(ErrorCode.PROVIDER_ERROR, "x"), RateLimitExceededError("y")]
    error = executor._exhausted(ctx, RateLimitExceededError("y"))
    assert isinstance(error, AllProvidersFailedError)


def test_backoff_honours_retry_after_and_caps() -> None:
    policy = RetryPolicy(initial_backoff=1, max_backoff=4, jitter=False)
    assert policy.backoff(1, retry_after=10) == 4
    assert policy.backoff(5) == 4
    assert 0 <= RetryPolicy(initial_backoff=1).backoff(1) <= 1


# -- Tool argument validation ----------------------------------------------


def _tool_def(parameters: dict[str, Any]) -> ToolDef:
    return ToolDef(function=FunctionDef(name="t", parameters=parameters))


@pytest.mark.parametrize(
    ("raw", "parsed"),
    [
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('{"a": 1,}', {"a": 1}),
        ("{a: 1}", {"a": 1}),
        ("{'a': (1, 2)}", {"a": (1, 2)}),
        ("{a: hello world}", {"a": "hello world"}),
        ("{a: true}", {"a": True}),
    ],
)
def test_argument_repair(raw: str, parsed: dict[str, Any]) -> None:
    assert validate_arguments(_tool_def({}), raw) == parsed


def test_argument_validation_errors() -> None:
    assert "not valid JSON" in validate_arguments(_tool_def({}), "{a: [}")["error"]
    assert "must be a JSON object" in validate_arguments(_tool_def({}), "[1]")["error"]
    schema = {
        "type": "object",
        "required": ["name", 5],
        "properties": {
            "name": {"type": "string"},
            "size": {"type": ["integer", "null"]},
            "mode": {"enum": ["a", "b"]},
            "flag": {"type": "integer"},
            "custom": {"type": "uuid"},
            "nested": {"type": "object", "required": ["inner"]},
            "ignored": "not-a-schema",
        },
    }
    errors = validate_arguments(
        _tool_def(schema),
        '{"size": null, "mode": "c", "flag": true, "custom": "x", "nested": {}, "ignored": 1}',
    )["error"]
    assert "arguments.name is required" in errors
    assert "arguments.mode must be one of" in errors
    assert "arguments.flag must be integer" in errors
    assert "arguments.nested.inner is required" in errors
    top_enum = validate_arguments(_tool_def({"enum": [{"a": 1}]}), '{"a": 2}')
    assert "arguments must be one of" in top_enum["error"]
    odd = validate_arguments(_tool_def({"required": "name", "properties": []}), "{}")
    assert odd == {}


# -- Tool loop --------------------------------------------------------------


class _McpRegistry:
    def __init__(self, tools: list[ToolDef]) -> None:
        self.tools = tools

    async def tools_for(self, server_ids: Any = None) -> list[ToolDef]:
        return self.tools

    def server_name(self, server_id: str) -> str:
        return server_id

    def resolve(self, name: str) -> tuple[str, str]:
        return "srv", name

    async def call_tool(
        self, server_id: str, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "done"}]}


async def test_tool_executor_isolates_unexpected_failures() -> None:
    executor = ToolExecutor(_McpRegistry([]))  # type: ignore[arg-type]

    async def explode(call: ToolCall) -> Any:
        raise KeyboardInterrupt

    executor.execute_detailed = explode  # type: ignore[method-assign]
    [(message, record)] = await executor.execute_all_detailed([ToolCall(id="c", name="t")])
    assert record["is_error"] and "Tool execution failed" in message.text()


async def test_tool_executor_single_call_and_validation_error() -> None:
    tool = _tool_def({"type": "object", "required": ["q"], "properties": {}})
    executor = ToolExecutor(_McpRegistry([tool]))  # type: ignore[arg-type]
    ok = await executor.execute(ToolCall(id="c", name="t", arguments='{"q": 1}'))
    assert ok.text() == "done"
    [invalid] = await executor.execute_all([ToolCall(id="c", name="t", arguments="{}")])
    assert "required" in invalid.text()


async def test_tool_loop_requires_an_iteration_and_stops_on_empty_choices() -> None:
    executor = ToolExecutor(_McpRegistry([]))  # type: ignore[arg-type]

    async def call_model(ctx: RequestContext) -> ChatResponse:
        return ChatResponse(model="m", choices=[])

    with pytest.raises(InvalidRequestError, match="at least 1"):
        await run_tool_loop(_ctx(), call_model, executor, max_iterations=0)
    ctx = _ctx()
    await run_tool_loop(ctx, call_model, executor, max_iterations=2)
    assert ctx.stop_reason == "completed"


async def test_tool_loop_executes_calls_until_the_model_answers() -> None:
    tool = _tool_def({})
    executor = ToolExecutor(_McpRegistry([tool]))  # type: ignore[arg-type]
    replies = [
        Choice(
            message=Message(role=Role.ASSISTANT, tool_calls=[ToolCall(id="1", name="t")]),
            finish_reason=FinishReason.TOOL_CALLS,
        ),
        Choice(message=Message(role=Role.ASSISTANT, content="answer")),
    ]

    async def call_model(ctx: RequestContext) -> ChatResponse:
        return ChatResponse(model="m", choices=[replies.pop(0)])

    ctx = _ctx()
    response = await run_tool_loop(ctx, call_model, executor, max_iterations=3)
    assert response.text == "answer"
    assert ctx.stop_reason == "completed"
    assert len(ctx.tool_executions) == 1


def test_normalized_arguments_tolerates_invalid_json() -> None:
    assert _normalized_arguments(" not json ") == "not json"
    assert _normalized_arguments('{"b":1, "a":2}') == '{"a":2,"b":1}'


def test_router_deduplicates_fallback_deployments() -> None:
    router = Router(_Registry([_dep("a")]), CircuitBreaker())  # type: ignore[arg-type]
    _, chain = router.route(_ctx(fallbacks=["m"]))
    assert [d.id for d in chain] == ["a"]


async def test_tool_executor_reports_unknown_tools() -> None:
    executor = ToolExecutor(_McpRegistry([]))  # type: ignore[arg-type]
    message, is_error = await executor.execute_detailed(ToolCall(id="c", name="missing"))
    assert is_error and "unavailable" in message.text()
