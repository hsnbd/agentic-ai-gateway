"""Request pipeline.

A request flows through ordered stages. Each stage may mutate the context,
short-circuit with a response (e.g. a cache hit), or raise a GatewayError.
Stages are deliberately unaware of provider specifics.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from app.core.schemas import ChatRequest, ChatResponse, StreamChunk

if TYPE_CHECKING:
    from app.core.state import GatewayState
    from app.db.models import VirtualKey
    from app.providers.base import Deployment


@dataclass
class RoutingDecision:
    """Which deployment was chosen, and why — surfaced in logs and the console."""

    deployment: Deployment
    strategy: str
    reason: str
    candidates_considered: int = 0
    candidates_rejected: dict[str, str] = field(default_factory=dict)


@dataclass
class RequestContext:
    """Mutable state carried through every stage of one request."""

    request: ChatRequest
    state: GatewayState

    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:24]}")
    started_at: float = field(default_factory=time.perf_counter)
    dialect: str = "openai"
    route: str = "/v1/chat/completions"

    # Auth
    virtual_key: VirtualKey | None = None
    key_id: str | None = None
    team_id: str | None = None
    end_user: str | None = None

    # Routing / execution
    routing: RoutingDecision | None = None
    attempted: list[str] = field(default_factory=list)
    attempt_count: int = 0
    fallback_used: bool = False
    errors: list[Exception] = field(default_factory=list)

    # Cache
    cache_hit: bool = False
    cache_similarity: float | None = None
    cache_key: str | None = None
    cost_saved_usd: float = 0.0

    # Guardrails
    guardrail_flagged: bool = False
    guardrail_results: dict[str, Any] = field(default_factory=dict)

    # Result
    response: ChatResponse | None = None
    cost_usd: float = 0.0
    time_to_first_token_ms: float | None = None

    # Diagnostics
    stage_timings: dict[str, float] = field(default_factory=dict)
    trace_id: str | None = None

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000

    def record_stage(self, name: str, started: float) -> None:
        self.stage_timings[name] = round((time.perf_counter() - started) * 1000, 3)

    def record_attempt(self, deployment_id: str) -> None:
        self.attempted.append(deployment_id)
        self.attempt_count += 1
        if len(self.attempted) > 1:
            self.fallback_used = True

    @property
    def model(self) -> str:
        return self.request.model


@runtime_checkable
class Stage(Protocol):
    """A pipeline stage.

    Return a ChatResponse to short-circuit the pipeline, or None to continue.
    """

    name: str

    async def process(self, ctx: RequestContext) -> ChatResponse | None: ...


@runtime_checkable
class PostStage(Protocol):
    """Runs after a response exists, for guardrails, caching, and accounting."""

    name: str

    async def finalize(self, ctx: RequestContext, response: ChatResponse) -> ChatResponse: ...


class Pipeline:
    """Orchestrates pre-stages, execution, and post-stages."""

    def __init__(
        self,
        pre_stages: list[Stage],
        executor: Executor,
        post_stages: list[PostStage],
    ) -> None:
        self.pre_stages = pre_stages
        self.executor = executor
        self.post_stages = post_stages

    async def run(self, ctx: RequestContext) -> ChatResponse:
        short_circuit: ChatResponse | None = None

        for stage in self.pre_stages:
            started = time.perf_counter()
            try:
                result = await stage.process(ctx)
            finally:
                ctx.record_stage(stage.name, started)
            if result is not None:
                short_circuit = result
                break

        if short_circuit is not None:
            response = short_circuit
        else:
            started = time.perf_counter()
            try:
                response = await self.executor.execute(ctx)
            finally:
                ctx.record_stage("execute", started)

        for post in self.post_stages:
            started = time.perf_counter()
            try:
                response = await post.finalize(ctx, response)
            finally:
                ctx.record_stage(post.name, started)

        response.latency_ms = ctx.elapsed_ms()
        response.attempt_count = max(ctx.attempt_count, 1)
        response.fallback_used = ctx.fallback_used
        response.cache_hit = ctx.cache_hit
        response.cache_similarity = ctx.cache_similarity
        ctx.response = response
        return response

    async def run_stream(self, ctx: RequestContext) -> AsyncIterator[StreamChunk]:
        """Stream variant.

        Pre-stages still run (auth, guardrails, cache). A cache hit is replayed
        as a synthetic stream. Otherwise chunks are forwarded as they arrive
        while being accumulated so post-stages can act on the complete text.
        """
        for stage in self.pre_stages:
            started = time.perf_counter()
            try:
                result = await stage.process(ctx)
            finally:
                ctx.record_stage(stage.name, started)
            if result is not None:
                async for chunk in _replay_as_stream(result):
                    yield chunk
                return

        async for chunk in self.executor.execute_stream(ctx):
            if ctx.time_to_first_token_ms is None and chunk.content:
                ctx.time_to_first_token_ms = ctx.elapsed_ms()
            yield chunk


@runtime_checkable
class Executor(Protocol):
    """Performs the actual provider call, including retries and fallbacks."""

    async def execute(self, ctx: RequestContext) -> ChatResponse: ...

    def execute_stream(self, ctx: RequestContext) -> AsyncIterator[StreamChunk]: ...


async def _replay_as_stream(response: ChatResponse) -> AsyncIterator[StreamChunk]:
    """Render a complete response as a minimal two-chunk stream."""
    from app.core.schemas import Role

    choice = response.choices[0] if response.choices else None
    yield StreamChunk(
        id=response.id,
        model=response.model,
        created=response.created,
        role=Role.ASSISTANT,
        content=choice.message.text() if choice else "",
        provider=response.provider,
        cache_hit=response.cache_hit,
    )
    yield StreamChunk(
        id=response.id,
        model=response.model,
        created=response.created,
        finish_reason=choice.finish_reason if choice else None,
        usage=response.usage,
        provider=response.provider,
        cache_hit=response.cache_hit,
    )
