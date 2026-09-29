"""Pipeline orchestration: pre-stages, execution, post-stages, streams, and cleanups."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.errors import GuardrailViolationError
from app.core.pipeline import (
    Pipeline,
    RequestContext,
    _filter_chunk,
    _replay_as_stream,
    _StreamAccumulator,
)
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    Choice,
    FinishReason,
    Message,
    Role,
    StreamChunk,
    ToolCallDelta,
    Usage,
)


def _ctx(**fields: Any) -> RequestContext:
    request = ChatRequest(model="m", messages=[Message(role=Role.USER, content="hi")])
    return RequestContext(request=request, state=SimpleNamespace(), **fields)  # type: ignore[arg-type]


def _response(text: str = "answer") -> ChatResponse:
    return ChatResponse(
        model="m",
        choices=[Choice(message=Message(role=Role.ASSISTANT, content=text))],
    )


class Executor:
    def __init__(
        self,
        chunks: list[StreamChunk] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.chunks = chunks or []
        self.error = error

    async def execute(self, ctx: RequestContext) -> ChatResponse:
        if self.error is not None:
            raise self.error
        return _response()

    async def execute_stream(self, ctx: RequestContext) -> AsyncIterator[StreamChunk]:
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


class ShortCircuit:
    name = "short"

    def __init__(self, response: ChatResponse | None) -> None:
        self.response = response

    async def process(self, ctx: RequestContext) -> ChatResponse | None:
        return self.response


class Recorder:
    """A post-stage that also observes failures."""

    name = "recorder"

    def __init__(self, *, fail_finalize: bool = False, fail_hook: bool = False) -> None:
        self.finalized: list[ChatResponse] = []
        self.failures: list[Exception] = []
        self.fail_finalize = fail_finalize
        self.fail_hook = fail_hook

    async def finalize(self, ctx: RequestContext, response: ChatResponse) -> ChatResponse:
        if self.fail_finalize:
            raise RuntimeError("post-stage broke")
        self.finalized.append(response)
        return response

    async def on_failure(self, ctx: RequestContext, error: Exception) -> None:
        if self.fail_hook:
            raise RuntimeError("hook broke")
        self.failures.append(error)


class PlainPost:
    """A post-stage with no failure hook."""

    name = "plain"

    async def finalize(self, ctx: RequestContext, response: ChatResponse) -> ChatResponse:
        return response


class Redactor:
    """Uppercases text and holds back the last character until finish()."""

    def __init__(self) -> None:
        self.pending = ""

    def feed(self, text: str) -> str:
        combined = self.pending + text
        self.pending = combined[-1:]
        return combined[:-1].upper()

    def finish(self) -> str:
        rest, self.pending = self.pending, ""
        return rest.upper()


class Guard(Recorder):
    name = "guard"

    def __init__(self, active: bool = True) -> None:
        super().__init__()
        self.active = active
        self.recorded: list[str] = []

    def stream_redactor(self, ctx: RequestContext) -> Redactor | None:
        return Redactor() if self.active else None

    async def record(self, ctx: RequestContext, text: str) -> None:
        self.recorded.append(text)


async def _collect(stream: AsyncIterator[StreamChunk]) -> list[StreamChunk]:
    return [chunk async for chunk in stream]


# -- Unary ------------------------------------------------------------------


async def test_run_annotates_response_with_rag_and_agent_details() -> None:
    post = Recorder()
    ctx = _ctx(key_id="k")
    ctx.rag_sources = [{"id": "chunk-1"}]
    ctx.stop_reason = "max_iterations"
    ctx.tool_executions = [{"tool": "a"}, {"tool": "b"}]
    response = await Pipeline([], Executor(), [post]).run(ctx)
    assert response.sources == [{"id": "chunk-1"}]
    assert response.stop_reason == "max_iterations"
    assert response.tool_calls_executed == 2
    assert "execute" in ctx.stage_timings and "recorder" in ctx.stage_timings
    assert ctx.response is response


async def test_run_short_circuits_on_pre_stage_response() -> None:
    cached = _response("cached")
    stages = [ShortCircuit(None), ShortCircuit(cached)]
    response = await Pipeline(stages, Executor(error=RuntimeError()), []).run(_ctx())
    assert response.text == "cached"


async def test_failures_notify_observers_only_for_authenticated_requests() -> None:
    post = Recorder()
    pipeline = Pipeline([], Executor(error=ValueError("upstream")), [PlainPost(), post])
    with pytest.raises(ValueError):
        await pipeline.run(_ctx())
    assert post.failures == []
    with pytest.raises(ValueError):
        await pipeline.run(_ctx(key_id="k"))
    assert len(post.failures) == 1


async def test_failing_failure_hooks_and_cleanups_are_logged_not_raised() -> None:
    released: list[str] = []

    async def broken_cleanup() -> None:
        raise RuntimeError("cleanup broke")

    async def good_cleanup() -> None:
        released.append("slot")

    ctx = _ctx(key_id="k")
    ctx.cleanups = [broken_cleanup, good_cleanup]
    pipeline = Pipeline([], Executor(error=ValueError("x")), [Recorder(fail_hook=True)])
    with pytest.raises(ValueError):
        await pipeline.run(ctx)
    assert released == ["slot"]
    assert ctx.cleanups == []


# -- Streaming --------------------------------------------------------------


async def test_stream_replays_cache_hits() -> None:
    post = Recorder()
    cached = _response("from cache")
    chunks = await _collect(Pipeline([ShortCircuit(cached)], Executor(), [post]).run_stream(_ctx()))
    assert chunks[0].content == "from cache"
    assert chunks[1].finish_reason == FinishReason.STOP
    assert post.finalized[0] is cached


async def test_stream_without_guard_forwards_chunks_and_accumulates() -> None:
    post = Recorder()
    ctx = _ctx()
    upstream = [
        StreamChunk(id="resp-1", model="m", provider="p", deployment_id="d", content="Hel"),
        StreamChunk(model="m", content="lo"),
        StreamChunk(model="m", finish_reason=FinishReason.STOP, usage=Usage(total_tokens=3)),
    ]
    chunks = await _collect(Pipeline([], Executor(upstream), [post]).run_stream(ctx))
    assert [c.content for c in chunks] == ["Hel", "lo", None]
    [final] = post.finalized
    assert final.text == "Hello"
    assert final.id == "resp-1"
    assert final.provider == "p"
    assert final.usage.total_tokens == 3
    assert ctx.time_to_first_token_ms is not None


async def test_stream_redactor_filters_and_flushes_held_back_text() -> None:
    guard = Guard()
    upstream = [StreamChunk(model="m", content="a"), StreamChunk(model="m", content="bc")]
    chunks = await _collect(Pipeline([], Executor(upstream), [guard]).run_stream(_ctx()))
    # "a" is held back entirely (no chunk); "bc" releases "AB"; finish() flushes "C".
    assert [c.content for c in chunks] == ["AB", "C"]
    assert guard.finalized[0].text == "abc"


async def test_inactive_redactor_passes_chunks_through() -> None:
    guard = Guard(active=False)
    upstream = [StreamChunk(model="m", content="raw")]
    chunks = await _collect(Pipeline([], Executor(upstream), [guard]).run_stream(_ctx()))
    assert [c.content for c in chunks] == ["raw"]


async def test_stream_with_nothing_held_back_emits_no_trailing_chunk() -> None:
    guard = Guard()
    upstream = [StreamChunk(model="m", content="xy", finish_reason=FinishReason.STOP)]
    chunks = await _collect(Pipeline([], Executor(upstream), [guard]).run_stream(_ctx()))
    assert [c.content for c in chunks] == ["XY"]


async def test_stream_guardrail_violation_is_recorded_and_raised() -> None:
    guard = Guard()
    upstream = [StreamChunk(model="m", content="partial")]
    pipeline = Pipeline([], Executor(upstream, GuardrailViolationError("blocked")), [guard])
    ctx = _ctx(key_id="k")
    with pytest.raises(GuardrailViolationError):
        await _collect(pipeline.run_stream(ctx))
    assert guard.recorded == ["partial"]
    assert len(guard.failures) == 1


async def test_stream_violation_without_guard_still_raises() -> None:
    pipeline = Pipeline([], Executor([], GuardrailViolationError("blocked")), [])
    with pytest.raises(GuardrailViolationError):
        await _collect(pipeline.run_stream(_ctx()))


async def test_post_stage_failures_after_a_stream_are_logged() -> None:
    upstream = [StreamChunk(model="m", content="done")]
    ctx = _ctx()
    pipeline = Pipeline([], Executor(upstream), [Recorder(fail_finalize=True)])
    chunks = await _collect(pipeline.run_stream(ctx))
    assert chunks[0].content == "done"
    assert ctx.response is not None


# -- Helpers ----------------------------------------------------------------


def test_accumulator_rebuilds_streamed_tool_calls() -> None:
    accumulator = _StreamAccumulator("m")
    accumulator.add(
        StreamChunk(model="", tool_calls=[ToolCallDelta(index=1, id="call_b", name="b")])
    )
    accumulator.add(StreamChunk(model="m2", tool_calls=[ToolCallDelta(index=0, name="a")]))
    accumulator.add(StreamChunk(model="", tool_calls=[ToolCallDelta(index=1, arguments='{"x"')]))
    accumulator.add(StreamChunk(model="", tool_calls=[ToolCallDelta(index=1, arguments=":1}")]))
    response = accumulator.response()
    assert accumulator.model == "m2"
    first, second = response.choices[0].message.tool_calls
    assert (first.id, first.name, first.arguments) == ("call_0", "a", "{}")
    assert (second.id, second.name, second.arguments) == ("call_b", "b", '{"x":1}')
    assert response.choices[0].finish_reason == FinishReason.STOP


def test_accumulator_without_chunks_keeps_generated_id() -> None:
    response = _StreamAccumulator("m").response()
    assert response.id.startswith("chatcmpl-")
    assert response.text == ""


def test_filter_chunk_drops_empty_chunks_but_keeps_payloads() -> None:
    redactor = Redactor()
    assert _filter_chunk(StreamChunk(model="m", content="a"), redactor) is None
    role_only = _filter_chunk(StreamChunk(model="m", role=Role.ASSISTANT), redactor)
    assert role_only is not None and role_only.content is None
    finished = _filter_chunk(StreamChunk(model="m", finish_reason=FinishReason.STOP), redactor)
    assert finished is not None and finished.content == "A"


async def test_replay_of_response_without_choices() -> None:
    empty = ChatResponse(model="m", choices=[])
    first, second = await _collect(_replay_as_stream(empty))
    assert first.content == ""
    assert second.finish_reason is None


def test_request_context_helpers() -> None:
    ctx = _ctx()
    assert ctx.model == "m"
    ctx.record_attempt("d1")
    ctx.record_attempt("d1")
    assert not ctx.fallback_used
    ctx.record_attempt("d2")
    assert ctx.fallback_used and ctx.attempt_count == 3
