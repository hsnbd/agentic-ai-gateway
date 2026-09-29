"""Edge cases for pricing, token counting, and usage persistence."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.accounting import pricing as pricing_module
from app.accounting import tokens as tokens_module
from app.accounting.pricing import PriceTable, estimate_cost
from app.accounting.tokens import count_message_tokens, count_tokens, estimate_request_tokens
from app.accounting.usage import UsageService, _error_code, _error_message
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
    ToolCall,
    Usage,
)
from app.db.models import Base
from app.providers.base import Deployment, Pricing

# -- PriceTable -------------------------------------------------------------


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pricing.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_empty_price_table_has_no_path() -> None:
    table = PriceTable()
    assert table.path is None
    assert table.models == {}
    assert table.get("anything") is None


def test_empty_pricing_file_loads_nothing(tmp_path: Path) -> None:
    table = PriceTable(_write(tmp_path, ""))
    assert table.models == {}
    assert table.path is not None


def test_pricing_file_must_be_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model mapping"):
        PriceTable(_write(tmp_path, "- a\n- b\n"))


def test_pricing_models_key_must_be_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'models' must be a mapping"):
        PriceTable(_write(tmp_path, "models: [1, 2]\n"))


def test_top_level_models_without_models_key(tmp_path: Path) -> None:
    table = PriceTable(_write(tmp_path, "m1: {input_per_mtok: 1.0, output_per_mtok: 2.0}\n"))
    assert table.get("m1") == Pricing(input_per_mtok=1.0, output_per_mtok=2.0)


def test_invalid_entries_are_skipped_and_overrides_load(tmp_path: Path) -> None:
    table = PriceTable(
        _write(
            tmp_path,
            """
models:
  1: {input_per_mtok: 9}
  not-a-mapping: 5
  overrides-only:
    providers:
      azure: {input_per_mtok: 3.0, output_per_mtok: 4.0}
      7: {input_per_mtok: 1.0}
      bad: nope
  legacy:
    input_per_mtok: 1.0
    provider_overrides:
      openai: {input_per_mtok: 0.5}
  empty-overrides:
    input_per_mtok: 1.0
    providers:
      bad: nope
  providers-not-mapping:
    input_per_mtok: 2.0
    providers: [1]
""",
        ),
    )
    assert "not-a-mapping" not in table.models
    assert 1 not in table.models
    assert "overrides-only" not in table.models
    assert table.get("overrides-only", "azure") == Pricing(input_per_mtok=3.0, output_per_mtok=4.0)
    assert table.get("overrides-only", "other") is None
    assert table.get("legacy", "openai") == Pricing(input_per_mtok=0.5)
    assert table.get("legacy", "other") == Pricing(input_per_mtok=1.0)
    assert "empty-overrides" not in table.provider_overrides
    assert table.get("providers-not-mapping") == Pricing(input_per_mtok=2.0)


def _deployment(**pricing: float) -> Deployment:
    return Deployment(
        id="dep",
        model_name="m",
        provider="p",
        provider_model="pm",
        pricing=Pricing(**pricing),
    )


def test_cost_falls_back_to_deployment_pricing() -> None:
    table = PriceTable()
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    deployment = _deployment(input_per_mtok=1.0, output_per_mtok=2.0)
    assert table.estimate_cost("m", "p", usage, deployment) == pytest.approx(3.0)


def test_negative_usage_values_are_clamped() -> None:
    table = PriceTable()
    deployment = _deployment(input_per_mtok=1.0)
    assert table.estimate_cost("m", "p", {"prompt_tokens": -5}, deployment) == 0.0


def test_savings_without_pricing_or_cached_rate_is_zero() -> None:
    table = PriceTable()
    usage = Usage(prompt_tokens=100, cached_tokens=50)
    assert table.estimate_savings("m", "p", usage) == 0.0
    no_cached_rate = _deployment(input_per_mtok=1.0)
    assert table.estimate_savings("m", "p", usage, no_cached_rate) == 0.0


def test_savings_uses_deployment_pricing_and_caps_cached_tokens() -> None:
    table = PriceTable()
    deployment = _deployment(input_per_mtok=3.0, cached_input_per_mtok=1.0)
    usage = {"prompt_tokens": 1_000_000, "cached_tokens": 5_000_000}
    assert table.estimate_savings("m", "p", usage, deployment) == pytest.approx(2.0)


def test_module_default_table_is_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pricing_module, "_default_price_table", None)
    usage = Usage(prompt_tokens=1_000_000)
    assert estimate_cost("gpt-4o", "openai", usage) == pytest.approx(2.5)
    first = pricing_module._default_price_table
    assert first is not None
    assert pricing_module.estimate_savings("gpt-4o", "openai", usage) == 0.0
    assert pricing_module._default_price_table is first


# -- Token counting ---------------------------------------------------------


def test_encoder_errors_fall_back_to_character_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        def encode(self, text: str, disallowed_special: Any = ()) -> list[int]:
            raise ValueError("bad")

    monkeypatch.setattr(tokens_module, "_encoder_for_model", lambda model: Broken())
    assert count_tokens("abcdefgh", "gpt-4o") == 2


def test_message_tokens_count_names_tool_ids_and_tool_calls() -> None:
    base = Message(role=Role.ASSISTANT, content="hi")
    rich = Message(
        role=Role.ASSISTANT,
        content="hi",
        name="agent",
        tool_calls=[ToolCall(id="call_1", name="lookup", arguments='{"q": "x"}')],
    )
    tool = Message(role=Role.TOOL, content="hi", tool_call_id="call_1")
    plain_total = count_message_tokens([base], "gpt-4o")
    assert count_message_tokens([rich], "gpt-4o") > plain_total
    assert count_message_tokens([tool], "gpt-4o") > plain_total


def test_tool_serialisation_falls_back_to_str() -> None:
    assert tokens_module._json_default(datetime(2026, 1, 1)) == "2026-01-01 00:00:00"
    request = ChatRequest(model="gpt-4o", messages=[Message(role=Role.USER, content="x")])
    assert estimate_request_tokens(request) == count_message_tokens(request.messages, "gpt-4o")


# -- UsageService -----------------------------------------------------------


class _SQLiteDatabase:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._sessions() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


class _BrokenDatabase:
    @asynccontextmanager
    async def session(self) -> AsyncIterator[Any]:
        raise RuntimeError("database down")
        yield  # pragma: no cover - unreachable, makes this a generator


@pytest.fixture
async def database() -> AsyncIterator[_SQLiteDatabase]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield _SQLiteDatabase(async_sessionmaker(engine, expire_on_commit=False))
    await engine.dispose()


def _context(**fields: Any) -> RequestContext:
    request = ChatRequest(
        model="gpt-4o",
        messages=[Message(role=Role.USER, content="hello")],
        tags=["t1"],
    )
    state = SimpleNamespace(settings=Settings(log_request_bodies=False), components={})
    return RequestContext(request=request, state=state, **fields)  # type: ignore[arg-type]


def _response(finish: FinishReason = FinishReason.STOP, cost: float | None = None) -> ChatResponse:
    return ChatResponse(
        model="gpt-4o",
        choices=[Choice(message=Message(role=Role.ASSISTANT, content="ok"), finish_reason=finish)],
        usage=Usage(prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000),
        provider="openai",
        latency_ms=5.0,
        cost_usd=cost,
    )


def _prices() -> PriceTable:
    return PriceTable(Path("config/pricing.yaml"))


async def test_cache_hit_is_free_and_records_savings(database: _SQLiteDatabase) -> None:
    service = UsageService(database, None, _prices())
    ctx = _context(cache_hit=True)
    response = _response()
    await service.record(ctx, response)
    assert response.cost_usd == 0.0
    assert ctx.cost_saved_usd == pytest.approx(2.5)
    [log] = await service.query_logs(cache_hit=True)
    assert log.cache_hit is True
    assert log.cost_usd == 0.0


async def test_cache_hit_reuses_original_cost_and_keeps_existing_savings(
    database: _SQLiteDatabase,
) -> None:
    service = UsageService(database, None, _prices())
    ctx = _context(cache_hit=True)
    await service.record(ctx, _response(cost=7.0))
    assert ctx.cost_saved_usd == 7.0

    preset = _context(cache_hit=True, cost_saved_usd=1.5)
    await service.record(preset, _response(cost=7.0))
    assert preset.cost_saved_usd == 1.5


async def test_error_finish_reason_is_logged_as_error(database: _SQLiteDatabase) -> None:
    service = UsageService(database, None, _prices())
    ctx = _context()
    await service.record(ctx, _response(FinishReason.ERROR))
    [log] = await service.query_logs(status="error")
    assert log.error_code == "RuntimeError"
    assert log.error_message == "response finish reason error"

    with_error = _context()
    with_error.errors.append(ProviderError(ErrorCode.PROVIDER_TIMEOUT, "slow upstream"))
    await service.record(with_error, _response(FinishReason.ERROR))
    codes = {log.error_code for log in await service.query_logs(status="error")}
    assert ErrorCode.PROVIDER_TIMEOUT.value in codes


async def test_record_error_persists_without_response(database: _SQLiteDatabase) -> None:
    settings = Settings(log_request_bodies=True)
    service = UsageService(database, None, _prices(), settings)
    ctx = _context(key_id=None)
    ctx.routing = RoutingDecision(
        deployment=Deployment(id="d1", model_name="gpt-4o", provider="openai", provider_model="x"),
        strategy="priority",
        reason="only candidate",
    )
    ctx.tool_executions.append({"tool": "lookup"})
    ctx._attempt_details = [{"deployment": "d1"}]  # type: ignore[attr-defined]
    ctx._rag_details = {"collection": "docs"}  # type: ignore[attr-defined]
    await service.record_error(ctx, ValueError("boom"))

    [log] = await service.query_logs(provider="openai")
    assert log.status == "error"
    assert log.provider == "openai"
    assert log.deployment_id == "d1"
    assert log.routing_strategy == "priority"
    assert log.request_body is not None
    assert log.response_body is None
    assert log.tool_calls_count == 1
    assert log.stage_timings["attempts"] == [{"deployment": "d1"}]
    assert log.stage_timings["rag_retrieval"] == {"collection": "docs"}
    assert log.stage_timings["tool_calls"] == [{"tool": "lookup"}]
    [rollup] = await service.query_usage(provider="openai")
    assert rollup.error_count == 1
    assert rollup.total_tokens == 0


async def test_unknown_provider_without_routing_or_response(database: _SQLiteDatabase) -> None:
    service = UsageService(database, None, _prices())
    await service.record_error(_context(), RuntimeError("x"))
    [log] = await service.query_logs(provider="unknown")
    assert log.provider == "unknown"


async def test_rollups_accumulate_on_repeat(database: _SQLiteDatabase) -> None:
    service = UsageService(database, None, _prices())
    for _ in range(3):
        await service.record(_context(fallback_used=True), _response())
    [rollup] = await service.query_usage(model="gpt-4o")
    assert rollup.request_count == 3
    assert rollup.fallback_count == 3


async def test_query_filters(database: _SQLiteDatabase) -> None:
    service = UsageService(database, None, _prices())
    ctx = _context(key_id="vk_1")
    await service.record(ctx, _response())
    now = datetime.now(UTC)
    past, future = now - timedelta(days=1), now + timedelta(days=1)

    logs = await service.query_logs(
        start_time=past,
        end_time=future,
        virtual_key_id="vk_1",
        model="gpt-4o",
        provider="openai",
        status="success",
        cache_hit=False,
        limit=5000,
        offset=-3,
    )
    assert len(logs) == 1
    assert await service.query_logs(virtual_key_id="vk_other") == []

    rollups = await service.query_usage(
        start_time=past - timedelta(hours=1),
        end_time=future,
        virtual_key_id="vk_1",
        model="gpt-4o",
        provider="openai",
    )
    assert len(rollups) == 1
    assert await service.query_usage(end_time=past) == []


async def test_persistence_and_query_failures_are_swallowed() -> None:
    service = UsageService(_BrokenDatabase(), None, _prices())
    ctx = _context()
    await service.record(ctx, _response())
    assert ctx.cost_usd == pytest.approx(2.5)
    assert await service.query_logs() == []
    assert await service.query_usage() == []


def test_error_code_and_message_helpers() -> None:
    assert _error_code(None) is None
    assert _error_message(None) is None
    plain = SimpleNamespace(code="custom_code")
    assert _error_code(plain) == "custom_code"  # type: ignore[arg-type]
    assert _error_code(RuntimeError("x")) == "RuntimeError"
    assert _error_code(ProviderError(ErrorCode.PROVIDER_ERROR, "m")) == "provider_error"
    assert _error_message(ProviderError(ErrorCode.PROVIDER_ERROR, "secret")) == "secret"
    assert _error_message(RuntimeError("plain")) == "plain"
