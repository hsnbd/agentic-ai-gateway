from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.accounting.pricing import PriceTable, estimate_cost, estimate_savings
from app.accounting.tokens import count_message_tokens, count_tokens, estimate_request_tokens
from app.accounting.usage import UsageService
from app.config.settings import Settings
from app.core.pipeline import RequestContext
from app.core.schemas import ChatRequest, ChatResponse, Choice, Message, Role, Usage
from app.db.models import Base


@pytest.fixture
def prices() -> PriceTable:
    return PriceTable(Path("config/pricing.yaml"))


def test_price_lookup_and_cached_token_cost(prices: PriceTable) -> None:
    usage = Usage(prompt_tokens=1000, completion_tokens=500, cached_tokens=400)
    cost = prices.estimate_cost("gpt-4o", "openai", usage)
    assert cost == pytest.approx((600 * 2.5 + 400 * 1.25 + 500 * 10) / 1_000_000)
    assert estimate_cost("gpt-4o", "openai", usage, price_table=prices) == cost


def test_unknown_model_cost_is_zero(prices: PriceTable) -> None:
    assert prices.estimate_cost("unlisted-model", "provider", Usage(prompt_tokens=2)) == 0.0


def test_tiktoken_fallback_for_unknown_model() -> None:
    assert count_tokens("abcdefghij", "anthropic-unknown-model") == 2


def test_message_token_estimate_grows_with_content() -> None:
    small = [Message(role=Role.USER, content="short")]
    large = [Message(role=Role.USER, content="short " * 100)]
    assert count_message_tokens(large, "gpt-4o") > count_message_tokens(small, "gpt-4o")


def test_cached_savings_estimate(prices: PriceTable) -> None:
    usage = Usage(prompt_tokens=1000, cached_tokens=400)
    expected = 400 * (2.5 - 1.25) / 1_000_000
    assert prices.estimate_savings("gpt-4o", "openai", usage) == pytest.approx(expected)
    assert estimate_savings("gpt-4o", "openai", usage, price_table=prices) == pytest.approx(
        expected
    )


def test_request_tokens_include_tool_definitions() -> None:
    plain = ChatRequest(model="gpt-4o", messages=[Message(role=Role.USER, content="hello")])
    with_tools = ChatRequest(
        model="gpt-4o",
        messages=plain.messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "a_long_weather_lookup_tool",
                    "description": "Fetch detailed weather forecast information",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
    )
    assert estimate_request_tokens(with_tools) > estimate_request_tokens(plain)


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


@pytest.mark.asyncio
async def test_usage_record_persists_log_and_sqlite_rollup() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        database = _SQLiteDatabase(async_sessionmaker(engine, expire_on_commit=False))
        settings = Settings(log_request_bodies=True)
        state = SimpleNamespace(settings=settings, components={})
        request = ChatRequest(
            model="gpt-4o",
            messages=[Message(role=Role.USER, content="credential sk-aigw-secret-value")],
        )
        context = RequestContext(request=request, state=state)
        response = ChatResponse(
            model="gpt-4o",
            choices=[Choice(message=Message(role=Role.ASSISTANT, content="answer"))],
            usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            provider="openai",
            latency_ms=25.0,
        )
        service = UsageService(database, None, PriceTable(Path("config/pricing.yaml")))

        await service.record(context, response)
        await service.record(context, response)

        logs = await service.query_logs(model="gpt-4o", provider="openai", limit=1)
        rollups = await service.query_usage(model="gpt-4o", provider="openai")
        assert len(logs) == 1
        assert logs[0].request_body is not None
        assert "sk-aigw-secret-value" not in str(logs[0].request_body)
        assert response.cost_usd == pytest.approx((100 * 2.5 + 20 * 10) / 1_000_000)
        assert len(rollups) == 1
        assert rollups[0].request_count == 2
        assert rollups[0].total_tokens == 240
    finally:
        await engine.dispose()
