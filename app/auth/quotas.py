"""Budget enforcement and spend accounting."""

from __future__ import annotations

import calendar
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import update

from app.auth.keys import ResolvedKey
from app.core.errors import BudgetExceededError
from app.db.models import Team, VirtualKey
from app.db.session import Database

logger = logging.getLogger(__name__)


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _advance_reset(reset_at: datetime, period: str, now: datetime) -> datetime:
    next_reset = _aware(reset_at)
    if period == "daily":
        while next_reset <= now:
            next_reset += timedelta(days=1)
    elif period == "weekly":
        while next_reset <= now:
            next_reset += timedelta(weeks=1)
    elif period == "monthly":
        while next_reset <= now:
            year = next_reset.year + (next_reset.month == 12)
            month = next_reset.month % 12 + 1
            day = min(next_reset.day, calendar.monthrange(year, month)[1])
            next_reset = next_reset.replace(year=year, month=month, day=day)
    else:
        raise ValueError(f"Unsupported budget period: {period}")
    return next_reset


class QuotaService:
    def __init__(self, db: Database, redis: Any) -> None:
        self.db = db
        self.redis = redis

    async def _current_spend(self, counter_key: str, db_spend: Decimal | float) -> Decimal:
        if self.redis is not None:
            try:
                value = await self.redis.get(counter_key)
                if value is not None:
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    return Decimal(str(value))
            except Exception:
                logger.warning("Spend counter unavailable; using database value", exc_info=True)
        return Decimal(str(db_spend or 0))

    async def check_budget(self, key: ResolvedKey) -> None:
        async with self.db.session() as session:
            key_row = await session.get(VirtualKey, key.id)
            key_db_spend = key_row.spend_usd if key_row is not None else key.spend_usd
            team = await session.get(Team, key.team_id) if key.team_id is not None else None
            team_budget = team.max_budget_usd if team is not None else None
            team_spend = team.spend_usd if team is not None else Decimal("0")
        key_spend = await self._current_spend(f"aigw:spend:key:{key.id}", key_db_spend)
        if key.max_budget_usd is not None and key_spend >= key.max_budget_usd:
            raise BudgetExceededError("Virtual key budget exceeded")
        if key.team_id is None:
            return
        if team is not None and team_budget is not None:
            current = await self._current_spend(f"aigw:spend:team:{key.team_id}", team_spend)
            if current >= Decimal(str(team_budget)):
                raise BudgetExceededError("Team budget exceeded")

    async def record_spend(
        self, key_id: str, team_id: str | None, amount_usd: Decimal | float
    ) -> None:
        """Charge the database, then mirror the new totals into Redis.

        The database is authoritative. Writing its post-update value to Redis
        (rather than incrementing a counter that may have been lost in a Redis
        restart and restarted from zero) keeps the fast-path check honest.
        """
        amount = Decimal(str(amount_usd))
        totals: dict[str, Decimal] = {}
        async with self.db.session() as session:
            key_total = await session.scalar(
                update(VirtualKey)
                .where(VirtualKey.id == key_id)
                .values(spend_usd=VirtualKey.spend_usd + amount)
                .returning(VirtualKey.spend_usd)
            )
            if key_total is not None:
                totals[f"aigw:spend:key:{key_id}"] = Decimal(str(key_total))
            if team_id is not None:
                team_total = await session.scalar(
                    update(Team)
                    .where(Team.id == team_id)
                    .values(spend_usd=Team.spend_usd + amount)
                    .returning(Team.spend_usd)
                )
                if team_total is not None:
                    totals[f"aigw:spend:team:{team_id}"] = Decimal(str(team_total))
        if self.redis is not None and totals:
            try:
                await self.redis.mset({name: str(value) for name, value in totals.items()})
            except Exception:
                logger.warning(
                    "Spend counter update failed; database remains authoritative",
                    exc_info=True,
                )

    async def reset_if_due(self, key: ResolvedKey) -> None:
        now = datetime.now(UTC)
        reset_keys: list[str] = []
        async with self.db.session() as session:
            row = await session.get(VirtualKey, key.id)
            if row is not None and row.budget_reset_at is not None:
                reset_at = _aware(row.budget_reset_at)
                if reset_at <= now:
                    row.spend_usd = Decimal("0")
                    row.budget_reset_at = _advance_reset(reset_at, row.budget_period, now)
                    reset_keys.append(f"aigw:spend:key:{row.id}")
            if key.team_id is not None:
                team = await session.get(Team, key.team_id)
                if team is not None and team.budget_reset_at is not None:
                    reset_at = _aware(team.budget_reset_at)
                    if reset_at <= now:
                        team.spend_usd = Decimal("0")
                        team.budget_reset_at = _advance_reset(reset_at, team.budget_period, now)
                        reset_keys.append(f"aigw:spend:team:{team.id}")
        if reset_keys and self.redis is not None:
            try:
                await self.redis.delete(*reset_keys)
            except Exception:
                logger.warning("Could not clear reset spend counters", exc_info=True)
