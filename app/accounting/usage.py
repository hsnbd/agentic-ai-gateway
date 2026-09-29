from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.accounting.pricing import PriceTable
from app.core.pipeline import RequestContext
from app.core.schemas import ChatResponse, FinishReason
from app.db.models import RequestLog, UsageRollup
from app.observability.logging import get_logger, redact, redact_mapping

logger = get_logger(__name__)


class UsageService:
    def __init__(
        self,
        db: Any,
        redis: Any,
        price_table: PriceTable,
        settings: Any | None = None,
    ) -> None:
        self.db = db
        self.redis = redis
        self.price_table = price_table
        self.settings = settings

    def _should_log_bodies(self, ctx: RequestContext) -> bool:
        settings = self.settings or getattr(ctx.state, "settings", None)
        return bool(getattr(settings, "log_request_bodies", False))

    def _deployment(self, ctx: RequestContext) -> Any | None:
        return ctx.routing.deployment if ctx.routing is not None else None

    def _provider(self, ctx: RequestContext, response: ChatResponse | None = None) -> str:
        deployment = self._deployment(ctx)
        return str(
            (deployment.provider if deployment is not None else None)
            or (response.provider if response is not None else None)
            or "unknown"
        )

    async def record(self, ctx: RequestContext, response: ChatResponse) -> None:
        provider = self._provider(ctx, response)
        deployment = self._deployment(ctx)
        if ctx.cache_hit:
            # A cache hit calls no provider: it is free, and what the original
            # request cost is recorded as saved instead.
            if not ctx.cost_saved_usd:
                ctx.cost_saved_usd = response.cost_usd or self.price_table.estimate_cost(
                    response.model, provider, response.usage, deployment
                )
            cost = 0.0
        else:
            cost = self.price_table.estimate_cost(
                response.model, provider, response.usage, deployment
            )
        response.cost_usd = cost
        ctx.cost_usd = cost
        failed = any(choice.finish_reason == FinishReason.ERROR for choice in response.choices)
        error = (
            (ctx.errors[-1] if ctx.errors else RuntimeError("response finish reason error"))
            if failed
            else None
        )
        await self._persist(
            ctx,
            response,
            status="error" if failed else "success",
            error=error,
            cost=cost,
        )

    async def record_error(self, ctx: RequestContext, error: Exception) -> None:
        await self._persist(ctx, None, status="error", error=error, cost=0.0)

    async def _persist(
        self,
        ctx: RequestContext,
        response: ChatResponse | None,
        *,
        status: str,
        error: Exception | None,
        cost: float,
    ) -> None:
        provider = self._provider(ctx, response)
        usage = response.usage if response is not None else None
        virtual_key = ctx.key_id or getattr(ctx.virtual_key, "id", None)
        now = datetime.now(UTC)
        bucket = now.replace(minute=0, second=0, microsecond=0)
        request_body: dict[str, Any] | None = None
        response_body: dict[str, Any] | None = None
        if self._should_log_bodies(ctx):
            request_body = redact_mapping(ctx.request.model_dump(mode="json"))
            if response is not None:
                response_body = redact_mapping(response.model_dump(mode="json"))

        route = ctx.routing
        stage_timings: dict[str, Any] = dict(ctx.stage_timings)
        attempt_details = getattr(ctx, "_attempt_details", None)
        if isinstance(attempt_details, list):
            stage_timings["attempts"] = attempt_details
        row = RequestLog(
            request_id=ctx.request_id,
            created_at=now,
            virtual_key_id=virtual_key,
            team_id=ctx.team_id,
            end_user=ctx.end_user,
            model=ctx.request.model,
            resolved_model=response.model if response is not None else None,
            provider=provider,
            deployment_id=route.deployment.id if route is not None else None,
            route=ctx.route,
            dialect=ctx.dialect,
            status=status,
            error_code=_error_code(error),
            error_message=_error_message(error),
            prompt_tokens=usage.prompt_tokens if usage is not None else 0,
            completion_tokens=usage.completion_tokens if usage is not None else 0,
            total_tokens=usage.total_tokens if usage is not None else 0,
            cached_tokens=usage.cached_tokens if usage is not None else 0,
            cost_usd=cost,
            latency_ms=response.latency_ms if response is not None else ctx.elapsed_ms(),
            time_to_first_token_ms=ctx.time_to_first_token_ms,
            stage_timings=stage_timings,
            attempt_count=max(ctx.attempt_count, 1),
            fallback_used=ctx.fallback_used,
            routing_strategy=route.strategy if route is not None else None,
            routing_reason=route.reason if route is not None else None,
            cache_hit=ctx.cache_hit,
            cache_similarity=ctx.cache_similarity,
            guardrail_flagged=ctx.guardrail_flagged,
            guardrail_results=dict(ctx.guardrail_results),
            stream=ctx.request.stream,
            tool_calls_count=len(response.tool_calls) if response is not None else 0,
            trace_id=ctx.trace_id,
            tags=list(ctx.request.tags),
            request_body=request_body,
            response_body=response_body,
        )
        values = _rollup_values(ctx, response, status, provider, virtual_key, bucket, cost)
        try:
            async with self.db.session() as session:
                session.add(row)
                if session.get_bind().dialect.name == "postgresql":
                    insert = postgres_insert(UsageRollup).values(**values)
                    update_columns = {
                        name: getattr(UsageRollup, name) + getattr(insert.excluded, name)
                        for name in (
                            "request_count",
                            "success_count",
                            "error_count",
                            "cache_hit_count",
                            "fallback_count",
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                            "cost_usd",
                            "cost_saved_usd",
                            "total_latency_ms",
                        )
                    }
                    await session.execute(
                        insert.on_conflict_do_update(
                            index_elements=[
                                UsageRollup.bucket,
                                UsageRollup.virtual_key_id,
                                UsageRollup.model,
                                UsageRollup.provider,
                            ],
                            set_=update_columns,
                        )
                    )
                else:
                    await self._portable_rollup_upsert(session, values)
        except Exception:
            logger.exception(
                "Could not persist request usage",
                request_id=ctx.request_id,
                model=ctx.request.model,
            )

    async def _portable_rollup_upsert(self, session: AsyncSession, values: dict[str, Any]) -> None:
        conditions = (
            UsageRollup.bucket == values["bucket"],
            UsageRollup.virtual_key_id == values["virtual_key_id"],
            UsageRollup.model == values["model"],
            UsageRollup.provider == values["provider"],
        )
        current = await session.scalar(select(UsageRollup).where(*conditions))
        if current is None:
            session.add(UsageRollup(**values))
            return
        for field in (
            "request_count",
            "success_count",
            "error_count",
            "cache_hit_count",
            "fallback_count",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost_usd",
            "cost_saved_usd",
            "total_latency_ms",
        ):
            setattr(current, field, getattr(current, field) + values[field])

    async def query_logs(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        virtual_key_id: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        status: str | None = None,
        cache_hit: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RequestLog]:
        statement: Select[tuple[RequestLog]] = select(RequestLog)
        if start_time is not None:
            statement = statement.where(RequestLog.created_at >= start_time)
        if end_time is not None:
            statement = statement.where(RequestLog.created_at <= end_time)
        if virtual_key_id is not None:
            statement = statement.where(RequestLog.virtual_key_id == virtual_key_id)
        if model is not None:
            statement = statement.where(RequestLog.model == model)
        if provider is not None:
            statement = statement.where(RequestLog.provider == provider)
        if status is not None:
            statement = statement.where(RequestLog.status == status)
        if cache_hit is not None:
            statement = statement.where(RequestLog.cache_hit == cache_hit)
        statement = statement.order_by(RequestLog.created_at.desc())
        rows = await self._query_rows(
            statement, max(0, min(limit, 1000)), max(0, offset), RequestLog
        )
        return [row for row in rows if isinstance(row, RequestLog)]

    async def query_usage(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        virtual_key_id: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[UsageRollup]:
        statement: Select[tuple[UsageRollup]] = select(UsageRollup)
        if start_time is not None:
            statement = statement.where(UsageRollup.bucket >= start_time)
        if end_time is not None:
            statement = statement.where(UsageRollup.bucket <= end_time)
        if virtual_key_id is not None:
            statement = statement.where(UsageRollup.virtual_key_id == virtual_key_id)
        if model is not None:
            statement = statement.where(UsageRollup.model == model)
        if provider is not None:
            statement = statement.where(UsageRollup.provider == provider)
        statement = statement.order_by(UsageRollup.bucket.desc())
        rows = await self._query_rows(
            statement, max(0, min(limit, 1000)), max(0, offset), UsageRollup
        )
        return [row for row in rows if isinstance(row, UsageRollup)]

    async def _query_rows(
        self,
        statement: Select[Any],
        limit: int,
        offset: int,
        row_type: type[RequestLog] | type[UsageRollup],
    ) -> list[Any]:
        try:
            async with self.db.session() as session:
                result = await session.scalars(statement.limit(limit).offset(offset))
                return list(result.all())
        except Exception:
            logger.exception("Could not query usage data", entity=row_type.__name__)
            return []


def _error_code(error: Exception | None) -> str | None:
    if error is None:
        return None
    code = getattr(error, "code", None)
    return str(getattr(code, "value", code) or type(error).__name__)[:64]


def _error_message(error: Exception | None) -> str | None:
    if error is None:
        return None
    return redact(str(getattr(error, "message", None) or error))


def _rollup_values(
    ctx: RequestContext,
    response: ChatResponse | None,
    status: str,
    provider: str,
    virtual_key: str | None,
    bucket: datetime,
    cost: float,
) -> dict[str, Any]:
    usage = response.usage if response is not None else None
    latency = response.latency_ms if response is not None else ctx.elapsed_ms()
    return {
        "bucket": bucket,
        "virtual_key_id": virtual_key,
        "team_id": ctx.team_id,
        "model": ctx.request.model,
        "provider": provider,
        "request_count": 1,
        "success_count": 1 if status == "success" else 0,
        "error_count": 1 if status != "success" else 0,
        "cache_hit_count": int(ctx.cache_hit),
        "fallback_count": int(ctx.fallback_used),
        "prompt_tokens": usage.prompt_tokens if usage is not None else 0,
        "completion_tokens": usage.completion_tokens if usage is not None else 0,
        "total_tokens": usage.total_tokens if usage is not None else 0,
        "cost_usd": cost,
        "cost_saved_usd": ctx.cost_saved_usd,
        "total_latency_ms": latency or 0.0,
    }
