"""Authenticated control-plane API for gateway administration."""

from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import logging
import math
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Any, Literal

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import case, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import AdminUserOnly, CurrentAdmin, Pagination, get_gateway_state
from app.api.schemas_admin import (
    AdminUserCreateRequest,
    AdminUserPasswordRequest,
    AdminUserResponse,
    AdminUserUpdateRequest,
    CacheEntryResponse,
    CacheInvalidateRequest,
    CacheInvalidateResponse,
    CacheStatsResponse,
    ChangePasswordRequest,
    ConfigReloadResponse,
    CostBreakdownResponse,
    CostProjectionResponse,
    DashboardSummary,
    DeploymentResponse,
    GuardrailPolicyResponse,
    GuardrailViolationResponse,
    HealthCheckResponse,
    KeyCreateRequest,
    KeyUpdateRequest,
    LoginRequest,
    LoginResponse,
    LogoutResponse,
    ModelResponse,
    Page,
    PlaygroundRequest,
    PlaygroundResponse,
    ProviderStatusResponse,
    RequestAttemptResponse,
    RequestLogDetailResponse,
    RequestLogResponse,
    SystemInfoResponse,
    TeamCreateRequest,
    TeamResponse,
    TeamUpdateRequest,
    TeamUsageResponse,
    TimeSeriesPoint,
    TimeSeriesResponse,
    UsageResponse,
    UsageRow,
    VirtualKeyResponse,
)
from app.auth.console import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.auth.keys import generate_key, invalidate_cache
from app.config.settings import get_settings
from app.core.errors import ConfigurationError, NotFoundError
from app.core.pipeline import RequestContext
from app.core.schemas import ChatRequest, Message
from app.core.state import GatewayState
from app.db.models import (
    AdminUser,
    GuardrailViolation,
    RequestLog,
    Team,
    UsageRollup,
    VirtualKey,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/api", tags=["admin"])
_STARTED_AT = time.monotonic()


def _user_response(user: AdminUser) -> AdminUserResponse:
    return AdminUserResponse.model_validate(user, from_attributes=True)


def _duration(window: str) -> timedelta:
    value = window.strip().lower()
    units = {"h": 3600, "d": 86400, "w": 604800}
    try:
        amount, unit = int(value[:-1]), value[-1]
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=422, detail="window must look like 24h, 7d, or 1w") from exc
    if amount < 1 or unit not in units or amount > 366 * 7:
        raise HTTPException(status_code=422, detail="window is outside the supported range")
    return timedelta(seconds=amount * units[unit])


def _window_start(window: str) -> datetime:
    return datetime.now(UTC) - _duration(window)


def _to_float(value: Decimal | float | int | None) -> float:
    return float(value or 0)


def _order_column(model: type[Any], requested: str, allowed: set[str]) -> Any:
    if requested not in allowed:
        raise HTTPException(status_code=422, detail=f"Unsupported order_by field: {requested}")
    return getattr(model, requested)


def _rollup_totals(rows: list[UsageRollup]) -> dict[str, float | int]:
    return {
        "requests": sum(row.request_count for row in rows),
        "successes": sum(row.success_count for row in rows),
        "errors": sum(row.error_count for row in rows),
        "cache_hits": sum(row.cache_hit_count for row in rows),
        "tokens": sum(row.total_tokens for row in rows),
        "cost": sum(row.cost_usd for row in rows),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


async def _latency_percentile(
    session: AsyncSession,
    start: datetime,
    percentile: float,
    end: datetime | None = None,
) -> float:
    filters = [RequestLog.created_at >= start, RequestLog.latency_ms.is_not(None)]
    if end is not None:
        filters.append(RequestLog.created_at < end)
    count = await session.scalar(
        select(func.count()).select_from(RequestLog).where(*filters)
    )
    if not count:
        return 0.0
    offset = max(0, math.ceil(percentile * count) - 1)
    value = await session.scalar(
        select(RequestLog.latency_ms)
        .where(*filters)
        .order_by(RequestLog.latency_ms.asc())
        .offset(offset)
        .limit(1)
    )
    return float(value or 0.0)


def _log_filter_query(
    start: datetime | None,
    end: datetime | None,
    virtual_key_id: str | None,
    team_id: str | None,
    model: str | None,
    provider: str | None,
    status: str | None,
    cache_hit: bool | None,
    min_latency_ms: float | None,
    search: str | None,
) -> list[Any]:
    filters: list[Any] = []
    if start is not None:
        filters.append(RequestLog.created_at >= start)
    if end is not None:
        filters.append(RequestLog.created_at <= end)
    if virtual_key_id:
        filters.append(RequestLog.virtual_key_id == virtual_key_id)
    if team_id:
        filters.append(RequestLog.team_id == team_id)
    if model:
        filters.append(RequestLog.model == model)
    if provider:
        filters.append(RequestLog.provider == provider)
    if status:
        filters.append(RequestLog.status == status)
    if cache_hit is not None:
        filters.append(RequestLog.cache_hit.is_(cache_hit))
    if min_latency_ms is not None:
        filters.append(RequestLog.latency_ms >= min_latency_ms)
    if search:
        pattern = f"%{search}%"
        filters.append(
            or_(
                RequestLog.request_id.ilike(pattern),
                RequestLog.model.ilike(pattern),
                RequestLog.provider.ilike(pattern),
                RequestLog.error_message.ilike(pattern),
            )
        )
    return filters


def _log_response(row: RequestLog) -> RequestLogResponse:
    return RequestLogResponse(
        id=row.id,
        request_id=row.request_id,
        created_at=row.created_at,
        virtual_key_id=row.virtual_key_id,
        team_id=row.team_id,
        model=row.model,
        resolved_model=row.resolved_model,
        provider=row.provider,
        deployment_id=row.deployment_id,
        status=row.status,
        status_code=row.status_code,
        error_code=row.error_code,
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        total_tokens=row.total_tokens,
        cost_usd=row.cost_usd,
        latency_ms=row.latency_ms,
        attempt_count=row.attempt_count,
        fallback_count=_fallback_count(row),
        cache_similarity=row.cache_similarity,
        stage_timings={
            name: value for name, value in (row.stage_timings or {}).items() if name != "attempts"
        },
        routing_reason=row.routing_reason,
        cache_hit=row.cache_hit,
        guardrail_flagged=row.guardrail_flagged,
        stream=row.stream,
    )


def _stored_attempts(row: RequestLog) -> list[dict[str, Any]]:
    attempts = (row.stage_timings or {}).get("attempts", [])
    return [attempt for attempt in attempts if isinstance(attempt, dict)]


def _fallback_count(row: RequestLog) -> int:
    attempts = _stored_attempts(row)
    deployment_ids = [str(attempt.get("deployment_id", "")) for attempt in attempts]
    distinct_deployments = list(dict.fromkeys(value for value in deployment_ids if value))
    if distinct_deployments:
        return max(len(distinct_deployments) - 1, 0)
    return int(row.fallback_used)


def _log_attempts(row: RequestLog) -> list[RequestAttemptResponse]:
    return [
        RequestAttemptResponse(
            deployment_id=str(attempt.get("deployment_id", "unknown")),
            provider=(str(attempt["provider"]) if attempt.get("provider") is not None else None),
            outcome=(
                attempt.get("outcome")
                if attempt.get("outcome") in {"success", "error", "unknown"}
                else "unknown"
            ),
            latency_ms=(
                float(attempt["latency_ms"]) if attempt.get("latency_ms") is not None else None
            ),
            error=(str(attempt["error"]) if attempt.get("error") is not None else None),
        )
        for attempt in _stored_attempts(row)
    ]


@router.post("/auth/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
) -> LoginResponse:
    async with state.db.session() as session:
        user = await session.scalar(
            select(AdminUser).where(func.lower(AdminUser.email) == body.email.strip().lower())
        )
        if (
            user is None
            or not user.is_active
            or not verify_password(body.password, user.password_hash)
        ):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        user.last_login_at = datetime.now(UTC)
        await session.flush()
        user_copy = user
    token = create_access_token(user_copy.id, user_copy.email, user_copy.role)
    return LoginResponse(
        access_token=token,
        expires_in=get_settings().jwt_access_ttl_seconds,
        user=_user_response(user_copy),
    )


@router.post("/auth/logout", response_model=LogoutResponse)
async def logout(_: CurrentAdmin) -> LogoutResponse:
    """Console access tokens are stateless; clients discard the token on logout."""
    return LogoutResponse()


@router.get("/auth/me", response_model=AdminUserResponse)
async def auth_me(user: CurrentAdmin) -> AdminUserResponse:
    return _user_response(user)


@router.post("/auth/change-password", response_model=LogoutResponse)
async def change_password(
    body: ChangePasswordRequest,
    user: AdminUserOnly,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
) -> LogoutResponse:
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    async with state.db.session() as session:
        db_user = await session.get(AdminUser, user.id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="Console user not found")
        db_user.password_hash = hash_password(body.new_password)
    return LogoutResponse()


@router.get("/dashboard/summary", response_model=DashboardSummary)
async def dashboard_summary(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: CurrentAdmin,
    window: str = Query(default="24h"),
) -> DashboardSummary:
    duration = _duration(window)
    current_end = datetime.now(UTC)
    current_start = current_end - duration
    previous_start = current_start - duration

    async def aggregate_window(
        session: AsyncSession, start: datetime, end: datetime
    ) -> dict[str, float | int]:
        rollups = list(
            (
                await session.scalars(
                    select(UsageRollup).where(
                        UsageRollup.bucket >= start, UsageRollup.bucket < end
                    )
                )
            ).all()
        )
        if rollups:
            totals = _rollup_totals(rollups)
            requests = int(totals["requests"])
            successes = int(totals["successes"])
            return {
                "requests": requests,
                "success_rate": successes / requests if requests else 0.0,
                "p50": await _latency_percentile(session, start, 0.50, end),
                "p95": await _latency_percentile(session, start, 0.95, end),
                "p99": await _latency_percentile(session, start, 0.99, end),
                "cost": float(totals["cost"]),
                "tokens": int(totals["tokens"]),
                "cache_ratio": int(totals["cache_hits"]) / requests if requests else 0.0,
                "active_models": len({row.model for row in rollups if row.request_count}),
                "errors": int(totals["errors"]),
                "fallbacks": sum(row.fallback_count for row in rollups),
            }

        success_statuses = ("success", "succeeded", "ok")
        aggregate = (
            await session.execute(
                select(
                    func.count(RequestLog.id),
                    func.sum(case((RequestLog.status.in_(success_statuses), 1), else_=0)),
                    func.sum(case((RequestLog.cache_hit.is_(True), 1), else_=0)),
                    func.count(func.distinct(RequestLog.model)),
                    func.sum(RequestLog.cost_usd),
                    func.sum(RequestLog.total_tokens),
                    func.sum(case((RequestLog.fallback_used.is_(True), 1), else_=0)),
                ).where(RequestLog.created_at >= start, RequestLog.created_at < end)
            )
        ).one()
        requests = int(aggregate[0] or 0)
        successes = int(aggregate[1] or 0)
        return {
            "requests": requests,
            "success_rate": successes / requests if requests else 0.0,
            "p50": await _latency_percentile(session, start, 0.50, end),
            "p95": await _latency_percentile(session, start, 0.95, end),
            "p99": await _latency_percentile(session, start, 0.99, end),
            "cost": float(aggregate[4] or 0.0),
            "tokens": int(aggregate[5] or 0),
            "cache_ratio": int(aggregate[2] or 0) / requests if requests else 0.0,
            "active_models": int(aggregate[3] or 0),
            "errors": requests - successes,
            "fallbacks": int(aggregate[6] or 0),
        }

    async with state.db.session() as session:
        current = await aggregate_window(session, current_start, current_end)
        previous = await aggregate_window(session, previous_start, current_start)
    from app.observability.metrics import ACTIVE_REQUESTS

    active_requests = sum(
        sample.value for metric in ACTIVE_REQUESTS.collect() for sample in metric.samples
    )
    return DashboardSummary(
        requests=int(current["requests"]),
        success_rate=float(current["success_rate"]),
        p50_latency_ms=float(current["p50"]),
        p95_latency_ms=float(current["p95"]),
        p99_latency_ms=float(current["p99"]),
        total_cost_usd=float(current["cost"]),
        total_tokens=int(current["tokens"]),
        cache_hit_ratio=float(current["cache_ratio"]),
        active_models=int(current["active_models"]),
        error_count=int(current["errors"]),
        fallback_count=int(current["fallbacks"]),
        active_requests=active_requests,
        previous_requests=int(previous["requests"]),
        previous_success_rate=float(previous["success_rate"]),
        previous_p50_latency_ms=float(previous["p50"]),
        previous_p95_latency_ms=float(previous["p95"]),
        previous_p99_latency_ms=float(previous["p99"]),
        previous_total_cost_usd=float(previous["cost"]),
        previous_total_tokens=int(previous["tokens"]),
        previous_cache_hit_ratio=float(previous["cache_ratio"]),
        previous_active_models=int(previous["active_models"]),
        previous_error_count=int(previous["errors"]),
        previous_fallback_count=int(previous["fallbacks"]),
    )

@router.get("/dashboard/timeseries", response_model=TimeSeriesResponse)
async def dashboard_timeseries(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: CurrentAdmin,
    window: str = Query(default="24h"),
    interval: Literal["hour", "day"] = Query(default="hour"),
    metric: Literal["requests", "successes", "errors", "tokens", "cost", "cache_hits"] = Query(
        default="requests"
    ),
) -> TimeSeriesResponse:
    start = _window_start(window)
    async with state.db.session() as session:
        rollups = list(
            (await session.scalars(select(UsageRollup).where(UsageRollup.bucket >= start))).all()
        )
        buckets: dict[datetime, float] = defaultdict(float)
        if rollups:
            names = {
                "requests": "request_count",
                "successes": "success_count",
                "errors": "error_count",
                "tokens": "total_tokens",
                "cost": "cost_usd",
                "cache_hits": "cache_hit_count",
            }
            for rollup in rollups:
                stamp = rollup.bucket.replace(minute=0, second=0, microsecond=0)
                if interval == "day":
                    stamp = stamp.replace(hour=0)
                buckets[stamp] += float(getattr(rollup, names[metric]))
        else:
            result = await session.stream(
                select(
                    RequestLog.created_at,
                    RequestLog.status,
                    RequestLog.total_tokens,
                    RequestLog.cost_usd,
                    RequestLog.cache_hit,
                )
                .where(RequestLog.created_at >= start)
                .execution_options(yield_per=500)
            )
            async for log_row in result:
                stamp = log_row.created_at.replace(minute=0, second=0, microsecond=0)
                if interval == "day":
                    stamp = stamp.replace(hour=0)
                value = {
                    "requests": 1,
                    "successes": int(log_row.status.lower() in {"success", "succeeded", "ok"}),
                    "errors": int(log_row.status.lower() not in {"success", "succeeded", "ok"}),
                    "tokens": log_row.total_tokens,
                    "cost": log_row.cost_usd,
                    "cache_hits": int(log_row.cache_hit),
                }[metric]
                buckets[stamp] += float(value)
    points = [
        TimeSeriesPoint(timestamp=stamp, value=value) for stamp, value in sorted(buckets.items())
    ]
    return TimeSeriesResponse(metric=metric, interval=interval, points=points)


@router.get(
    "/keys",
    response_model=Page[VirtualKeyResponse],
    response_model_exclude={"items": {"__all__": {"key"}}},
)
async def list_keys(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    page: Pagination,
    _: CurrentAdmin,
) -> Page[VirtualKeyResponse]:
    async with state.db.session() as session:
        total = int(await session.scalar(select(func.count()).select_from(VirtualKey)) or 0)
        rows = list(
            (
                await session.scalars(
                    select(VirtualKey)
                    .order_by(VirtualKey.created_at.desc())
                    .offset(page.offset)
                    .limit(page.limit)
                )
            ).all()
        )
    return Page(
        items=[VirtualKeyResponse.model_validate(row, from_attributes=True) for row in rows],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.post("/keys", response_model=VirtualKeyResponse, status_code=201)
async def create_key(
    body: KeyCreateRequest,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: AdminUserOnly,
) -> VirtualKeyResponse:
    raw_key, digest, prefix = generate_key()
    async with state.db.session() as session:
        if body.team_id and await session.get(Team, body.team_id) is None:
            raise HTTPException(status_code=404, detail="Team not found")
        key = VirtualKey(
            key_hash=digest,
            key_prefix=prefix,
            name=body.name,
            team_id=body.team_id,
            max_budget_usd=body.max_budget_usd,
            budget_period=body.budget_duration,
            rpm_limit=body.rpm_limit,
            tpm_limit=body.tpm_limit,
            max_parallel_requests=body.max_parallel_requests,
            allowed_models=body.allowed_models,
            blocked_models=body.blocked_models,
            guardrail_policy=body.guardrail_policy,
            allowed_routes=body.allowed_routes,
            is_active=body.enabled,
            expires_at=body.expires_at,
            metadata_=body.metadata,
        )
        session.add(key)
        try:
            await session.flush()
        except IntegrityError as exc:
            raise HTTPException(
                status_code=409, detail="Virtual key conflicts with an existing record"
            ) from exc
    return VirtualKeyResponse.model_validate(key, from_attributes=True).model_copy(
        update={"key": raw_key}
    )


@router.get("/keys/{key_id}", response_model=VirtualKeyResponse, response_model_exclude={"key"})
async def get_key(
    key_id: str,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: CurrentAdmin,
) -> VirtualKeyResponse:
    async with state.db.session() as session:
        key = await session.get(VirtualKey, key_id)
        if key is None:
            raise HTTPException(status_code=404, detail="Virtual key not found")
    return VirtualKeyResponse.model_validate(key, from_attributes=True)


@router.patch("/keys/{key_id}", response_model=VirtualKeyResponse, response_model_exclude={"key"})
async def update_key(
    key_id: str,
    body: KeyUpdateRequest,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: AdminUserOnly,
) -> VirtualKeyResponse:
    async with state.db.session() as session:
        key = await session.get(VirtualKey, key_id)
        if key is None:
            raise HTTPException(status_code=404, detail="Virtual key not found")
        values = body.model_dump(exclude_unset=True)
        if (
            "team_id" in values
            and values["team_id"]
            and await session.get(Team, values["team_id"]) is None
        ):
            raise HTTPException(status_code=404, detail="Team not found")
        remap = {
            "budget_duration": "budget_period",
            "enabled": "is_active",
            "metadata": "metadata_",
        }
        for name, value in values.items():
            setattr(key, remap.get(name, name), value)
        await session.flush()
        await invalidate_cache(state.redis, key.key_hash)
    return VirtualKeyResponse.model_validate(key, from_attributes=True)


@router.delete("/keys/{key_id}", response_model=LogoutResponse)
async def delete_key(
    key_id: str,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: AdminUserOnly,
) -> LogoutResponse:
    async with state.db.session() as session:
        key = await session.get(VirtualKey, key_id)
        if key is None:
            raise HTTPException(status_code=404, detail="Virtual key not found")
        await invalidate_cache(state.redis, key.key_hash)
        await session.delete(key)
    return LogoutResponse()


@router.post("/keys/{key_id}/regenerate", response_model=VirtualKeyResponse)
async def regenerate_key(
    key_id: str,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: AdminUserOnly,
) -> VirtualKeyResponse:
    raw_key, digest, prefix = generate_key()
    async with state.db.session() as session:
        key = await session.get(VirtualKey, key_id)
        if key is None:
            raise HTTPException(status_code=404, detail="Virtual key not found")
        await invalidate_cache(state.redis, key.key_hash)
        key.key_hash = digest
        key.key_prefix = prefix
        await session.flush()
    return VirtualKeyResponse.model_validate(key, from_attributes=True).model_copy(
        update={"key": raw_key}
    )


@router.get("/teams", response_model=Page[TeamResponse])
async def list_teams(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    page: Pagination,
    _: CurrentAdmin,
) -> Page[TeamResponse]:
    async with state.db.session() as session:
        total = int(await session.scalar(select(func.count()).select_from(Team)) or 0)
        rows = list(
            (
                await session.scalars(
                    select(Team).order_by(Team.name).offset(page.offset).limit(page.limit)
                )
            ).all()
        )
    return Page(
        items=[TeamResponse.model_validate(row, from_attributes=True) for row in rows],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.post("/teams", response_model=TeamResponse, status_code=201)
async def create_team(
    body: TeamCreateRequest,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: AdminUserOnly,
) -> TeamResponse:
    async with state.db.session() as session:
        team = Team(
            name=body.name,
            description=body.description,
            max_budget_usd=body.max_budget_usd,
            budget_period=body.budget_period,
            metadata_=body.metadata,
        )
        session.add(team)
        try:
            await session.flush()
        except IntegrityError as exc:
            raise HTTPException(status_code=409, detail="Team name already exists") from exc
    return TeamResponse.model_validate(team, from_attributes=True)


@router.get("/teams/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: str,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: CurrentAdmin,
) -> TeamResponse:
    async with state.db.session() as session:
        team = await session.get(Team, team_id)
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")
    return TeamResponse.model_validate(team, from_attributes=True)


@router.patch("/teams/{team_id}", response_model=TeamResponse)
async def update_team(
    team_id: str,
    body: TeamUpdateRequest,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: AdminUserOnly,
) -> TeamResponse:
    async with state.db.session() as session:
        team = await session.get(Team, team_id)
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")
        for name, value in body.model_dump(exclude_unset=True).items():
            setattr(team, "metadata_" if name == "metadata" else name, value)
        try:
            await session.flush()
        except IntegrityError as exc:
            raise HTTPException(status_code=409, detail="Team name already exists") from exc
    return TeamResponse.model_validate(team, from_attributes=True)


@router.delete("/teams/{team_id}", response_model=LogoutResponse)
async def delete_team(
    team_id: str,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: AdminUserOnly,
) -> LogoutResponse:
    async with state.db.session() as session:
        team = await session.get(Team, team_id)
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")
        await session.delete(team)
    return LogoutResponse()


@router.get("/teams/{team_id}/usage", response_model=TeamUsageResponse)
async def team_usage(
    team_id: str,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: CurrentAdmin,
    window: str = Query(default="30d"),
) -> TeamUsageResponse:
    start = _window_start(window)
    async with state.db.session() as session:
        if await session.get(Team, team_id) is None:
            raise HTTPException(status_code=404, detail="Team not found")
        rollups = list(
            (
                await session.scalars(
                    select(UsageRollup).where(
                        UsageRollup.team_id == team_id, UsageRollup.bucket >= start
                    )
                )
            ).all()
        )
        if rollups:
            totals = _rollup_totals(rollups)
            requests, successes, errors, tokens, cost = (
                int(totals["requests"]),
                int(totals["successes"]),
                int(totals["errors"]),
                int(totals["tokens"]),
                float(totals["cost"]),
            )
        else:
            successful = case((RequestLog.status.in_(("success", "succeeded", "ok")), 1), else_=0)
            aggregate = (
                await session.execute(
                    select(
                        func.count(RequestLog.id),
                        func.sum(successful),
                        func.sum(RequestLog.total_tokens),
                        func.sum(RequestLog.cost_usd),
                    ).where(RequestLog.team_id == team_id, RequestLog.created_at >= start)
                )
            ).one()
            requests = int(aggregate[0] or 0)
            successes = int(aggregate[1] or 0)
            errors = requests - successes
            tokens = int(aggregate[2] or 0)
            cost = float(aggregate[3] or 0.0)
    return TeamUsageResponse(
        team_id=team_id,
        requests=requests,
        success_count=successes,
        error_count=errors,
        total_tokens=tokens,
        cost_usd=cost,
    )


async def _active_admin_ids_for_update(session: AsyncSession) -> list[str]:
    """Lock active admin rows while enforcing the last-admin invariant."""
    return list(
        (
            await session.scalars(
                select(AdminUser.id)
                .where(AdminUser.role == "admin", AdminUser.is_active.is_(True))
                .with_for_update()
            )
        ).all()
    )


@router.get("/users", response_model=Page[AdminUserResponse])
async def list_admin_users(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    page: Pagination,
    _: AdminUserOnly,
) -> Page[AdminUserResponse]:
    async with state.db.session() as session:
        total = int(await session.scalar(select(func.count()).select_from(AdminUser)) or 0)
        rows = list(
            (
                await session.scalars(
                    select(AdminUser)
                    .order_by(AdminUser.created_at.desc(), AdminUser.id)
                    .offset(page.offset)
                    .limit(page.limit)
                )
            ).all()
        )
    return Page(
        items=[_user_response(user) for user in rows],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.post("/users", response_model=AdminUserResponse, status_code=201)
async def create_admin_user(
    body: AdminUserCreateRequest,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: AdminUserOnly,
) -> AdminUserResponse:
    async with state.db.session() as session:
        user = AdminUser(
            email=body.email.strip().lower(),
            password_hash=hash_password(body.password),
            role=body.role,
            is_active=True,
        )
        session.add(user)
        try:
            await session.flush()
        except IntegrityError as exc:
            raise HTTPException(
                status_code=409, detail="A console user with that email already exists"
            ) from exc
    return _user_response(user)


@router.patch("/users/{user_id}", response_model=AdminUserResponse)
async def update_admin_user(
    user_id: str,
    body: AdminUserUpdateRequest,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    actor: AdminUserOnly,
) -> AdminUserResponse:
    changes = body.model_dump(exclude_unset=True)
    if not changes or any(value is None for value in changes.values()):
        raise HTTPException(status_code=422, detail="Provide a non-null role or is_active update")
    async with state.db.session() as session:
        user = await session.get(AdminUser, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="Console user not found")
        losing_admin = (
            user.role == "admin"
            and user.is_active
            and (
                changes.get("role", user.role) != "admin"
                or changes.get("is_active", user.is_active) is False
            )
        )
        if user.id == actor.id and (
            changes.get("role", user.role) != "admin"
            or changes.get("is_active", user.is_active) is False
        ):
            raise HTTPException(
                status_code=409, detail="You cannot demote or deactivate your own admin account"
            )
        if losing_admin:
            active_admins = await _active_admin_ids_for_update(session)
            if len(active_admins) <= 1:
                raise HTTPException(
                    status_code=409, detail="The last active admin cannot be demoted or deactivated"
                )
        for field, value in changes.items():
            setattr(user, field, value)
        await session.flush()
    return _user_response(user)


@router.delete("/users/{user_id}", response_model=LogoutResponse)
async def delete_admin_user(
    user_id: str,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    actor: AdminUserOnly,
) -> LogoutResponse:
    async with state.db.session() as session:
        user = await session.get(AdminUser, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="Console user not found")
        if user.id == actor.id:
            raise HTTPException(
                status_code=409, detail="You cannot delete your own console account"
            )
        if user.role == "admin" and user.is_active:
            active_admins = await _active_admin_ids_for_update(session)
            if len(active_admins) <= 1:
                raise HTTPException(
                    status_code=409, detail="The last active admin cannot be deleted"
                )
        await session.delete(user)
    return LogoutResponse()


@router.post("/users/me/change-password", response_model=LogoutResponse)
async def change_own_password(
    body: AdminUserPasswordRequest,
    actor: CurrentAdmin,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
) -> LogoutResponse:
    if not verify_password(body.current_password, actor.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    async with state.db.session() as session:
        user = await session.get(AdminUser, actor.id)
        if user is None or not user.is_active:
            raise HTTPException(status_code=401, detail="Console user is no longer active")
        user.password_hash = hash_password(body.new_password)
    return LogoutResponse()


@router.get("/providers/status", response_model=Page[ProviderStatusResponse])
async def provider_status(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    page: Pagination,
    _: AdminUserOnly,
) -> Page[ProviderStatusResponse]:
    deployments = state.registry.list_deployments()
    by_provider: dict[str, list[Any]] = defaultdict(list)
    for deployment in deployments:
        by_provider[deployment.provider].append(deployment)

    results: list[ProviderStatusResponse] = []
    for provider_name in sorted(by_provider):
        provider_deployments = by_provider[provider_name]
        configured = (
            True
            if provider_name == "ollama"
            else any(deployment.api_key for deployment in provider_deployments)
        )
        breaker_snapshot = state.breaker.snapshot() if state.breaker is not None else {}
        health_states = [
            breaker_snapshot[deployment.id].state.value
            for deployment in provider_deployments
            if deployment.id in breaker_snapshot
        ]
        if "closed" in health_states or not health_states:
            health_state = "closed"
        elif "half_open" in health_states:
            health_state = "half_open"
        else:
            health_state = "open"

        reachable = False
        for deployment in provider_deployments:
            if not deployment.enabled:
                continue
            try:
                provider = state.registry.provider_for(deployment)
                reachable = await provider.health_check(deployment)
                break
            except Exception:
                logger.warning("Provider health check failed for %s", provider_name, exc_info=True)
                reachable = False
                break
        results.append(
            ProviderStatusResponse(
                provider=provider_name,
                configured=configured,
                reachable=reachable,
                health_state=health_state,
            )
        )
    return Page(
        items=results[page.offset : page.offset + page.limit],
        total=len(results),
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/deployments", response_model=Page[DeploymentResponse])
async def list_deployments(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    page: Pagination,
    _: CurrentAdmin,
) -> Page[DeploymentResponse]:
    deployments = state.registry.list_deployments()
    deployments.sort(key=lambda item: item.id)
    health_map = state.breaker.snapshot() if state.breaker is not None else {}
    items: list[DeploymentResponse] = []
    for deployment in deployments[page.offset : page.offset + page.limit]:
        health = health_map.get(deployment.id)
        items.append(
            DeploymentResponse(
                id=deployment.id,
                model=deployment.model_name,
                provider=deployment.provider,
                provider_model=deployment.provider_model,
                enabled=deployment.enabled,
                capabilities=deployment.capabilities.model_dump(),
                pricing=deployment.pricing.model_dump(),
                health_state=str(health.state.value if health else "closed"),
                consecutive_failures=health.consecutive_failures if health else 0,
                failure_rate=health.failure_rate if health else 0.0,
                ewma_latency_ms=health.ewma_latency_ms if health else 0.0,
                priority=deployment.priority,
                weight=deployment.weight,
                tags=deployment.tags,
            )
        )
    return Page(items=items, total=len(deployments), limit=page.limit, offset=page.offset)


@router.get("/models", response_model=Page[ModelResponse])
async def list_models(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    page: Pagination,
    _: CurrentAdmin,
) -> Page[ModelResponse]:
    model_names = state.registry.list_models()
    names_and_deployments: list[tuple[str, list[Any]]] = []
    for name in model_names:
        try:
            deployments = state.registry.deployments_for(name, include_disabled=True)
        except NotFoundError:
            deployments = []
        names_and_deployments.append((name, deployments))
    items: list[ModelResponse] = []
    for name, deployments in sorted(names_and_deployments, key=lambda item: item[0])[
        page.offset : page.offset + page.limit
    ]:
        if not deployments:
            continue
        capabilities: dict[str, Any] = {}
        for field, value in deployments[0].capabilities.model_dump().items():
            if isinstance(value, bool):
                capabilities[field] = any(getattr(item.capabilities, field) for item in deployments)
            else:
                capabilities[field] = value
        pricing = deployments[0].pricing.model_dump()
        items.append(
            ModelResponse(
                name=name,
                capabilities=capabilities,
                pricing=pricing,
                deployments=[item.id for item in deployments],
            )
        )
    total = len(model_names)
    return Page(items=items, total=total, limit=page.limit, offset=page.offset)


@router.post("/deployments/{deployment_id}/health-check", response_model=HealthCheckResponse)
async def deployment_health_check(
    deployment_id: str,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: AdminUserOnly,
) -> HealthCheckResponse:
    try:
        deployment = state.registry.get_deployment(deployment_id)
        healthy = await state.registry.provider_for(deployment).health_check(deployment)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from exc
    return HealthCheckResponse(deployment_id=deployment_id, healthy=healthy)


@router.post(
    "/config/reload",
    response_model=ConfigReloadResponse,
)
async def reload_config(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: AdminUserOnly,
) -> ConfigReloadResponse:
    try:
        state.registry.load_config(state.settings.models_config_path)
    except (
        ConfigurationError,
        yaml.YAMLError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:
        logger.warning("Admin config reload rejected: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Model configuration is invalid: {exc}",
        ) from exc
    return ConfigReloadResponse(
        models=state.registry.list_models(), deployment_count=len(state.registry.list_deployments())
    )


@router.get("/logs", response_model=Page[RequestLogResponse])
async def list_logs(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    page: Pagination,
    _: CurrentAdmin,
    start: datetime | None = None,
    end: datetime | None = None,
    virtual_key_id: str | None = None,
    team_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    status: str | None = None,
    cache_hit: bool | None = None,
    min_latency_ms: float | None = Query(default=None, ge=0),
    search: str | None = Query(default=None, max_length=255),
) -> Page[RequestLogResponse]:
    filters = _log_filter_query(
        start,
        end,
        virtual_key_id,
        team_id,
        model,
        provider,
        status,
        cache_hit,
        min_latency_ms,
        search,
    )
    async with state.db.session() as session:
        total = int(
            await session.scalar(select(func.count()).select_from(RequestLog).where(*filters)) or 0
        )
        column = _order_column(
            RequestLog,
            page.order_by,
            {"created_at", "latency_ms", "cost_usd", "model", "status", "provider"},
        )
        ordering = column.desc() if page.order_dir == "desc" else column.asc()
        rows = list(
            (
                await session.scalars(
                    select(RequestLog)
                    .where(*filters)
                    .order_by(ordering)
                    .offset(page.offset)
                    .limit(page.limit)
                )
            ).all()
        )
    return Page(
        items=[_log_response(row) for row in rows],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/logs/export", response_model=None, responses={200: {"content": {"text/csv": {}}}})
async def export_logs(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: CurrentAdmin,
    start: datetime | None = None,
    end: datetime | None = None,
    virtual_key_id: str | None = None,
    team_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    status: str | None = None,
    cache_hit: bool | None = None,
    min_latency_ms: float | None = Query(default=None, ge=0),
    search: str | None = Query(default=None, max_length=255),
) -> StreamingResponse:
    filters = _log_filter_query(
        start,
        end,
        virtual_key_id,
        team_id,
        model,
        provider,
        status,
        cache_hit,
        min_latency_ms,
        search,
    )

    async def csv_rows() -> AsyncIterator[str]:
        columns = [
            "request_id",
            "created_at",
            "model",
            "provider",
            "status",
            "status_code",
            "virtual_key_id",
            "team_id",
            "total_tokens",
            "cost_usd",
            "latency_ms",
            "cache_hit",
        ]
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(columns)
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)
        async with state.db.session() as session:
            statement = (
                select(RequestLog).where(*filters).order_by(RequestLog.created_at, RequestLog.id)
            )
            result = await session.stream_scalars(statement.execution_options(yield_per=500))
            async for row in result:
                writer.writerow([getattr(row, name) for name in columns])
                yield output.getvalue()
                output.seek(0)
                output.truncate(0)

    return StreamingResponse(
        csv_rows(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=request-logs.csv"},
    )


@router.get("/logs/{request_id}", response_model=RequestLogDetailResponse)
async def get_log_detail(
    request_id: str,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    current_admin: CurrentAdmin,
    reveal: bool = Query(default=False),
) -> RequestLogDetailResponse:
    async with state.db.session() as session:
        row = await session.scalar(select(RequestLog).where(RequestLog.request_id == request_id))
        if row is None:
            raise HTTPException(status_code=404, detail="Request log not found")
    if reveal and current_admin.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Administrator role required to reveal log bodies",
        )

    body_redacted = bool(
        (row.request_body is not None or row.response_body is not None) and not reveal
    )
    if reveal and (row.request_body is not None or row.response_body is not None):
        from app.observability.logging import get_logger

        get_logger(__name__).warning(
            "admin_log_body_reveal",
            actor_id=current_admin.id,
            actor_email=current_admin.email,
            log_id=row.id,
            timestamp=datetime.now(UTC).isoformat(),
        )

    return RequestLogDetailResponse(
        **_log_response(row).model_dump(),
        routing_strategy=row.routing_strategy,
        guardrail_results=row.guardrail_results or {},
        request_body=row.request_body if reveal else None,
        response_body=row.response_body if reveal else None,
        body_redacted=body_redacted,
        attempts=_log_attempts(row),
    )


@router.get("/usage", response_model=UsageResponse)
async def usage(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    page: Pagination,
    _: CurrentAdmin,
    group_by: Literal["model", "provider", "key", "team", "day", "hour"] = Query(default="model"),
    window: str = Query(default="30d"),
) -> UsageResponse:
    start = _window_start(window)
    dimension = {
        "model": "model",
        "provider": "provider",
        "key": "virtual_key_id",
        "team": "team_id",
    }.get(group_by)
    async with state.db.session() as session:
        if group_by == "day":
            group_expression = func.date(UsageRollup.bucket)
            log_group = func.date(RequestLog.created_at)
        elif group_by == "hour":
            if session.get_bind().dialect.name == "sqlite":
                group_expression = func.strftime("%Y-%m-%d %H:00:00", UsageRollup.bucket)
                log_group = func.strftime("%Y-%m-%d %H:00:00", RequestLog.created_at)
            else:
                group_expression = func.date_trunc("hour", UsageRollup.bucket)
                log_group = func.date_trunc("hour", RequestLog.created_at)
        else:
            assert dimension is not None
            rollup_dimension = getattr(UsageRollup, dimension)
            group_expression = func.coalesce(rollup_dimension, "unassigned")
            log_dimension = getattr(RequestLog, dimension)
            log_group = func.coalesce(log_dimension, "unassigned")
        rollup_rows = (
            await session.execute(
                select(
                    group_expression,
                    func.sum(UsageRollup.request_count),
                    func.sum(UsageRollup.success_count),
                    func.sum(UsageRollup.error_count),
                    func.sum(UsageRollup.cache_hit_count),
                    func.sum(UsageRollup.total_tokens),
                    func.sum(UsageRollup.cost_usd),
                )
                .where(UsageRollup.bucket >= start)
                .group_by(group_expression)
            )
        ).all()
        if rollup_rows:
            rows = [
                UsageRow(
                    group=str(row[0]),
                    requests=int(row[1] or 0),
                    success_count=int(row[2] or 0),
                    error_count=int(row[3] or 0),
                    cache_hit_count=int(row[4] or 0),
                    total_tokens=int(row[5] or 0),
                    cost_usd=float(row[6] or 0.0),
                )
                for row in rollup_rows
            ]
        else:
            successful = case((RequestLog.status.in_(("success", "succeeded", "ok")), 1), else_=0)
            raw_rows = (
                await session.execute(
                    select(
                        log_group,
                        func.count(RequestLog.id),
                        func.sum(successful),
                        func.sum(case((RequestLog.cache_hit.is_(True), 1), else_=0)),
                        func.sum(RequestLog.total_tokens),
                        func.sum(RequestLog.cost_usd),
                    )
                    .where(RequestLog.created_at >= start)
                    .group_by(log_group)
                )
            ).all()
            rows = [
                UsageRow(
                    group=str(row[0]),
                    requests=int(row[1] or 0),
                    success_count=int(row[2] or 0),
                    error_count=int(row[1] or 0) - int(row[2] or 0),
                    cache_hit_count=int(row[3] or 0),
                    total_tokens=int(row[4] or 0),
                    cost_usd=float(row[5] or 0.0),
                )
                for row in raw_rows
            ]
    rows.sort(key=lambda row: row.group)
    return UsageResponse(
        group_by=group_by,
        rows=rows[page.offset : page.offset + page.limit],
        total=len(rows),
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/usage/costs", response_model=CostBreakdownResponse)
async def usage_costs(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: CurrentAdmin,
) -> CostBreakdownResponse:
    now = datetime.now(UTC)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if now.month == 12:
        end = now.replace(
            year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0
        )
    else:
        end = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
    async with state.db.session() as session:
        rollup_rows = (
            await session.execute(
                select(
                    UsageRollup.model,
                    func.sum(UsageRollup.request_count),
                    func.sum(UsageRollup.success_count),
                    func.sum(UsageRollup.error_count),
                    func.sum(UsageRollup.cache_hit_count),
                    func.sum(UsageRollup.total_tokens),
                    func.sum(UsageRollup.cost_usd),
                )
                .where(UsageRollup.bucket >= start)
                .group_by(UsageRollup.model)
            )
        ).all()
        if rollup_rows:
            by_model = [
                UsageRow(
                    group=str(row[0]),
                    requests=int(row[1] or 0),
                    success_count=int(row[2] or 0),
                    error_count=int(row[3] or 0),
                    cache_hit_count=int(row[4] or 0),
                    total_tokens=int(row[5] or 0),
                    cost_usd=float(row[6] or 0.0),
                )
                for row in rollup_rows
            ]
        else:
            successful = case((RequestLog.status.in_(("success", "succeeded", "ok")), 1), else_=0)
            raw_rows = (
                await session.execute(
                    select(
                        RequestLog.model,
                        func.count(RequestLog.id),
                        func.sum(successful),
                        func.sum(case((RequestLog.cache_hit.is_(True), 1), else_=0)),
                        func.sum(RequestLog.total_tokens),
                        func.sum(RequestLog.cost_usd),
                    )
                    .where(RequestLog.created_at >= start)
                    .group_by(RequestLog.model)
                )
            ).all()
            by_model = [
                UsageRow(
                    group=str(row[0]),
                    requests=int(row[1] or 0),
                    success_count=int(row[2] or 0),
                    error_count=int(row[1] or 0) - int(row[2] or 0),
                    cache_hit_count=int(row[3] or 0),
                    total_tokens=int(row[4] or 0),
                    cost_usd=float(row[5] or 0.0),
                )
                for row in raw_rows
            ]
        by_model.sort(key=lambda row: row.group)
        spent = sum(row.cost_usd for row in by_model)
        elapsed = max((now - start).total_seconds(), 1)
        period = max((end - start).total_seconds(), 1)
    return CostBreakdownResponse(
        total_cost_usd=spent,
        by_model=by_model,
        projection=CostProjectionResponse(
            period_start=start,
            period_end=end,
            spent_usd=spent,
            projected_total_usd=spent * period / elapsed,
            budget_usd=None,
        ),
    )


@router.get("/guardrails/violations", response_model=Page[GuardrailViolationResponse])
async def guardrail_violations(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    page: Pagination,
    _: CurrentAdmin,
) -> Page[GuardrailViolationResponse]:
    async with state.db.session() as session:
        total = int(await session.scalar(select(func.count()).select_from(GuardrailViolation)) or 0)
        rows = list(
            (
                await session.scalars(
                    select(GuardrailViolation)
                    .order_by(GuardrailViolation.created_at.desc())
                    .offset(page.offset)
                    .limit(page.limit)
                )
            ).all()
        )
    items = [GuardrailViolationResponse.model_validate(row, from_attributes=True) for row in rows]
    return Page(items=items, total=total, limit=page.limit, offset=page.offset)


@router.get("/guardrails/policies", response_model=Page[GuardrailPolicyResponse])
async def guardrail_policies(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    page: Pagination,
    _: CurrentAdmin,
) -> Page[GuardrailPolicyResponse]:
    config = state.components.get("guardrails")
    if config is None:
        try:
            from app.guardrails.registry import GuardrailRegistry

            config = GuardrailRegistry.load(state.settings.guardrails_config_path)
        except ImportError:
            logger.info("Guardrail subsystem is unavailable")
    policies = getattr(config, "_policies", None) if config is not None else None
    if policies is None:
        policies = getattr(config, "policies", {}) if config is not None else {}
    if isinstance(policies, dict):
        items = []
        for name, policy in sorted(policies.items()):
            rules = getattr(policy, "rules", None)
            if rules is None:
                rules = [
                    *getattr(policy, "input_rules", []),
                    *getattr(policy, "output_rules", []),
                ]
            items.append(
                GuardrailPolicyResponse(
                    name=str(name),
                    enabled=bool(getattr(policy, "enabled", True)),
                    rules=[str(getattr(rule, "name", rule)) for rule in rules],
                )
            )
    else:
        items = []
    return Page(
        items=items[page.offset : page.offset + page.limit],
        total=len(items),
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: CurrentAdmin,
) -> CacheStatsResponse:
    enabled = bool(state.settings.cache_enabled)
    cache = state.components.get("cache")
    stats_method: Callable[[], Any] | None = getattr(cache, "stats", None)
    values: dict[str, Any] = {}
    if callable(stats_method):
        result = stats_method()
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, dict):
            values = result
    available = bool(state.redis is not None and getattr(cache, "available", True))
    index_size_bytes: int | None = None
    if available and state.redis is not None:
        try:
            raw_info = await state.redis.execute_command(
                "FT.INFO", state.settings.cache_index_name
            )
            info: dict[str, Any] = {}
            if isinstance(raw_info, (list, tuple)):
                for index in range(0, len(raw_info) - 1, 2):
                    key = raw_info[index]
                    value = raw_info[index + 1]
                    key_text = key.decode() if isinstance(key, bytes) else str(key)
                    info[key_text.lstrip(":")] = value
            size_mb = info.get("vector_index_sz_mb")
            if size_mb is not None:
                index_size_bytes = int(float(size_mb) * 1024 * 1024)
        except Exception:
            logger.debug("Redis Search index size is unavailable", exc_info=True)
    async with state.db.session() as session:
        saved_cost = await session.scalar(select(func.sum(UsageRollup.cost_saved_usd)))
    return CacheStatsResponse(
        enabled=enabled,
        available=available,
        hits=(values.get("hits") if available else None),
        misses=(values.get("misses") if available else None),
        entries=(values.get("entries") if available else None),
        similarity_threshold=state.settings.cache_similarity_threshold,
        index_size_bytes=index_size_bytes,
        estimated_cost_saved_usd=float(saved_cost) if saved_cost is not None else 0.0,
        estimated_latency_saved_ms=None,
    )


@router.get("/cache/entries", response_model=Page[CacheEntryResponse])
async def cache_entries(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    page: Pagination,
    _: CurrentAdmin,
) -> Page[CacheEntryResponse]:
    if state.redis is None:
        raise HTTPException(status_code=503, detail="Redis cache is unavailable")

    selected: list[str] = []
    total = 0
    async for raw_key in state.redis.scan_iter(match="aigw:cache:entry:*"):
        key = raw_key.decode() if isinstance(raw_key, bytes) else str(raw_key)
        if page.offset <= total < page.offset + page.limit:
            selected.append(key)
        total += 1

    def decode(value: Any) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    entries: list[CacheEntryResponse] = []
    now = time.time()
    for key in selected:
        raw_fields = await state.redis.hgetall(key)
        fields = {
            decode(name): decode(value)
            for name, value in raw_fields.items()
            if decode(name) != "embedding"
        }
        created_at: float | None = None
        with contextlib.suppress(KeyError, ValueError):
            created_at = float(fields["created_at"])
        response_model: str | None = None
        if fields.get("response"):
            try:
                import json

                cached_response = json.loads(fields["response"])
                if isinstance(cached_response, dict) and cached_response.get("model") is not None:
                    response_model = str(cached_response["model"])
            except (ValueError, TypeError):
                logger.warning("Cache entry has an invalid response payload: %s", key)
        ttl_value = await state.redis.ttl(key)
        prompt = fields.get("prompt") or fields.get("cached_prompt")
        if prompt is not None:
            from app.observability.logging import redact

            prompt = redact(prompt)
        entries.append(
            CacheEntryResponse(
                key=key,
                model=response_model,
                namespace=fields.get("namespace"),
                hit_count=(
                    int(fields["hit_count"])
                    if fields.get("hit_count", "").isdigit()
                    else None
                ),
                age_seconds=max(now - created_at, 0.0) if created_at is not None else None,
                ttl_remaining_seconds=int(ttl_value) if ttl_value >= 0 else None,
                cached_prompt=prompt,
            )
        )
    return Page(items=entries, total=total, limit=page.limit, offset=page.offset)


@router.post("/cache/invalidate", response_model=CacheInvalidateResponse)
async def cache_invalidate(
    body: CacheInvalidateRequest,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: AdminUserOnly,
) -> CacheInvalidateResponse:
    if state.redis is None:
        raise HTTPException(status_code=503, detail="Redis cache is unavailable")
    options = sum((body.key is not None, body.namespace is not None, body.all_entries))
    if options != 1:
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of key, namespace, or all_entries=true",
        )
    if body.key is not None:
        removed = await state.redis.delete(body.key)
        return CacheInvalidateResponse(invalidated=int(removed))
    cache = state.components.get("cache")
    invalidate = getattr(cache, "invalidate", None)
    if callable(invalidate):
        result = invalidate(body.namespace)
        removed = await result if asyncio.iscoroutine(result) else result
        return CacheInvalidateResponse(invalidated=int(removed))

    pattern = (
        f"aigw:cache:entry:{body.namespace}:*"
        if body.namespace is not None
        else "aigw:cache:entry:*"
    )
    keys: list[str] = []
    removed = 0
    async for key in state.redis.scan_iter(match=pattern):
        keys.append(key.decode() if isinstance(key, bytes) else str(key))
        if len(keys) >= 500:
            removed += int(await state.redis.delete(*keys))
            keys.clear()
    if keys:
        removed += int(await state.redis.delete(*keys))
    return CacheInvalidateResponse(invalidated=removed)

@router.get("/system/info", response_model=SystemInfoResponse)
async def system_info(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: CurrentAdmin,
) -> SystemInfoResponse:
    try:
        db_ok = await state.db.healthy()
    except Exception:
        logger.warning("Database health check failed", exc_info=True)
        db_ok = False
    try:
        redis_ok = await state.redis_healthy()
    except Exception:
        logger.warning("Redis health check failed", exc_info=True)
        redis_ok = False
    components = state.components
    features = {
        "cache": bool(state.settings.cache_enabled and components.get("cache")),
        "guardrails": bool(components.get("guardrails")),
        "rag": bool(components.get("rag")),
        "mcp": bool(components.get("mcp")),
        "observability": bool(state.settings.metrics_enabled),
    }
    return SystemInfoResponse(
        version="0.1.0",
        uptime_seconds=time.monotonic() - _STARTED_AT,
        providers=state.registry.provider_names,
        features=features,
        database_connected=db_ok,
        redis_connected=redis_ok,
        active_routing_strategy=state.settings.routing_strategy,
    )


@router.post("/playground/chat", response_model=PlaygroundResponse)
async def playground_chat(
    body: PlaygroundRequest,
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    _: AdminUserOnly,
) -> PlaygroundResponse | StreamingResponse:
    try:
        request_metadata = dict(body.metadata)
        # The console is admin-authenticated, so its playground runs with gateway authority.
        request_metadata["api_key"] = state.settings.master_key.get_secret_value()
        request = ChatRequest(
            model=body.model,
            messages=[Message.model_validate(message) for message in body.messages],
            stream=body.stream,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stop=body.stop,
            seed=body.seed,
            presence_penalty=body.presence_penalty,
            frequency_penalty=body.frequency_penalty,
            n=body.n,
            tools=body.tools,
            tool_choice=body.tool_choice,
            parallel_tool_calls=body.parallel_tool_calls,
            response_format=body.response_format,
            user=body.user,
            metadata=request_metadata,
            no_cache=body.no_cache,
            cache_ttl=body.cache_ttl,
            fallbacks=body.fallbacks,
            routing_strategy=body.routing_strategy,
            guardrail_policy=body.guardrail_policy,
            tags=body.tags,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    ctx = RequestContext(request=request, state=state, dialect="admin")
    pipeline = state.require_pipeline()
    if body.stream:

        async def events() -> AsyncIterator[str]:
            async for chunk in pipeline.run_stream(ctx):
                yield f"data: {chunk.model_dump_json()}\n\n"
            routing = None
            if ctx.routing is not None:
                routing = {
                    "deployment_id": ctx.routing.deployment.id,
                    "strategy": ctx.routing.strategy,
                    "reason": ctx.routing.reason,
                }
            trailer = {
                "routing_decision": routing,
                "request_id": ctx.request_id,
                "deployment_id": ctx.routing.deployment.id if ctx.routing else None,
                "provider": (
                    ctx.routing.deployment.provider
                    if ctx.routing
                    else ctx.response.provider
                    if ctx.response
                    else None
                ),
                "stage_timings": ctx.stage_timings,
                "latency_ms": ctx.response.latency_ms if ctx.response else None,
                "time_to_first_token_ms": ctx.time_to_first_token_ms,
                "cache_hit": ctx.cache_hit,
                "cache_similarity": ctx.cache_similarity,
                "retry_count": len(ctx.errors),
                "fallback_count": max(len(ctx.attempted) - 1, 0),
                "fallback_used": ctx.fallback_used,
                "guardrail_flagged": ctx.guardrail_flagged,
                "guardrail_results": ctx.guardrail_results,
                "token_usage": ctx.response.usage.model_dump() if ctx.response else None,
                "estimated_cost_usd": ctx.response.cost_usd if ctx.response else None,
            }
            import json

            yield f"event: metadata\ndata: {json.dumps(trailer)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")
    response = await pipeline.run(ctx)
    routing = None
    if ctx.routing is not None:
        routing = {
            "deployment_id": ctx.routing.deployment.id,
            "strategy": ctx.routing.strategy,
            "reason": ctx.routing.reason,
        }
    return PlaygroundResponse(
        response=response.model_dump(mode="json"),
        routing_decision=routing,
        request_id=ctx.request_id,
        deployment_id=ctx.routing.deployment.id if ctx.routing else response.deployment_id,
        provider=ctx.routing.deployment.provider if ctx.routing else response.provider,
        stage_timings=ctx.stage_timings,
        latency_ms=response.latency_ms,
        time_to_first_token_ms=ctx.time_to_first_token_ms,
        cache_hit=ctx.cache_hit,
        cache_similarity=ctx.cache_similarity,
        retry_count=len(ctx.errors),
        fallback_count=max(len(ctx.attempted) - 1, 0),
        fallback_used=ctx.fallback_used,
        guardrail_flagged=ctx.guardrail_flagged,
        guardrail_results=ctx.guardrail_results,
        token_usage=response.usage,
        estimated_cost_usd=response.cost_usd,
    )
