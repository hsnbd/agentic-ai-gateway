"""Liveness, readiness, and metrics endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response

from app.core.state import GatewayState

router = APIRouter(tags=["health"])


def _state(request: Request) -> GatewayState:
    state: GatewayState = request.app.state.gateway
    return state


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    """Always 200 while the process is running."""
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe")
async def readyz(request: Request, response: Response) -> dict[str, Any]:
    """503 unless every dependency the gateway needs is reachable."""
    state = _state(request)
    checks = {
        "database": await state.db.healthy(),
        "redis": await state.redis_healthy(),
        "providers": bool(state.registry.list_deployments()),
    }
    ready = all(checks.values())
    if not ready:
        response.status_code = 503
    return {"status": "ready" if ready else "not_ready", "checks": checks}


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    """Prometheus exposition endpoint."""
    state = _state(request)
    if not state.settings.metrics_enabled:
        return Response(status_code=404)

    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
