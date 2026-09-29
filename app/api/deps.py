"""Shared dependencies for the admin API."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from app.auth.console import decode_token
from app.auth.keys import KeyService
from app.core.errors import GatewayError
from app.core.state import GatewayState
from app.db.models import AdminUser


def get_gateway_state(request: Request) -> GatewayState:
    """Resolve the process-owned gateway state from the current app."""
    state: GatewayState | None = getattr(request.app.state, "gateway", None)
    if state is None:
        raise HTTPException(status_code=503, detail="Gateway state is not initialized")
    return state


_bearer = HTTPBearer(auto_error=False)


async def get_current_admin(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> AdminUser:
    """Decode the existing console JWT and verify the database user is active."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="A console bearer token is required")
    try:
        claims = decode_token(credentials.credentials, "access")
    except GatewayError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    user_id = claims.get("sub")
    if not isinstance(user_id, str):
        raise HTTPException(status_code=401, detail="Console token is missing its subject")
    async with state.db.session() as session:
        user = await session.scalar(select(AdminUser).where(AdminUser.id == user_id))
        if user is None or not user.is_active:
            raise HTTPException(status_code=401, detail="Invalid or inactive console user")
        return user


def require_role(role: str) -> Any:
    """Return a dependency requiring the authenticated user to have ``role``."""

    async def role_guard(
        user: Annotated[AdminUser, Depends(get_current_admin)],
    ) -> AdminUser:
        if role == "admin" and user.role != "admin":
            raise HTTPException(status_code=403, detail="Administrator role required")
        if role not in {"admin", "viewer"}:
            raise ValueError(f"Unsupported console role: {role}")
        return user

    return role_guard


@dataclass(frozen=True)
class GatewayPrincipal:
    """Who is calling a data-plane route, and by which credential."""

    kind: str
    identifier: str


async def require_gateway_principal(
    state: Annotated[GatewayState, Depends(get_gateway_state)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
) -> GatewayPrincipal:
    """Authenticate a gateway data-plane route.

    RAG and MCP routes are reachable both by applications holding a virtual key
    and by operators driving the console, so either credential is accepted. A
    console JWT is tried last because virtual keys are the common case.
    """
    raw = x_api_key
    if raw is None and credentials is not None and credentials.scheme.lower() == "bearer":
        raw = credentials.credentials
    if not raw:
        raise HTTPException(status_code=401, detail="An API key or console token is required")

    if hmac.compare_digest(raw, state.settings.master_key.get_secret_value()):
        return GatewayPrincipal(kind="master", identifier="master")

    key = await KeyService(state.db, state.redis).lookup(raw)
    if key is not None and key.is_valid():
        return GatewayPrincipal(kind="virtual_key", identifier=key.id)

    try:
        claims = decode_token(raw, "access")
    except GatewayError:
        raise HTTPException(status_code=401, detail="Invalid API key or console token") from None
    user_id = claims.get("sub")
    if not isinstance(user_id, str):
        raise HTTPException(status_code=401, detail="Console token is missing its subject")
    async with state.db.session() as session:
        user = await session.scalar(select(AdminUser).where(AdminUser.id == user_id))
        if user is None or not user.is_active:
            raise HTTPException(status_code=401, detail="Invalid or inactive console user")
    return GatewayPrincipal(kind="console", identifier=user_id)


@dataclass(frozen=True)
class PaginationParams:
    limit: int = 50
    offset: int = 0
    order_by: str = "created_at"
    order_dir: str = "desc"


def pagination_params(
    limit: int = Query(default=50, ge=1),
    offset: int = Query(default=0, ge=0),
    order_by: str = Query(default="created_at", min_length=1),
    order_dir: str = Query(default="desc", pattern="^(asc|desc)$"),
) -> PaginationParams:
    """Normalize list paging with a hard cap to keep requests bounded."""
    return PaginationParams(
        limit=min(limit, 200),
        offset=offset,
        order_by=order_by,
        order_dir=order_dir,
    )


Pagination = Annotated[PaginationParams, Depends(pagination_params)]
CurrentAdmin = Annotated[AdminUser, Depends(get_current_admin)]
AdminUserOnly = Annotated[AdminUser, Depends(require_role("admin"))]
GatewayCaller = Annotated[GatewayPrincipal, Depends(require_gateway_principal)]
