"""Admin-console password authentication, JWTs, and role dependencies."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import func, select

from app.config.settings import get_settings
from app.core.errors import AuthenticationError
from app.db.models import AdminUser
from app.db.session import Database

logger = logging.getLogger(__name__)
_password_hasher = PasswordHasher()
_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        return _password_hasher.verify(encoded_hash, password)
    except VerifyMismatchError:
        return False


def _create_token(
    user_id: str,
    token_type: str,
    ttl_seconds: int,
    email: str | None = None,
    role: str | None = None,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": user_id,
        "type": token_type,
        "iat": now,
        "exp": now + ttl_seconds,
    }
    if email is not None:
        payload["email"] = email
    if role is not None:
        payload["role"] = role
    settings = get_settings()
    return jwt.encode(
        payload,
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def create_access_token(user_id: str, email: str, role: str) -> str:
    return _create_token(user_id, "access", get_settings().jwt_access_ttl_seconds, email, role)


def create_refresh_token(user_id: str) -> str:
    return _create_token(user_id, "refresh", get_settings().jwt_refresh_ttl_seconds)


def decode_token(token: str, expected_type: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Invalid or expired console token") from exc
    if payload.get("type") != expected_type:
        raise AuthenticationError("Invalid console token type")
    return payload


class ConsoleAuthService:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def authenticate(self, email: str, password: str) -> AdminUser:
        async with self.db.session() as session:
            result = await session.execute(
                select(AdminUser).where(func.lower(AdminUser.email) == email.strip().lower())
            )
            user = result.scalar_one_or_none()
            if (
                user is None
                or not user.is_active
                or not verify_password(password, user.password_hash)
            ):
                raise AuthenticationError("Invalid email or password")
            user.last_login_at = datetime.now(UTC)
            await session.flush()
            return user

    async def create_user(
        self, email: str, password: str, role: str, full_name: str | None = None
    ) -> AdminUser:
        if role not in {"admin", "viewer"}:
            raise ValueError("Role must be 'admin' or 'viewer'")
        async with self.db.session() as session:
            user = AdminUser(
                email=email.strip().lower(),
                password_hash=hash_password(password),
                role=role,
                full_name=full_name,
            )
            session.add(user)
            await session.flush()
            return user

    async def ensure_bootstrap_admin(self) -> None:
        settings = get_settings()
        async with self.db.session() as session:
            count = await session.scalar(select(func.count()).select_from(AdminUser))
            if count:
                return
            password = settings.bootstrap_admin_password.get_secret_value()
            if password == "admin":
                logger.warning(
                    "SECURITY WARNING: bootstrap administrator is using the default "
                    "password 'admin'"
                )
            session.add(
                AdminUser(
                    email=settings.bootstrap_admin_email.strip().lower(),
                    password_hash=hash_password(password),
                    role="admin",
                    is_active=True,
                )
            )


@dataclass(frozen=True)
class ConsolePrincipal:
    id: str
    email: str
    role: str


async def require_console_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> ConsolePrincipal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError("A Bearer access token is required")
    payload = decode_token(credentials.credentials, "access")
    user_id = payload.get("sub")
    email = payload.get("email")
    role = payload.get("role")
    if not isinstance(user_id, str) or not isinstance(email, str) or not isinstance(role, str):
        raise AuthenticationError("Console token is missing required claims")
    return ConsolePrincipal(id=user_id, email=email, role=role)


async def require_admin_role(
    principal: ConsolePrincipal = Depends(require_console_user),
) -> ConsolePrincipal:
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator role required")
    return principal
