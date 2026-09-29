"""Virtual API key creation, lookup, and lifecycle management."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.models import VirtualKey
from app.db.session import Database

logger = logging.getLogger(__name__)


def generate_key() -> tuple[str, str, str]:
    """Generate a one-time raw key, its digest, and display prefix."""
    raw_key = f"sk-aigw-{secrets.token_urlsafe(40)}"
    return raw_key, hash_key(raw_key), raw_key[:16]


def hash_key(raw: str) -> str:
    """Return the hexadecimal SHA-256 digest for a raw API key."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedKey:
    """Minimal detached key data needed during request authentication."""

    id: str
    team_id: str | None
    is_active: bool
    expires_at: datetime | None
    allowed_models: list[str]
    blocked_models: list[str]
    rpm_limit: int | None
    tpm_limit: int | None
    max_budget_usd: Decimal | None
    spend_usd: Decimal
    guardrail_policy: str | None
    max_parallel_requests: int | None = None
    allowed_routes: list[str] = field(default_factory=list)

    def is_valid(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        expiry = self.expires_at
        if expiry is not None and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        return self.is_active and not (expiry is not None and expiry <= now)

    def permits_model(self, model: str) -> bool:
        if model in self.blocked_models:
            return False
        return not self.allowed_models or model in self.allowed_models

    def permits_route(self, route: str) -> bool:
        """Routes are allowed by path prefix, e.g. "/v1/chat" or "/v1/embeddings"."""
        return not self.allowed_routes or any(
            route.startswith(prefix) for prefix in self.allowed_routes
        )


def _resolved_from_row(row: VirtualKey) -> ResolvedKey:
    return ResolvedKey(
        id=row.id,
        team_id=row.team_id,
        is_active=row.is_active,
        expires_at=row.expires_at,
        allowed_models=list(row.allowed_models or []),
        blocked_models=list(row.blocked_models or []),
        rpm_limit=row.rpm_limit,
        tpm_limit=row.tpm_limit,
        max_budget_usd=Decimal(str(row.max_budget_usd)) if row.max_budget_usd is not None else None,
        spend_usd=Decimal(str(row.spend_usd or 0)),
        guardrail_policy=row.guardrail_policy,
        max_parallel_requests=row.max_parallel_requests,
        allowed_routes=list(row.allowed_routes or []),
    )


def _serialize_key(key: ResolvedKey) -> str:
    data = {
        "id": key.id,
        "team_id": key.team_id,
        "is_active": key.is_active,
        "expires_at": key.expires_at.isoformat() if key.expires_at else None,
        "allowed_models": key.allowed_models,
        "blocked_models": key.blocked_models,
        "rpm_limit": key.rpm_limit,
        "tpm_limit": key.tpm_limit,
        "max_budget_usd": str(key.max_budget_usd) if key.max_budget_usd is not None else None,
        "spend_usd": str(key.spend_usd),
        "guardrail_policy": key.guardrail_policy,
        "max_parallel_requests": key.max_parallel_requests,
        "allowed_routes": key.allowed_routes,
    }
    return json.dumps(data, separators=(",", ":"))


def _deserialize_key(snapshot: str | bytes) -> ResolvedKey:
    if isinstance(snapshot, bytes):
        snapshot = snapshot.decode("utf-8")
    data = json.loads(snapshot)
    expiry = datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None
    if expiry is not None and expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    budget = data["max_budget_usd"]
    return ResolvedKey(
        id=data["id"],
        team_id=data["team_id"],
        is_active=data["is_active"],
        expires_at=expiry,
        allowed_models=data["allowed_models"],
        blocked_models=data["blocked_models"],
        rpm_limit=data["rpm_limit"],
        tpm_limit=data["tpm_limit"],
        max_budget_usd=Decimal(budget) if budget is not None else None,
        spend_usd=Decimal(data["spend_usd"]),
        guardrail_policy=data["guardrail_policy"],
        # .get: snapshots cached before these fields existed stay readable.
        max_parallel_requests=data.get("max_parallel_requests"),
        allowed_routes=data.get("allowed_routes") or [],
    )


async def invalidate_cache(redis: Any, key_hash: str) -> None:
    """Remove one resolved-key snapshot without exposing its raw key."""
    if redis is None:
        return
    try:
        await redis.delete(f"aigw:key:{key_hash}")
    except Exception:
        logger.warning("Could not invalidate virtual-key cache", exc_info=True)


class KeyService:
    def __init__(self, db: Database, redis: Any) -> None:
        self.db = db
        self.redis = redis

    async def invalidate_cache(self, key_hash: str) -> None:
        await invalidate_cache(self.redis, key_hash)

    async def create_key(
        self,
        name: str,
        team_id: str | None = None,
        max_budget_usd: Decimal | float | None = None,
        rpm_limit: int | None = None,
        tpm_limit: int | None = None,
        allowed_models: list[str] | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[VirtualKey, str]:
        raw_key, key_digest, prefix = generate_key()
        async with self.db.session() as session:
            row = VirtualKey(
                key_hash=key_digest,
                key_prefix=prefix,
                name=name,
                team_id=team_id,
                max_budget_usd=max_budget_usd,
                rpm_limit=rpm_limit,
                tpm_limit=tpm_limit,
                allowed_models=allowed_models or [],
                expires_at=expires_at,
                metadata_=metadata or {},
            )
            session.add(row)
            await session.flush()
        return row, raw_key

    async def lookup(self, raw_key: str) -> ResolvedKey | None:
        key_digest = hash_key(raw_key)
        cache_key = f"aigw:key:{key_digest}"
        if self.redis is not None:
            try:
                cached = await self.redis.get(cache_key)
                if cached is not None:
                    return _deserialize_key(cached)
            except Exception:
                logger.warning("Virtual-key cache read failed; querying database", exc_info=True)

        async with self.db.session() as session:
            result = await session.execute(
                select(VirtualKey)
                .options(selectinload(VirtualKey.team))
                .where(VirtualKey.key_hash == key_digest)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            resolved = _resolved_from_row(row)

        if self.redis is not None:
            try:
                await self.redis.set(cache_key, _serialize_key(resolved), ex=30)
            except Exception:
                logger.warning("Virtual-key cache write failed", exc_info=True)
        return resolved

    async def revoke(self, key_id: str) -> None:
        old_hash: str | None = None
        async with self.db.session() as session:
            row = await session.get(VirtualKey, key_id)
            if row is not None:
                old_hash = row.key_hash
                row.is_active = False
        if old_hash is not None:
            await self.invalidate_cache(old_hash)

    async def rotate(self, key_id: str) -> tuple[ResolvedKey, str]:
        raw_key, new_hash, prefix = generate_key()
        async with self.db.session() as session:
            row = await session.get(VirtualKey, key_id)
            if row is None:
                raise LookupError(f"Virtual key {key_id!r} does not exist")
            old_hash = row.key_hash
            row.key_hash = new_hash
            row.key_prefix = prefix
            await session.flush()
            resolved = _resolved_from_row(row)
        await self.invalidate_cache(old_hash)
        await self.invalidate_cache(new_hash)
        return resolved, raw_key

    async def list_keys(self, limit: int, offset: int) -> list[VirtualKey]:
        async with self.db.session() as session:
            result = await session.execute(
                select(VirtualKey)
                .options(selectinload(VirtualKey.team))
                .order_by(VirtualKey.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    async def update_key(self, key_id: str, **fields: Any) -> VirtualKey:
        allowed = {
            "name", "team_id", "max_budget_usd", "spend_usd", "budget_period",
            "budget_reset_at", "rpm_limit", "tpm_limit", "max_parallel_requests",
            "allowed_models", "blocked_models", "guardrail_policy", "allowed_routes",
            "is_active", "expires_at", "metadata", "metadata_",
        }
        unknown = fields.keys() - allowed
        if unknown:
            raise ValueError(f"Unsupported virtual-key fields: {', '.join(sorted(unknown))}")
        async with self.db.session() as session:
            row = await session.get(VirtualKey, key_id)
            if row is None:
                raise LookupError(f"Virtual key {key_id!r} does not exist")
            for field_name, value in fields.items():
                setattr(row, "metadata_" if field_name == "metadata" else field_name, value)
            key_digest = row.key_hash
            await session.flush()
        await self.invalidate_cache(key_digest)
        return row
