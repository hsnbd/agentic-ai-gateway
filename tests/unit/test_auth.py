from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import jwt
import pytest

from app.auth.console import (
    ConsoleAuthService,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.auth.keys import KeyService, ResolvedKey, generate_key, hash_key
from app.auth.quotas import QuotaService
from app.auth.ratelimit import SlidingWindowLimiter
from app.auth.stage import AuthStage
from app.config.settings import Settings
from app.core.errors import AuthenticationError, BudgetExceededError
from app.core.schemas import ChatRequest
from app.db.session import Database


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.commands: list[tuple[str, tuple[Any, ...]]] = []

    def execute_command(self, command: str, *args: Any) -> FakePipeline:
        self.commands.append((command.lower(), args))
        return self

    def zremrangebyscore(self, key: str, minimum: str, maximum: float) -> FakePipeline:
        self.commands.append(("zremrangebyscore", (key, minimum, maximum)))
        return self

    def zcard(self, key: str) -> FakePipeline:
        self.commands.append(("zcard", (key,)))
        return self

    def zadd(self, key: str, members: dict[str, float]) -> FakePipeline:
        self.commands.append(("zadd", (key, members)))
        return self

    def expire(self, key: str, seconds: int) -> FakePipeline:
        self.commands.append(("expire", (key, seconds)))
        return self

    async def execute(self) -> list[Any]:
        results: list[Any] = []
        for command, args in self.commands:
            results.append(await getattr(self.redis, command)(*args))
        return results


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.sorted_sets: dict[str, dict[str, float]] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            removed += int(self.values.pop(key, None) is not None)
            removed += int(self.sorted_sets.pop(key, None) is not None)
        return removed

    async def incrbyfloat(self, key: str, amount: float) -> float:
        value = float(self.values.get(key, "0")) + amount
        self.values[key] = str(value)
        return value

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        assert transaction
        return FakePipeline(self)

    async def eval(
        self,
        script: str,
        number_of_keys: int,
        key: str,
        now: float,
        window: int,
        limit: int,
        cost: int,
        *members: str,
    ) -> list[int]:
        del script
        assert number_of_keys == 1
        await self.zremrangebyscore(key, "-inf", now - window)
        used = await self.zcard(key)
        if used + cost > limit:
            return [0, max(0, limit - used), window]
        await self.zadd(key, dict.fromkeys(members, now))
        await self.expire(key, max(1, window))
        return [1, max(0, limit - used - cost), 0]

    async def zremrangebyscore(self, key: str, minimum: str, maximum: float) -> int:
        entries = self.sorted_sets.setdefault(key, {})
        expired = [member for member, score in entries.items() if score <= maximum]
        for member in expired:
            del entries[member]
        return len(expired)

    async def zcard(self, key: str) -> int:
        return len(self.sorted_sets.get(key, {}))

    async def zadd(self, key: str, members: dict[str, float]) -> int:
        entries = self.sorted_sets.setdefault(key, {})
        added = 0
        for member, score in members.items():
            added += int(member not in entries)
            entries[member] = score
        return added

    async def expire(self, key: str, seconds: int) -> bool:
        return True


class BrokenRedis:
    def pipeline(self, transaction: bool = True) -> BrokenRedis:
        return self

    def zremrangebyscore(self, *args: Any) -> BrokenRedis:
        return self

    def zcard(self, *args: Any) -> BrokenRedis:
        return self

    async def execute(self) -> list[Any]:
        raise ConnectionError("Redis unavailable")


@pytest.fixture
async def services() -> Any:
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        jwt_secret="test-jwt-secret-that-is-long-enough-for-hs256",
        master_key="master-test-key",
    )
    db = Database(settings)
    await db.startup()
    await db.create_all()
    redis = FakeRedis()
    yield db, redis, settings
    await db.shutdown()


def make_request(api_key: str, model: str = "test-model") -> ChatRequest:
    return ChatRequest(
        model=model,
        messages=[],
        metadata={"api_key": api_key},
    )


async def make_context(db: Database, redis: FakeRedis, settings: Settings, api_key: str) -> Any:
    state = SimpleNamespace(db=db, redis=redis, settings=settings)
    return SimpleNamespace(request=make_request(api_key), state=state)


def test_key_generation_format_and_hash_stability() -> None:
    raw, digest, prefix = generate_key()
    assert raw.startswith("sk-aigw-")
    assert len(raw.removeprefix("sk-aigw-")) >= 40
    assert digest == hash_key(raw)
    assert len(digest) == 64
    assert prefix == raw[:16]
    assert hash_key(raw) == hash_key(raw)


def test_resolved_key_model_permissions() -> None:
    key = ResolvedKey(
        id="k",
        team_id=None,
        is_active=True,
        expires_at=None,
        allowed_models=["allowed"],
        blocked_models=["blocked"],
        rpm_limit=None,
        tpm_limit=None,
        max_budget_usd=None,
        spend_usd=Decimal(0),
        guardrail_policy=None,
    )
    assert key.permits_model("allowed")
    assert not key.permits_model("blocked")
    assert not key.permits_model("other")


@pytest.mark.parametrize("expired,inactive", [(True, False), (False, True)])
async def test_auth_rejects_expired_or_inactive_keys(
    services: Any, expired: bool, inactive: bool
) -> None:
    db, redis, settings = services
    key_service = KeyService(db, redis)
    expiry = datetime.now(UTC) - timedelta(seconds=1) if expired else None
    row, raw = await key_service.create_key("test", expires_at=expiry)
    if inactive:
        await key_service.update_key(row.id, is_active=False)
    context = await make_context(db, redis, settings, raw)
    with pytest.raises(AuthenticationError):
        await AuthStage().process(context)


async def test_budget_exceeded_raises(services: Any) -> None:
    db, redis, _ = services
    key_service = KeyService(db, redis)
    row, raw = await key_service.create_key("budgeted", max_budget_usd=1)
    resolved = await key_service.lookup(raw)
    assert resolved is not None
    await QuotaService(db, redis).record_spend(row.id, None, 1.0)
    with pytest.raises(BudgetExceededError):
        await QuotaService(db, redis).check_budget(resolved)


async def test_sliding_window_limiter_allows_and_blocks() -> None:
    limiter = SlidingWindowLimiter(FakeRedis())
    assert await limiter.check_and_consume("rpm", 2, 60) == (True, 1, 0.0)
    allowed, remaining, _ = await limiter.check_and_consume("rpm", 2, 60)
    assert allowed and remaining == 0
    allowed, remaining, retry_after = await limiter.check_and_consume("rpm", 2, 60)
    assert not allowed and remaining == 0 and retry_after > 0
    assert await limiter.peek("rpm", 2, 60) == (2, 0)


async def test_limiter_fails_open_when_redis_raises() -> None:
    allowed, remaining, retry_after = await SlidingWindowLimiter(BrokenRedis()).check_and_consume(
        "rpm", 3, 60
    )
    assert allowed and remaining == 3 and retry_after == 0


def test_password_hash_verify_round_trip() -> None:
    encoded = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("incorrect", encoded)


async def test_jwt_round_trip_rejects_wrong_type_and_expired(
    services: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, settings = services
    monkeypatch.setattr("app.auth.console.get_settings", lambda: settings)
    access = create_access_token("u1", "user@example.com", "admin")
    claims = decode_token(access, "access")
    assert claims["sub"] == "u1"
    assert claims["role"] == "admin"
    with pytest.raises(AuthenticationError):
        decode_token(create_refresh_token("u1"), "access")

    expired = jwt.encode(
        {"sub": "u1", "type": "access", "iat": int(time.time()) - 10, "exp": int(time.time()) - 1},
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(AuthenticationError):
        decode_token(expired, "access")


async def test_master_key_bypasses_database_and_quotas(services: Any) -> None:
    db, redis, settings = services
    context = await make_context(db, redis, settings, settings.master_key.get_secret_value())
    result = await AuthStage().process(context)
    assert result is None
    assert context.key_id == "master"
    assert context.virtual_key is None


async def test_console_authentication_updates_login_time(services: Any) -> None:
    db, _, _ = services
    auth = ConsoleAuthService(db)
    user = await auth.create_user("Admin@Example.com", "secret", "admin")
    assert user.email == "admin@example.com"
    authenticated = await auth.authenticate("ADMIN@example.com", "secret")
    assert authenticated.last_login_at is not None
    with pytest.raises(AuthenticationError):
        await auth.authenticate("admin@example.com", "wrong")


async def test_key_lookup_caches_compact_snapshot(services: Any) -> None:
    db, redis, _ = services
    service = KeyService(db, redis)
    row, raw = await service.create_key("cached", allowed_models=["m"], rpm_limit=5)
    first = await service.lookup(raw)
    second = await service.lookup(raw)
    assert first is not None and second == first
    assert first.id == row.id
    assert f"aigw:key:{hash_key(raw)}" in redis.values


async def test_key_revocation_invalidates_snapshot(services: Any) -> None:
    db, redis, _ = services
    service = KeyService(db, redis)
    row, raw = await service.create_key("revocable")
    await service.lookup(raw)
    await service.revoke(row.id)
    assert f"aigw:key:{hash_key(raw)}" not in redis.values
    assert await service.lookup(raw) is not None
    assert not (await service.lookup(raw)).is_valid()
