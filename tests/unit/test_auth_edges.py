"""Edge cases for keys, quotas, limiters, revocation, the auth stage, and console auth."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import SecretStr

from app.auth import console as console_module
from app.auth.console import (
    ConsoleAuthService,
    ConsolePrincipal,
    create_access_token,
    require_admin_role,
    require_console_user,
)
from app.auth.keys import (
    KeyService,
    ResolvedKey,
    _deserialize_key,
    _serialize_key,
    invalidate_cache,
)
from app.auth.quotas import QuotaService, _advance_reset, _aware
from app.auth.ratelimit import ConcurrencyLimiter, SlidingWindowLimiter, TokenWindow
from app.auth.sessions import TokenRevocation
from app.auth.stage import AuthStage, deployment_tpm_counter, key_tpm_counter
from app.config.settings import Settings
from app.core.errors import (
    AuthenticationError,
    BudgetExceededError,
    PermissionDeniedError,
    RateLimitExceededError,
)
from app.core.schemas import ChatRequest, Message, Role
from app.db.models import Team, VirtualKey
from app.db.session import Database
from tests.unit.test_auth import FakeRedis


class ExplodingRedis:
    """Every command raises, to exercise each fail-open path."""

    def __getattr__(self, name: str) -> Any:
        async def _fail(*args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("redis down")

        if name == "pipeline":
            return lambda transaction=True: ExplodingPipeline()
        return _fail


class ExplodingPipeline:
    def __getattr__(self, name: str) -> Any:
        if name == "execute":

            async def _fail() -> Any:
                raise ConnectionError("redis down")

            return _fail
        return lambda *args, **kwargs: self


class CounterRedis(FakeRedis):
    """FakeRedis plus the counter commands the token and concurrency limiters use."""

    def __init__(self) -> None:
        super().__init__()
        self.counters: dict[str, int] = {}

    async def mget(self, *keys: str) -> list[Any]:
        return [self.counters.get(key, self.values.get(key)) for key in keys]

    async def mset(self, mapping: dict[str, str]) -> None:
        self.values.update(mapping)

    async def incrby(self, key: str, amount: int) -> int:
        self.counters[key] = self.counters.get(key, 0) + amount
        return self.counters[key]

    async def incr(self, key: str) -> int:
        return await self.incrby(key, 1)

    async def decr(self, key: str) -> int:
        return await self.incrby(key, -1)

    async def set(self, key: str, value: Any, ex: int | None = None) -> None:
        if isinstance(value, int):
            self.counters[key] = value
        else:
            self.values[key] = value

    def pipeline(self, transaction: bool = True) -> Any:
        return CounterPipeline(self)


class CounterPipeline:
    def __init__(self, redis: CounterRedis) -> None:
        self.redis = redis
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def __getattr__(self, name: str) -> Any:
        def _queue(*args: Any) -> CounterPipeline:
            self.calls.append((name, args))
            return self

        return _queue

    async def execute(self) -> list[Any]:
        results = []
        for name, args in self.calls:
            if name == "execute_command":
                results.append(await self.redis.eval(*args[1:]))
            else:
                results.append(await getattr(self.redis, name)(*args))
        return results


@pytest.fixture
async def db() -> Any:
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        jwt_secret="test-jwt-secret-that-is-long-enough-for-hs256",
        master_key="master-test-key",
        bootstrap_admin_email="Boot@Example.com",
        bootstrap_admin_password="admin",
    )
    database = Database(settings)
    await database.startup()
    await database.create_all()
    database.settings = settings  # type: ignore[attr-defined]
    yield database
    await database.shutdown()


def resolved(**fields: Any) -> ResolvedKey:
    defaults: dict[str, Any] = {
        "id": "k1",
        "team_id": None,
        "is_active": True,
        "expires_at": None,
        "allowed_models": [],
        "blocked_models": [],
        "rpm_limit": None,
        "tpm_limit": None,
        "max_budget_usd": None,
        "spend_usd": Decimal(0),
        "guardrail_policy": None,
    }
    return ResolvedKey(**{**defaults, **fields})


# -- ResolvedKey and snapshots ----------------------------------------------


def test_resolved_key_validity_handles_naive_and_past_expiry() -> None:
    now = datetime.now(UTC)
    assert resolved(expires_at=(now + timedelta(hours=1)).replace(tzinfo=None)).is_valid()
    assert not resolved(expires_at=now - timedelta(seconds=1)).is_valid()
    assert not resolved(is_active=False).is_valid()


def test_route_permissions_are_prefix_based() -> None:
    assert resolved().permits_route("/v1/anything")
    key = resolved(allowed_routes=["/v1/chat"])
    assert key.permits_route("/v1/chat/completions")
    assert not key.permits_route("/v1/embeddings")


def test_snapshot_round_trip_including_bytes_and_legacy_fields() -> None:
    key = resolved(
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
        max_budget_usd=Decimal("5.5"),
        max_parallel_requests=3,
        allowed_routes=["/v1/chat"],
    )
    snapshot = _serialize_key(key)
    assert _deserialize_key(snapshot.encode()) == key

    legacy = (
        '{"id":"k","team_id":null,"is_active":true,"expires_at":"2030-01-01T00:00:00",'
        '"allowed_models":[],"blocked_models":[],"rpm_limit":null,"tpm_limit":null,'
        '"max_budget_usd":null,"spend_usd":"0","guardrail_policy":null}'
    )
    loaded = _deserialize_key(legacy)
    assert loaded.expires_at == datetime(2030, 1, 1, tzinfo=UTC)
    assert loaded.max_parallel_requests is None
    assert loaded.allowed_routes == []


async def test_invalidate_cache_tolerates_missing_or_broken_redis() -> None:
    await invalidate_cache(None, "digest")
    await invalidate_cache(ExplodingRedis(), "digest")


# -- KeyService -------------------------------------------------------------


async def test_lookup_without_redis_and_with_broken_redis(db: Database) -> None:
    row, raw = await KeyService(db, None).create_key("plain", metadata={"a": 1})
    assert (await KeyService(db, None).lookup(raw)).id == row.id  # type: ignore[union-attr]
    assert (await KeyService(db, ExplodingRedis()).lookup(raw)).id == row.id  # type: ignore[union-attr]
    assert await KeyService(db, None).lookup("sk-aigw-unknown") is None


async def test_revoke_unknown_key_is_a_no_op(db: Database) -> None:
    await KeyService(db, FakeRedis()).revoke("missing")


async def test_rotate_replaces_hash_and_rejects_unknown(db: Database) -> None:
    redis = FakeRedis()
    service = KeyService(db, redis)
    row, raw = await service.create_key("rotating")
    await service.lookup(raw)
    rotated, new_raw = await service.rotate(row.id)
    assert rotated.id == row.id
    assert new_raw != raw
    assert await service.lookup(raw) is None
    assert (await service.lookup(new_raw)).id == row.id  # type: ignore[union-attr]
    with pytest.raises(LookupError):
        await service.rotate("missing")


async def test_list_and_update_keys(db: Database) -> None:
    service = KeyService(db, None)
    first, _ = await service.create_key("one")
    await service.create_key("two")
    assert {key.name for key in await service.list_keys(10, 0)} == {"one", "two"}
    assert len(await service.list_keys(1, 1)) == 1

    updated = await service.update_key(first.id, name="renamed", metadata={"owner": "me"})
    assert updated.name == "renamed"
    assert updated.metadata_ == {"owner": "me"}
    with pytest.raises(ValueError, match="Unsupported"):
        await service.update_key(first.id, key_hash="x")
    with pytest.raises(LookupError):
        await service.update_key("missing", name="x")


def test_orm_key_helpers() -> None:
    now = datetime.now(UTC)
    row = VirtualKey(is_active=True, expires_at=None, allowed_models=["a"], blocked_models=["b"])
    assert row.is_valid(now)
    row.expires_at = now - timedelta(seconds=1)
    assert not row.is_valid(now)
    assert not row.is_valid()
    row.is_active = False
    assert not row.is_valid(now)
    assert row.permits_model("a")
    assert not row.permits_model("b")
    assert not row.permits_model("c")


# -- Quotas -----------------------------------------------------------------


def test_aware_and_reset_periods() -> None:
    naive = datetime(2026, 1, 31, 12, 0)
    assert _aware(naive).tzinfo is UTC
    aware = datetime(2026, 1, 31, tzinfo=UTC)
    assert _aware(aware) is aware

    now = datetime(2026, 2, 3, tzinfo=UTC)
    assert _advance_reset(aware, "daily", now) == datetime(2026, 2, 4, tzinfo=UTC)
    assert _advance_reset(aware, "weekly", now) == datetime(2026, 2, 7, tzinfo=UTC)
    # Jan 31 -> Feb 28 (clamped) -> Mar 28.
    assert _advance_reset(aware, "monthly", now) == datetime(2026, 2, 28, tzinfo=UTC)
    december = datetime(2025, 12, 15, tzinfo=UTC)
    assert _advance_reset(december, "monthly", datetime(2025, 12, 20, tzinfo=UTC)) == datetime(
        2026, 1, 15, tzinfo=UTC
    )
    with pytest.raises(ValueError, match="Unsupported budget period"):
        _advance_reset(aware, "hourly", now)


async def test_current_spend_prefers_redis_and_falls_back(db: Database) -> None:
    redis = CounterRedis()
    redis.values["counter"] = b"2.5"  # type: ignore[assignment]
    service = QuotaService(db, redis)
    assert await service._current_spend("counter", Decimal("1")) == Decimal("2.5")
    assert await service._current_spend("missing", Decimal("1")) == Decimal("1")
    assert await QuotaService(db, ExplodingRedis())._current_spend("x", 3) == Decimal("3")
    assert await QuotaService(db, None)._current_spend("x", 0) == Decimal("0")


async def test_team_budget_is_enforced(db: Database) -> None:
    async with db.session() as session:
        team = Team(name="t", max_budget_usd=Decimal("1"), spend_usd=Decimal("0"))
        unlimited = Team(name="free")
        session.add_all([team, unlimited])
    redis = CounterRedis()
    service = QuotaService(db, redis)
    key_row, _ = await KeyService(db, redis).create_key("teamed", team_id=team.id)
    key = resolved(id=key_row.id, team_id=team.id)

    await service.check_budget(key)
    await service.record_spend(key_row.id, team.id, 2)
    assert redis.values[f"aigw:spend:team:{team.id}"] == "2.000000"
    with pytest.raises(BudgetExceededError, match="Team budget"):
        await service.check_budget(key)

    await service.check_budget(resolved(id=key_row.id, team_id=unlimited.id))
    await service.check_budget(resolved(id="ghost", team_id="ghost-team"))


async def test_record_spend_edge_cases(db: Database) -> None:
    # Unknown key and team: nothing to mirror into Redis.
    redis = CounterRedis()
    await QuotaService(db, redis).record_spend("ghost", "ghost-team", 1)
    assert redis.values == {}
    row, _ = await KeyService(db, None).create_key("spender")
    await QuotaService(db, None).record_spend(row.id, None, 1)
    await QuotaService(db, ExplodingRedis()).record_spend(row.id, None, 1)
    async with db.session() as session:
        assert (await session.get(VirtualKey, row.id)).spend_usd == Decimal(2)  # type: ignore[union-attr]


async def test_reset_if_due_resets_key_and_team(db: Database) -> None:
    past = datetime.now(UTC) - timedelta(days=2)
    future = datetime.now(UTC) + timedelta(days=2)
    async with db.session() as session:
        team = Team(
            name="resetting", spend_usd=Decimal(5), budget_reset_at=past, budget_period="daily"
        )
        not_due = Team(name="later", spend_usd=Decimal(5), budget_reset_at=future)
        no_reset = Team(name="never", spend_usd=Decimal(5))
        session.add_all([team, not_due, no_reset])
    service = KeyService(db, None)
    key_row, _ = await service.create_key("resetting", team_id=team.id)
    await service.update_key(
        key_row.id, spend_usd=Decimal(5), budget_reset_at=past, budget_period="weekly"
    )

    redis = CounterRedis()
    redis.values[f"aigw:spend:key:{key_row.id}"] = "5"
    await QuotaService(db, redis).reset_if_due(resolved(id=key_row.id, team_id=team.id))
    assert f"aigw:spend:key:{key_row.id}" not in redis.values
    async with db.session() as session:
        key = await session.get(VirtualKey, key_row.id)
        reset_team = await session.get(Team, team.id)
        assert key.spend_usd == 0  # type: ignore[union-attr]
        assert reset_team.spend_usd == 0  # type: ignore[union-attr]

    # Not due, no reset date, unknown rows, and a failing Redis are all tolerated.
    quotas = QuotaService(db, ExplodingRedis())
    await quotas.reset_if_due(resolved(id=key_row.id, team_id=not_due.id))
    await quotas.reset_if_due(resolved(id="ghost", team_id=no_reset.id))
    await quotas.reset_if_due(resolved(id="ghost", team_id="ghost-team"))
    await service.update_key(key_row.id, budget_reset_at=past)
    await quotas.reset_if_due(resolved(id=key_row.id))
    async with db.session() as session:
        assert (await session.get(Team, not_due.id)).spend_usd == 5  # type: ignore[union-attr]


# -- Limiters ---------------------------------------------------------------


async def test_peek_fails_open() -> None:
    assert await SlidingWindowLimiter(ExplodingRedis()).peek("k", 5, 60) == (0, 5.0)


async def test_token_window_usage_add_and_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = CounterRedis()
    window = TokenWindow(redis)
    monkeypatch.setattr("app.auth.ratelimit.time.time", lambda: 120.0)
    await window.add("tpm", 0)
    assert redis.counters == {}
    await window.add("tpm", 10)
    assert await window.usage("tpm") == 10
    monkeypatch.setattr("app.auth.ratelimit.time.time", lambda: 210.0)
    # Half of the previous minute still overlaps the window.
    assert await window.usage("tpm") == 5
    assert window.seconds_until_reset() == 30

    broken = TokenWindow(ExplodingRedis())
    assert await broken.usage("tpm") == 0.0
    await broken.add("tpm", 5)


async def test_concurrency_limiter_caps_and_repairs_negative_counts() -> None:
    redis = CounterRedis()
    limiter = ConcurrencyLimiter(redis, ttl_seconds=10)
    assert await limiter.acquire("par", 1)
    assert not await limiter.acquire("par", 1)
    assert redis.counters["par"] == 1
    await limiter.release("par")
    await limiter.release("par")
    assert redis.counters["par"] == 0

    broken = ConcurrencyLimiter(ExplodingRedis())
    assert await broken.acquire("par", 1)
    await broken.release("par")


# -- Token revocation -------------------------------------------------------


async def test_revocation_requires_redis_and_claims() -> None:
    offline = TokenRevocation(None)
    await offline.revoke({"jti": "a", "exp": time.time() + 60})
    await offline.revoke_user("u")
    assert not await offline.is_revoked({"jti": "a"})

    redis = CounterRedis()
    revocation = TokenRevocation(redis)
    await revocation.revoke({"jti": None, "exp": 1})
    await revocation.revoke({"jti": "a", "exp": "soon"})
    assert redis.values == {}


async def test_revocation_by_token_and_by_user() -> None:
    redis = CounterRedis()
    revocation = TokenRevocation(redis, user_ttl_seconds=60)
    now = time.time()
    await revocation.revoke({"jti": "old", "exp": now + 60})
    assert await revocation.is_revoked({"jti": "old", "sub": "u"})
    assert not await revocation.is_revoked({"jti": "fresh", "sub": "u"})

    await revocation.revoke_user("u")
    assert await revocation.is_revoked({"jti": "x", "sub": "u", "iat": int(now) - 10})
    future_ms = int((now + 10) * 1000)
    assert not await revocation.is_revoked({"jti": "x", "sub": "u", "iat_ms": future_ms})


async def test_revocation_fails_open() -> None:
    revocation = TokenRevocation(ExplodingRedis())
    await revocation.revoke({"jti": "a", "exp": time.time() + 60})
    await revocation.revoke_user("u")
    assert not await revocation.is_revoked({"jti": "a", "sub": "u"})


# -- AuthStage --------------------------------------------------------------


def test_tpm_counter_names() -> None:
    assert key_tpm_counter("k") == "aigw:tpm:key:k"
    assert deployment_tpm_counter("d") == "aigw:tpm:dep:d"


def _ctx(db: Database, redis: Any, api_key: Any, model: str = "m", **request: Any) -> Any:
    chat = ChatRequest(
        model=model,
        messages=[Message(role=Role.USER, content="hello there")],
        metadata={} if api_key is None else {"api_key": api_key},
        user="end-user",
        **request,
    )
    state = SimpleNamespace(db=db, redis=redis, settings=db.settings)  # type: ignore[attr-defined]
    return SimpleNamespace(request=chat, state=state, route="/v1/chat/completions", cleanups=[])


async def test_auth_stage_requires_a_key(db: Database) -> None:
    with pytest.raises(AuthenticationError):
        await AuthStage().process(_ctx(db, None, None))
    with pytest.raises(AuthenticationError):
        await AuthStage().process(_ctx(db, None, ""))
    with pytest.raises(AuthenticationError):
        await AuthStage().process(_ctx(db, None, "sk-aigw-unknown"))


async def test_auth_stage_enforces_model_and_route_permissions(db: Database) -> None:
    service = KeyService(db, None)
    row, raw = await service.create_key("scoped", allowed_models=["allowed"])
    with pytest.raises(PermissionDeniedError, match="model"):
        await AuthStage().process(_ctx(db, None, raw, model="other"))
    await service.update_key(row.id, allowed_routes=["/v1/embeddings"])
    with pytest.raises(PermissionDeniedError, match="/v1/chat/completions"):
        await AuthStage().process(_ctx(db, None, raw, model="allowed"))


async def test_auth_stage_rpm_limit(db: Database) -> None:
    redis = CounterRedis()
    _, raw = await KeyService(db, None).create_key("rpm", rpm_limit=1)
    await AuthStage().process(_ctx(db, redis, raw))
    with pytest.raises(RateLimitExceededError, match="request rate"):
        await AuthStage().process(_ctx(db, redis, raw))


async def test_auth_stage_tpm_limit(db: Database) -> None:
    redis = CounterRedis()
    row, raw = await KeyService(db, None).create_key("tpm", tpm_limit=10_000)
    await AuthStage().process(_ctx(db, redis, raw))
    await TokenWindow(redis).add(key_tpm_counter(row.id), 10_000)
    with pytest.raises(RateLimitExceededError, match="token rate"):
        await AuthStage().process(_ctx(db, redis, raw))


async def test_auth_stage_concurrency_and_policy(db: Database) -> None:
    redis = CounterRedis()
    service = KeyService(db, None)
    row, raw = await service.create_key("parallel")
    await service.update_key(row.id, max_parallel_requests=1, guardrail_policy="strict")

    first = _ctx(db, redis, raw)
    await AuthStage().process(first)
    assert first.key_id == row.id
    assert first.end_user == "end-user"
    assert first.request.guardrail_policy == "strict"
    assert len(first.cleanups) == 1

    with pytest.raises(RateLimitExceededError, match="in flight"):
        await AuthStage().process(_ctx(db, redis, raw))
    await first.cleanups[0]()

    explicit = _ctx(db, redis, raw, guardrail_policy="lenient")
    await AuthStage().process(explicit)
    assert explicit.request.guardrail_policy == "lenient"


# -- Console auth -----------------------------------------------------------


async def test_create_user_rejects_unknown_roles(db: Database) -> None:
    with pytest.raises(ValueError, match="Role"):
        await ConsoleAuthService(db).create_user("a@b.c", "pw", "owner")


async def test_inactive_users_cannot_log_in(db: Database) -> None:
    auth = ConsoleAuthService(db)
    user = await auth.create_user("off@example.com", "pw", "viewer", full_name="Off")
    async with db.session() as session:
        (await session.get(type(user), user.id)).is_active = False  # type: ignore[union-attr]
    with pytest.raises(AuthenticationError):
        await auth.authenticate("off@example.com", "pw")
    with pytest.raises(AuthenticationError):
        await auth.authenticate("nobody@example.com", "pw")


async def test_bootstrap_admin_is_created_once(
    db: Database, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(console_module, "get_settings", lambda: db.settings)  # type: ignore[attr-defined]
    auth = ConsoleAuthService(db)
    await auth.ensure_bootstrap_admin()
    assert "default password" in caplog.text
    await auth.ensure_bootstrap_admin()
    admin = await auth.authenticate("boot@example.com", "admin")
    assert admin.role == "admin"


async def test_require_console_user_validates_scheme_and_claims(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(console_module, "get_settings", lambda: db.settings)  # type: ignore[attr-defined]
    with pytest.raises(AuthenticationError):
        await require_console_user(None)
    with pytest.raises(AuthenticationError):
        await require_console_user(HTTPAuthorizationCredentials(scheme="Basic", credentials="x"))

    token = create_access_token("u1", "u@example.com", "viewer")
    principal = await require_console_user(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    )
    assert principal == ConsolePrincipal(id="u1", email="u@example.com", role="viewer")

    import jwt

    settings = db.settings  # type: ignore[attr-defined]
    missing = jwt.encode(
        {"sub": "u1", "type": "access", "exp": int(time.time()) + 60},
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(AuthenticationError, match="claims"):
        await require_console_user(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=missing)
        )


async def test_require_admin_role() -> None:
    admin = ConsolePrincipal(id="1", email="a@b.c", role="admin")
    assert await require_admin_role(admin) is admin
    with pytest.raises(HTTPException) as raised:
        await require_admin_role(ConsolePrincipal(id="2", email="v@b.c", role="viewer"))
    assert raised.value.status_code == 403


# -- Database ---------------------------------------------------------------


async def test_database_requires_startup_and_reports_health() -> None:
    database = Database(Settings(database_url="sqlite+aiosqlite:///:memory:"))
    with pytest.raises(RuntimeError):
        await database.create_all()
    with pytest.raises(RuntimeError):
        _ = database.engine
    with pytest.raises(RuntimeError):
        async with database.session():
            pass  # pragma: no cover - never entered
    assert not await database.healthy()
    await database.shutdown()

    await database.startup()
    assert database.engine is not None
    assert await database.healthy()
    with pytest.raises(ValueError):
        async with database.session():
            raise ValueError("rolled back")
    await database.shutdown()


async def test_bootstrap_admin_with_strong_password_does_not_warn(
    db: Database, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = db.settings.model_copy(  # type: ignore[attr-defined]
        update={"bootstrap_admin_password": SecretStr("a-strong-password")}
    )
    monkeypatch.setattr(console_module, "get_settings", lambda: settings)
    await ConsoleAuthService(db).ensure_bootstrap_admin()
    assert "default password" not in caplog.text
