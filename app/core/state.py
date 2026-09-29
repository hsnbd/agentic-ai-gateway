"""Shared application state.

A single `GatewayState` instance is built during startup and attached to
`app.state.gateway`, so routes and pipeline stages resolve their dependencies
from one place rather than importing globals.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import redis.asyncio as aioredis

from app.config.settings import Settings
from app.db.session import Database
from app.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from fastapi import FastAPI

    from app.core.pipeline import Pipeline
    from app.routing.breaker import CircuitBreaker
    from app.routing.router import Router


logger = logging.getLogger(__name__)


class GatewayState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = ProviderRegistry(settings)
        self.db = Database(settings)
        self.redis: aioredis.Redis | None = None

        #: Built by `app.core.builder.build_pipeline` once providers are up.
        self.pipeline: Pipeline | None = None
        self.router: Router | None = None
        self.breaker: CircuitBreaker | None = None

        #: Populated by later subsystems (cache, guardrails, rag, mcp, ...).
        self.components: dict[str, Any] = {}

    async def startup(self) -> None:
        await self.registry.startup()
        await self.db.startup()
        if self.settings.auto_create_schema:
            await self._create_schema()
        self.redis = aioredis.from_url(
            self.settings.redis_url,
            encoding="utf-8",
            decode_responses=False,
            health_check_interval=30,
        )

        # Built last: stages resolve providers, the database, and Redis from
        # this same object, so they must already exist.
        from app.core.builder import build_pipeline

        self.pipeline = build_pipeline(self)

        await self._bootstrap_console_admin()

    async def _create_schema(self) -> None:
        """Ensure the tables exist before anything tries to use them.

        A fresh deployment otherwise starts healthy but fails every database
        call, which looks like a bug in the feature rather than a missing
        schema. `create_all` is additive, so this is a no-op once the tables
        are there. Set `AUTO_CREATE_SCHEMA=false` when a migration tool owns
        the schema — it cannot apply column changes to existing tables.
        """
        try:
            await self.db.create_all()
        except Exception:  # pragma: no cover - startup must stay observable
            logger.exception("schema creation failed; database features will not work")

    async def _bootstrap_console_admin(self) -> None:
        """Seed the first console administrator on an empty database.

        Without this the console is unreachable on a fresh deployment: there
        is no sign-up flow, and every admin endpoint that could create a user
        already requires an authenticated admin.
        """
        from app.auth.console import ConsoleAuthService

        try:
            await ConsoleAuthService(self.db).ensure_bootstrap_admin()
        except Exception:  # pragma: no cover - never block startup on this
            logger.exception("bootstrap administrator could not be created")

    def require_pipeline(self) -> Pipeline:
        """Fetch the pipeline, failing loudly if startup did not complete."""
        if self.pipeline is None:
            raise RuntimeError("Gateway pipeline is not initialised; startup did not run")
        return self.pipeline

    async def shutdown(self) -> None:
        if self.redis is not None:
            await self.redis.aclose()
            self.redis = None
        await self.db.shutdown()
        await self.registry.shutdown()

    async def redis_healthy(self) -> bool:
        if self.redis is None:
            return False
        try:
            await self.redis.ping()
            return True
        except Exception:
            return False


def get_state(app: FastAPI) -> GatewayState:
    state: GatewayState = app.state.gateway
    return state
