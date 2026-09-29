"""Async database engine and session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config.settings import Settings
from app.db.models import Base


class Database:
    """Owns the engine and session factory for the process."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

    async def startup(self) -> None:
        url = self._settings.database_url
        # SQLite (used by tests) rejects pool sizing arguments.
        kwargs: dict[str, object] = {"echo": self._settings.db_echo, "pool_pre_ping": True}
        if not url.startswith("sqlite"):
            kwargs["pool_size"] = self._settings.db_pool_size
            kwargs["max_overflow"] = self._settings.db_max_overflow

        self._engine = create_async_engine(url, **kwargs)  # type: ignore[arg-type]
        self._sessionmaker = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )

    async def shutdown(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None

    async def create_all(self) -> None:
        """Create any missing tables from the ORM metadata.

        `create_all` only adds what is absent; it never alters or drops an
        existing table, so it is safe to run on every startup but cannot
        perform a schema migration. Turn it off with `AUTO_CREATE_SCHEMA=false`
        once a real migration tool owns the schema.
        """
        if self._engine is None:
            raise RuntimeError("Database.startup() must be called first")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("Database.startup() must be called first")
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Transactional scope: commits on success, rolls back on error."""
        if self._sessionmaker is None:
            raise RuntimeError("Database.startup() must be called first")
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def healthy(self) -> bool:
        from sqlalchemy import text

        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
