"""Alembic migrations stay in lockstep with the ORM models.

If a model changes without a migration, `test_migrations_match_the_models`
fails with the exact diff Alembic would generate.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.db import migrate
from app.db.models import Base
from tests.integration.conftest import TEST_DATABASE_URL


def _admin(statement: str) -> None:
    async def run() -> None:
        engine = create_async_engine(TEST_DATABASE_URL, isolation_level="AUTOCOMMIT")
        try:
            async with engine.connect() as conn:
                await conn.execute(text(statement))
        finally:
            await engine.dispose()

    asyncio.run(run())


def _inspect(url: str, fn: Any) -> Any:
    async def run() -> Any:
        engine = create_async_engine(url)
        try:
            async with engine.connect() as conn:
                return await conn.run_sync(fn)
        finally:
            await engine.dispose()

    return asyncio.run(run())


@pytest.fixture
def empty_database() -> Iterator[str]:
    name = f"migrations_{uuid.uuid4().hex[:8]}"
    _admin(f'CREATE DATABASE "{name}"')
    try:
        yield make_url(TEST_DATABASE_URL).set(database=name).render_as_string(
            hide_password=False
        )
    finally:
        _admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def _diff(connection: Any) -> list[Any]:
    context = MigrationContext.configure(connection, opts={"compare_type": True})
    return list(compare_metadata(context, Base.metadata))


def test_migrations_match_the_models(empty_database: str) -> None:
    migrate.upgrade(url=empty_database)
    assert _inspect(empty_database, _diff) == []


def test_downgrade_and_upgrade_round_trip(empty_database: str) -> None:
    migrate.upgrade(url=empty_database)
    migrate.downgrade("base", url=empty_database)
    tables = _inspect(empty_database, lambda conn: inspect(conn).get_table_names())
    assert set(tables) <= {"alembic_version"}
    migrate.upgrade(url=empty_database)
    tables = _inspect(empty_database, lambda conn: inspect(conn).get_table_names())
    assert set(Base.metadata.tables) <= set(tables)


def test_existing_create_all_schema_can_be_stamped(empty_database: str) -> None:
    async def create_all() -> None:
        engine = create_async_engine(empty_database)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        finally:
            await engine.dispose()

    asyncio.run(create_all())
    migrate.stamp(url=empty_database)
    version = _inspect(
        empty_database,
        lambda conn: conn.execute(text("SELECT version_num FROM alembic_version")).scalar(),
    )
    assert version == "0001"
    # Upgrading a stamped database is a no-op, not a "table already exists" error.
    migrate.upgrade(url=empty_database)
