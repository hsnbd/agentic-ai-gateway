"""Schema migrations with Alembic, usable without an `alembic.ini` on disk.

The CLI (`aigateway migrate`, `aigateway db-stamp`) and the test suite both
go through here, so the container image needs only the `app` package.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def alembic_config(url: str | None = None) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    if url:
        # ConfigParser interpolation: a literal "%" (e.g. in a password) must be doubled.
        config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    config.attributes["configure_logging"] = False
    return config


def upgrade(revision: str = "head", *, url: str | None = None) -> None:
    command.upgrade(alembic_config(url), revision)


def downgrade(revision: str, *, url: str | None = None) -> None:
    command.downgrade(alembic_config(url), revision)


def stamp(revision: str = "head", *, url: str | None = None) -> None:
    """Record a revision without running it, for schemas made by `create_all`."""
    command.stamp(alembic_config(url), revision)
