"""mcp server timeout

Per-server request timeout for MCP servers; null keeps MCP_TIMEOUT_SECONDS.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-30 12:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("mcp_servers", sa.Column("timeout_seconds", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("mcp_servers", "timeout_seconds")
