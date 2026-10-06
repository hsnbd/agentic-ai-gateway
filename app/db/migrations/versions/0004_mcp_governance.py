"""mcp governance

Per-key MCP server and tool allowlists, and an audit log of tool calls.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-30 13:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    for column in ("allowed_mcp_servers", "allowed_tools"):
        op.add_column(
            "virtual_keys",
            sa.Column(column, _JSON, nullable=False, server_default=sa.text("'[]'")),
        )
        op.alter_column("virtual_keys", column, server_default=None)
    op.create_table(
        "tool_call_logs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("virtual_key_id", sa.String(length=36), nullable=True),
        sa.Column("team_id", sa.String(length=36), nullable=True),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("server_id", sa.String(length=36), nullable=True),
        sa.Column("tool", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("arguments_hash", sa.String(length=64), nullable=True),
        sa.Column("result_chars", sa.Integer(), nullable=False),
        sa.Column("truncated", sa.Boolean(), nullable=False),
        sa.Column("guardrail", sa.String(length=255), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_tool_call_logs_created", "tool_call_logs", ["created_at"])
    for column in ("request_id", "virtual_key_id", "team_id", "server_id", "tool", "status"):
        op.create_index(op.f(f"ix_tool_call_logs_{column}"), "tool_call_logs", [column])


def downgrade() -> None:
    for column in ("request_id", "virtual_key_id", "team_id", "server_id", "tool", "status"):
        op.drop_index(op.f(f"ix_tool_call_logs_{column}"), table_name="tool_call_logs")
    op.drop_index("ix_tool_call_logs_created", table_name="tool_call_logs")
    op.drop_table("tool_call_logs")
    op.drop_column("virtual_keys", "allowed_tools")
    op.drop_column("virtual_keys", "allowed_mcp_servers")
