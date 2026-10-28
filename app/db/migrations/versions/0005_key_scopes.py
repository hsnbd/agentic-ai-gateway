"""key scopes

Per-key RAG collection allowlists, tool arguments bound to the caller's
identity, and the audit of which arguments were bound.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-25 10:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")

#: (table, column, server default for existing rows)
_COLUMNS = (
    ("virtual_keys", "allowed_collections", "'[]'"),
    ("virtual_keys", "tool_arg_bindings", "'{}'"),
    ("tool_call_logs", "bound_args", "'[]'"),
)


def upgrade() -> None:
    for table, column, default in _COLUMNS:
        op.add_column(
            table, sa.Column(column, _JSON, nullable=False, server_default=sa.text(default))
        )
        op.alter_column(table, column, server_default=None)


def downgrade() -> None:
    for table, column, _ in reversed(_COLUMNS):
        op.drop_column(table, column)
