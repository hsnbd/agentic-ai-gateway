"""rag collection owners

Collections gain an owning team or virtual key, and names become unique per
owner instead of globally. Existing collections keep no owner, so they stay
global and readable by every caller, as before.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-30 11:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "rag_collections", sa.Column("owner_team_id", sa.String(length=36), nullable=True)
    )
    op.add_column("rag_collections", sa.Column("owner_key_id", sa.String(length=36), nullable=True))
    op.create_index(op.f("ix_rag_collections_owner_team_id"), "rag_collections", ["owner_team_id"])
    op.create_index(op.f("ix_rag_collections_owner_key_id"), "rag_collections", ["owner_key_id"])
    op.drop_index(op.f("ix_rag_collections_name"), table_name="rag_collections")
    op.create_index(op.f("ix_rag_collections_name"), "rag_collections", ["name"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_rag_collections_name"), table_name="rag_collections")
    op.create_index(op.f("ix_rag_collections_name"), "rag_collections", ["name"], unique=True)
    op.drop_index(op.f("ix_rag_collections_owner_key_id"), table_name="rag_collections")
    op.drop_index(op.f("ix_rag_collections_owner_team_id"), table_name="rag_collections")
    op.drop_column("rag_collections", "owner_key_id")
    op.drop_column("rag_collections", "owner_team_id")
