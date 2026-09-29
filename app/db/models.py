"""SQLAlchemy ORM models.

Covers virtual keys and budgets, admin console users, request logs and usage
rollups, RAG collections/documents/chunks, and the MCP server registry.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

#: JSONB on Postgres, plain JSON elsewhere (keeps SQLite usable in tests).
JSONType = JSON().with_variant(JSONB(), "postgresql")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


# --------------------------------------------------------------------------
# Tenancy, keys, budgets
# --------------------------------------------------------------------------


class Team(Base, TimestampMixin):
    __tablename__ = "teams"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    max_budget_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    spend_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), default=Decimal("0"))
    budget_period: Mapped[str] = mapped_column(String(20), default="monthly")
    budget_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)

    keys: Mapped[list[VirtualKey]] = relationship(back_populates="team")


class VirtualKey(Base, TimestampMixin):
    """A client-facing API key. The raw secret is never stored."""

    __tablename__ = "virtual_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    #: SHA-256 of the raw key; the lookup index.
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    #: Display-only prefix, e.g. "sk-aigw-a1b2...".
    key_prefix: Mapped[str] = mapped_column(String(24), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    team_id: Mapped[str | None] = mapped_column(ForeignKey("teams.id", ondelete="SET NULL"))
    team: Mapped[Team | None] = relationship(back_populates="keys")

    # Budget
    max_budget_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    spend_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), default=Decimal("0"))
    budget_period: Mapped[str] = mapped_column(String(20), default="monthly")
    budget_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Limits
    rpm_limit: Mapped[int | None] = mapped_column(Integer)
    tpm_limit: Mapped[int | None] = mapped_column(Integer)
    max_parallel_requests: Mapped[int | None] = mapped_column(Integer)

    # Policy
    allowed_models: Mapped[list[str]] = mapped_column(JSONType, default=list)
    blocked_models: Mapped[list[str]] = mapped_column(JSONType, default=list)
    guardrail_policy: Mapped[str | None] = mapped_column(String(100))
    allowed_routes: Mapped[list[str]] = mapped_column(JSONType, default=list)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)

    def is_valid(self, now: datetime | None = None) -> bool:
        now = now or _utcnow()
        if not self.is_active:
            return False
        return not (self.expires_at and self.expires_at <= now)

    def permits_model(self, model: str) -> bool:
        if model in self.blocked_models:
            return False
        return not self.allowed_models or model in self.allowed_models


# --------------------------------------------------------------------------
# Console users
# --------------------------------------------------------------------------


class AdminUser(Base, TimestampMixin):
    __tablename__ = "admin_users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    #: "admin" (read/write) or "viewer" (read-only).
    role: Mapped[str] = mapped_column(String(20), default="viewer", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# --------------------------------------------------------------------------
# Request logging and usage
# --------------------------------------------------------------------------


class RequestLog(Base):
    """One row per gateway request, powering the console log explorer."""

    __tablename__ = "request_logs"
    __table_args__ = (
        Index("ix_request_logs_created_key", "created_at", "virtual_key_id"),
        Index("ix_request_logs_created_model", "created_at", "model"),
        Index("ix_request_logs_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    request_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    virtual_key_id: Mapped[str | None] = mapped_column(String(36), index=True)
    team_id: Mapped[str | None] = mapped_column(String(36), index=True)
    end_user: Mapped[str | None] = mapped_column(String(255))

    # What was asked for vs. what actually served it
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    resolved_model: Mapped[str | None] = mapped_column(String(255))
    provider: Mapped[str | None] = mapped_column(String(64), index=True)
    deployment_id: Mapped[str | None] = mapped_column(String(255))
    route: Mapped[str | None] = mapped_column(String(128))
    dialect: Mapped[str | None] = mapped_column(String(32))

    status: Mapped[str] = mapped_column(String(20), nullable=False)
    status_code: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)

    # Tokens and cost
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # Behaviour
    latency_ms: Mapped[float | None] = mapped_column(Float)
    time_to_first_token_ms: Mapped[float | None] = mapped_column(Float)
    stage_timings: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    attempt_count: Mapped[int] = mapped_column(Integer, default=1)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    routing_strategy: Mapped[str | None] = mapped_column(String(64))
    routing_reason: Mapped[str | None] = mapped_column(Text)

    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    cache_similarity: Mapped[float | None] = mapped_column(Float)

    guardrail_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    guardrail_results: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    stream: Mapped[bool] = mapped_column(Boolean, default=False)
    tool_calls_count: Mapped[int] = mapped_column(Integer, default=0)
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True)
    tags: Mapped[list[str]] = mapped_column(JSONType, default=list)

    #: Only persisted when log_request_bodies is enabled; redaction applies.
    request_body: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    response_body: Mapped[dict[str, Any] | None] = mapped_column(JSONType)


class UsageRollup(Base):
    """Pre-aggregated hourly usage, so console charts never scan raw logs."""

    __tablename__ = "usage_rollups"
    __table_args__ = (
        UniqueConstraint(
            "bucket", "virtual_key_id", "model", "provider", name="uq_usage_rollup_bucket"
        ),
        Index("ix_usage_rollups_bucket", "bucket"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    bucket: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    virtual_key_id: Mapped[str | None] = mapped_column(String(36), index=True)
    team_id: Mapped[str | None] = mapped_column(String(36), index=True)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)

    request_count: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    cache_hit_count: Mapped[int] = mapped_column(Integer, default=0)
    fallback_count: Mapped[int] = mapped_column(Integer, default=0)

    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    #: Estimated spend avoided by cache hits.
    cost_saved_usd: Mapped[float] = mapped_column(Float, default=0.0)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)


# --------------------------------------------------------------------------
# RAG
# --------------------------------------------------------------------------


class RagCollection(Base, TimestampMixin):
    __tablename__ = "rag_collections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)

    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, default=1000)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=150)

    document_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)

    documents: Mapped[list[RagDocument]] = relationship(
        back_populates="collection", cascade="all, delete-orphan"
    )


class RagDocument(Base, TimestampMixin):
    __tablename__ = "rag_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    collection_id: Mapped[str] = mapped_column(
        ForeignKey("rag_collections.id", ondelete="CASCADE"), index=True
    )
    collection: Mapped[RagCollection] = relationship(back_populates="documents")

    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source: Mapped[str | None] = mapped_column(String(1024))
    content_type: Mapped[str] = mapped_column(String(100), default="text/plain")
    #: SHA-256 of the raw content, used to skip re-ingesting duplicates.
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    byte_size: Mapped[int] = mapped_column(Integer, default=0)

    status: Mapped[str] = mapped_column(String(20), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)

    chunks: Mapped[list[RagChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class RagChunk(Base):
    """Chunk text and metadata. Vectors themselves live in Redis."""

    __tablename__ = "rag_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_rag_chunk_position"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("rag_documents.id", ondelete="CASCADE"), index=True
    )
    document: Mapped[RagDocument] = relationship(back_populates="chunks")
    collection_id: Mapped[str] = mapped_column(String(36), index=True)

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    #: Redis key holding this chunk's vector.
    vector_id: Mapped[str | None] = mapped_column(String(255), index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )


# --------------------------------------------------------------------------
# MCP
# --------------------------------------------------------------------------


class McpServer(Base, TimestampMixin):
    __tablename__ = "mcp_servers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)

    #: "stdio" or "http"
    transport: Mapped[str] = mapped_column(String(20), nullable=False)
    command: Mapped[str | None] = mapped_column(String(1024))
    args: Mapped[list[str]] = mapped_column(JSONType, default=list)
    env: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    url: Mapped[str | None] = mapped_column(String(1024))
    headers: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    health_status: Mapped[str] = mapped_column(String(20), default="unknown")
    last_health_check_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: Cached tool descriptors from the last successful discovery.
    discovered_tools: Mapped[list[dict[str, Any]]] = mapped_column(JSONType, default=list)
    #: Prefix applied to tool names to avoid collisions across servers.
    tool_prefix: Mapped[str | None] = mapped_column(String(64))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONType, default=dict)


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------


class GuardrailViolation(Base):
    __tablename__ = "guardrail_violations"
    __table_args__ = (Index("ix_guardrail_violations_created", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    virtual_key_id: Mapped[str | None] = mapped_column(String(36), index=True)

    policy: Mapped[str] = mapped_column(String(100), nullable=False)
    rule: Mapped[str] = mapped_column(String(255), nullable=False)
    #: "input" or "output"
    phase: Mapped[str] = mapped_column(String(20), nullable=False)
    #: "block", "redact", or "flag"
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default="medium")
    match_count: Mapped[int] = mapped_column(Integer, default=1)
    excerpt: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
