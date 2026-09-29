"""Application settings loaded from environment variables and .env files."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Service ---
    app_name: str = "aigateway"
    environment: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 4000
    root_path: str = ""

    # --- Config files ---
    models_config_path: str = "config/models.yaml"
    guardrails_config_path: str = "config/guardrails.yaml"
    pricing_config_path: str = "config/pricing.yaml"

    # --- Datastores ---
    database_url: str = "postgresql+asyncpg://aigateway:aigateway@localhost:5432/aigateway"
    redis_url: str = "redis://localhost:6379/0"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False
    #: Create missing tables at startup. Additive only — it cannot migrate an
    #: existing schema, so turn it off once a migration tool owns the database.
    auto_create_schema: bool = True

    # --- Security ---
    master_key: SecretStr = SecretStr("sk-gateway-master-change-me")
    jwt_secret: SecretStr = SecretStr("change-me-in-production")
    jwt_algorithm: str = "HS256"
    jwt_access_ttl_seconds: int = 3600
    jwt_refresh_ttl_seconds: int = 604800
    bootstrap_admin_email: str = "admin@local"
    bootstrap_admin_password: SecretStr = SecretStr("admin")

    # --- Provider credentials ---
    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: SecretStr | None = None
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    gemini_api_key: SecretStr | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    ollama_base_url: str = "http://localhost:11434"

    # --- Request handling ---
    request_timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 10.0
    max_retries: int = 2
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 8.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown_seconds: float = 30.0

    # --- Routing ---
    routing_strategy: str = "priority"
    max_fallbacks: int = 3

    # --- Guardrails ---
    guardrails_enabled: bool = True

    # --- Semantic cache ---
    cache_enabled: bool = True
    cache_similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    cache_ttl_seconds: int = 3600
    cache_embedding_model: str = "nomic-embed-text"
    cache_embedding_dimensions: int = 768
    cache_index_name: str = "aigw:cache:idx"
    cache_max_temperature: float = 0.3

    # --- RAG ---
    rag_embedding_model: str = "nomic-embed-text"
    rag_embedding_dimensions: int = 768
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 150
    rag_default_top_k: int = 5
    rag_index_name: str = "aigw:rag:idx"

    # --- MCP ---
    mcp_timeout_seconds: float = 10.0
    mcp_tool_cache_ttl_seconds: float = 300.0

    # --- Observability ---
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    log_request_bodies: bool = False
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    otlp_endpoint: str | None = None

    # --- Console UI ---
    ui_enabled: bool = True
    ui_path: str = "/ui"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
