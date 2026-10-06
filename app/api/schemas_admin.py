"""Typed request and response contracts for the administration API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.schemas import ToolChoice, ToolDef, Usage


class Page[T](BaseModel):
    items: list[T]
    total: int
    limit: int
    offset: int


class AdminUserResponse(BaseModel):
    id: str
    email: str
    full_name: str | None
    role: Literal["admin", "viewer"]
    is_active: bool
    created_at: datetime


class AdminUserCreateRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=12, max_length=256)
    role: Literal["admin", "viewer"]


class AdminUserUpdateRequest(BaseModel):
    role: Literal["admin", "viewer"] | None = None
    is_active: bool | None = None


class AdminUserPasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=12, max_length=256)


class ProviderStatusResponse(BaseModel):
    provider: str
    configured: bool
    reachable: bool
    health_state: str


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: AdminUserResponse
    #: Exchange at /auth/refresh for a new pair; each refresh token works once.
    refresh_token: str | None = None
    refresh_expires_in: int | None = None


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    #: Also revoke this refresh token, so the session cannot be renewed.
    refresh_token: str | None = None


class LogoutResponse(BaseModel):
    success: bool = True


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=12, max_length=256)


class DashboardSummary(BaseModel):
    requests: int
    success_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_cost_usd: float
    total_tokens: int
    cache_hit_ratio: float
    active_models: int
    error_count: int
    fallback_count: int
    active_requests: float
    previous_requests: int
    previous_success_rate: float
    previous_p50_latency_ms: float
    previous_p95_latency_ms: float
    previous_p99_latency_ms: float
    previous_total_cost_usd: float
    previous_total_tokens: int
    previous_cache_hit_ratio: float
    previous_active_models: int
    previous_error_count: int
    previous_fallback_count: int


class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    value: float


class TimeSeriesResponse(BaseModel):
    metric: str
    interval: str
    points: list[TimeSeriesPoint]


class KeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    team_id: str | None = None
    max_budget_usd: float | None = Field(default=None, ge=0)
    budget_duration: str = "monthly"
    rpm_limit: int | None = Field(default=None, gt=0)
    tpm_limit: int | None = Field(default=None, gt=0)
    max_parallel_requests: int | None = Field(default=None, gt=0)
    allowed_models: list[str] = Field(default_factory=list)
    blocked_models: list[str] = Field(default_factory=list)
    guardrail_policy: str | None = None
    allowed_routes: list[str] = Field(default_factory=list)
    allowed_mcp_servers: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    expires_at: datetime | None = None
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class KeyUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    team_id: str | None = None
    max_budget_usd: float | None = Field(default=None, ge=0)
    budget_duration: str | None = None
    rpm_limit: int | None = Field(default=None, gt=0)
    tpm_limit: int | None = Field(default=None, gt=0)
    max_parallel_requests: int | None = Field(default=None, gt=0)
    allowed_models: list[str] | None = None
    blocked_models: list[str] | None = None
    guardrail_policy: str | None = None
    allowed_routes: list[str] | None = None
    allowed_mcp_servers: list[str] | None = None
    allowed_tools: list[str] | None = None
    expires_at: datetime | None = None
    enabled: bool | None = None
    metadata: dict[str, Any] | None = None


class VirtualKeyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    key_prefix: str
    name: str
    team_id: str | None
    max_budget_usd: float | None
    spend_usd: float
    budget_duration: str = Field(validation_alias="budget_period")
    rpm_limit: int | None
    tpm_limit: int | None
    max_parallel_requests: int | None
    allowed_models: list[str]
    blocked_models: list[str]
    guardrail_policy: str | None
    allowed_routes: list[str]
    allowed_mcp_servers: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    enabled: bool = Field(validation_alias="is_active")
    expires_at: datetime | None
    last_used_at: datetime | None
    metadata: dict[str, Any] = Field(validation_alias="metadata_")
    created_at: datetime
    updated_at: datetime
    key: str | None = Field(
        default=None,
        description="Shown only at creation; the regenerate endpoint also returns it once.",
    )


class TeamCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    max_budget_usd: float | None = Field(default=None, ge=0)
    budget_period: str = "monthly"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TeamUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    max_budget_usd: float | None = Field(default=None, ge=0)
    budget_period: str | None = None
    metadata: dict[str, Any] | None = None


class TeamResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    name: str
    description: str | None
    max_budget_usd: float | None
    spend_usd: float
    budget_period: str
    budget_reset_at: datetime | None
    metadata: dict[str, Any] = Field(validation_alias="metadata_")
    created_at: datetime
    updated_at: datetime


class TeamUsageResponse(BaseModel):
    team_id: str
    requests: int
    success_count: int
    error_count: int
    total_tokens: int
    cost_usd: float


class DeploymentResponse(BaseModel):
    id: str
    model: str
    provider: str
    provider_model: str
    enabled: bool
    capabilities: dict[str, Any]
    pricing: dict[str, float | None]
    health_state: str
    consecutive_failures: int
    failure_rate: float
    ewma_latency_ms: float
    priority: int = Field(description="Lower is preferred when routing by priority.")
    weight: int
    tags: list[str]


class ModelResponse(BaseModel):
    name: str
    capabilities: dict[str, Any]
    pricing: dict[str, float | None]
    deployments: list[str]


class HealthCheckResponse(BaseModel):
    deployment_id: str
    healthy: bool


class ConfigReloadResponse(BaseModel):
    models: list[str]
    deployment_count: int


class RequestLogResponse(BaseModel):
    id: str
    request_id: str
    created_at: datetime
    virtual_key_id: str | None
    team_id: str | None
    model: str
    resolved_model: str | None
    provider: str | None
    deployment_id: str | None
    status: str
    status_code: int | None
    error_code: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float | None
    attempt_count: int
    fallback_count: int
    cache_similarity: float | None
    stage_timings: dict[str, Any]
    routing_reason: str | None
    cache_hit: bool
    guardrail_flagged: bool
    stream: bool
    #: Tool calls the gateway executed (agent mode) or the model requested.
    tool_calls_count: int = 0


class RequestLogDetailResponse(RequestLogResponse):
    routing_strategy: str | None
    guardrail_results: dict[str, Any]
    request_body: dict[str, Any] | None
    response_body: dict[str, Any] | None
    body_redacted: bool = Field(
        description="True when stored request/response bodies were withheld by access policy."
    )
    attempts: list[RequestAttemptResponse]


class RequestAttemptResponse(BaseModel):
    deployment_id: str
    provider: str | None
    outcome: Literal["success", "error", "unknown"]
    latency_ms: float | None = Field(
        description="Unavailable for historical rows created before per-attempt instrumentation."
    )
    error: str | None


class UsageRow(BaseModel):
    group: str
    requests: int
    success_count: int
    error_count: int
    cache_hit_count: int
    total_tokens: int
    cost_usd: float


class UsageResponse(BaseModel):
    group_by: Literal["model", "provider", "key", "team", "day", "hour"]
    rows: list[UsageRow]
    total: int
    limit: int
    offset: int


class CostProjectionResponse(BaseModel):
    period_start: datetime
    period_end: datetime
    spent_usd: float
    projected_total_usd: float
    budget_usd: float | None


class CostBreakdownResponse(BaseModel):
    total_cost_usd: float
    by_model: list[UsageRow]
    projection: CostProjectionResponse


class GuardrailViolationResponse(BaseModel):
    id: str
    created_at: datetime
    request_id: str
    virtual_key_id: str | None
    policy: str
    rule: str
    phase: str
    action: str
    severity: str
    match_count: int
    excerpt: str | None
    details: dict[str, Any]


class ToolCallLogResponse(BaseModel):
    id: str
    created_at: datetime
    request_id: str | None
    virtual_key_id: str | None
    team_id: str | None
    source: str
    server_id: str | None
    tool: str
    status: str
    duration_ms: float
    arguments_hash: str | None
    result_chars: int
    truncated: bool
    guardrail: str | None
    error: str | None


class GuardrailPolicyResponse(BaseModel):
    name: str
    enabled: bool
    rules: list[str]


class CacheStatsResponse(BaseModel):
    enabled: bool
    available: bool
    hits: int | None = None
    misses: int | None = None
    entries: int | None = None
    similarity_threshold: float
    index_size_bytes: int | None = Field(
        default=None, description="Unavailable when Redis Search does not expose index info."
    )
    estimated_cost_saved_usd: float | None = Field(
        default=None, description="Unavailable when cache savings are not recorded."
    )
    estimated_latency_saved_ms: float | None = Field(
        default=None,
        description="Provider time avoided by cache hits (original latency minus lookup time).",
    )


class CacheEntryResponse(BaseModel):
    key: str
    model: str | None
    namespace: str | None
    hit_count: int | None = Field(
        description="Unavailable because the cache does not record per-entry hit counts."
    )
    age_seconds: float | None
    ttl_remaining_seconds: int | None
    cached_prompt: str | None = Field(
        description="Redacted cached prompt, or null when prompts are not stored."
    )


class CacheInvalidateRequest(BaseModel):
    key: str | None = None
    namespace: str | None = None
    all_entries: bool = False


class CacheInvalidateResponse(BaseModel):
    invalidated: int


class SystemInfoResponse(BaseModel):
    version: str
    uptime_seconds: float
    providers: list[str]
    features: dict[str, bool]
    database_connected: bool
    redis_connected: bool
    active_routing_strategy: str


class PlaygroundRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stop: list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    n: int = Field(default=1, ge=1)
    tools: list[ToolDef] = Field(default_factory=list)
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    user: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    no_cache: bool = False
    cache_ttl: int | None = Field(default=None, gt=0)
    fallbacks: list[str] = Field(default_factory=list)
    routing_strategy: str | None = None
    guardrail_policy: str | None = None
    tags: list[str] = Field(default_factory=list)


class PlaygroundResponse(BaseModel):
    response: dict[str, Any]
    routing_decision: dict[str, Any] | None
    request_id: str
    deployment_id: str | None
    provider: str | None
    stage_timings: dict[str, float]
    latency_ms: float | None
    time_to_first_token_ms: float | None
    cache_hit: bool
    cache_similarity: float | None
    retry_count: int
    fallback_count: int
    fallback_used: bool
    guardrail_flagged: bool
    guardrail_results: dict[str, Any]
    token_usage: Usage
    estimated_cost_usd: float | None


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


class CSVExportResponse(BaseModel):
    """OpenAPI metadata for the streamed CSV export."""

    content_type: Literal["text/csv"] = "text/csv"
