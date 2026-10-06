export interface Page<T> { items: T[]; total: number; limit: number; offset: number }
export interface ProviderStatus { provider: string; configured: boolean; reachable: boolean; health_state: string }
export interface AdminUser { id: string; email: string; full_name: string | null; role: 'admin' | 'viewer'; is_active: boolean; created_at: string }
export interface LoginResponse { access_token: string; token_type: 'bearer'; expires_in: number; user: AdminUser; refresh_token?: string | null; refresh_expires_in?: number | null }
export interface DashboardSummary {
  requests: number; success_rate: number; p50_latency_ms: number; p95_latency_ms: number;
  p99_latency_ms: number; total_cost_usd: number; total_tokens: number;
  cache_hit_ratio: number; active_models: number; error_count: number;
}
export interface TimeSeriesPoint { timestamp: string; value: number }
export interface TimeSeriesResponse { metric: string; interval: 'hour' | 'day'; points: TimeSeriesPoint[] }
export interface VirtualKey {
  id: string; key_prefix: string; name: string; team_id: string | null; max_budget_usd: number | null;
  spend_usd: number; budget_duration: string; rpm_limit: number | null; tpm_limit: number | null;
  max_parallel_requests: number | null; allowed_models: string[]; blocked_models: string[];
  guardrail_policy: string | null; allowed_routes: string[]; allowed_mcp_servers: string[]; allowed_tools: string[]; enabled: boolean; expires_at: string | null;
  last_used_at: string | null; metadata: Record<string, unknown>; created_at: string; updated_at: string; key?: string | null;
}
export interface Model { name: string; capabilities: Record<string, unknown>; pricing: Record<string, number | null>; deployments: string[] }
export interface RequestLog {
  id: string; request_id: string; created_at: string; virtual_key_id: string | null; team_id: string | null;
  model: string; resolved_model: string | null; provider: string | null; deployment_id: string | null;
  status: string; status_code: number | null; error_code: string | null; prompt_tokens: number;
  completion_tokens: number; total_tokens: number; cost_usd: number; latency_ms: number | null;
  stage_timings: Record<string, unknown>; routing_reason: string | null; cache_hit: boolean;
  guardrail_flagged: boolean; stream: boolean;
}
export interface UsageRow { group: string; requests: number; success_count: number; error_count: number; cache_hit_count: number; total_tokens: number; cost_usd: number }
export interface UsageResponse { group_by: 'model' | 'provider' | 'key' | 'team'; rows: UsageRow[] }
export interface HealthResponse { status: string }
export interface ApiErrorEnvelope { error: { message: string; type: string; code: string } }

export interface KeyCreateRequest {
  name: string; team_id?: string | null; max_budget_usd?: number | null; budget_duration?: string;
  rpm_limit?: number | null; tpm_limit?: number | null; max_parallel_requests?: number | null;
  allowed_models?: string[]; blocked_models?: string[]; guardrail_policy?: string | null;
  allowed_routes?: string[]; allowed_mcp_servers?: string[]; allowed_tools?: string[]; expires_at?: string | null; enabled?: boolean; metadata?: Record<string, unknown>;
}
export interface Team {
  id: string; name: string; description: string | null; max_budget_usd: number | null; spend_usd: number;
  budget_period: string; budget_reset_at: string | null; metadata: Record<string, unknown>; created_at: string; updated_at: string;
}
export interface TeamUsage {
  team_id: string; requests: number; success_count: number; error_count: number; total_tokens: number; cost_usd: number;
}
export interface Deployment {
  id: string; model: string; provider: string; provider_model: string; enabled: boolean;
  capabilities: Record<string, unknown>; pricing: Record<string, number | null>; health_state: string;
  consecutive_failures: number; failure_rate: number; ewma_latency_ms: number;
  priority: number; weight: number; tags: string[];
}
export interface RequestLogDetail extends RequestLog {
  routing_strategy: string | null; guardrail_results: Record<string, unknown>;
  request_body: Record<string, unknown> | null; response_body: Record<string, unknown> | null;
}
export interface CostProjection {
  period_start: string; period_end: string; spent_usd: number; projected_total_usd: number; budget_usd: number | null;
}
export interface CostBreakdown { total_cost_usd: number; by_model: UsageRow[]; projection: CostProjection }
export interface GuardrailViolation {
  id: string; created_at: string; request_id: string; virtual_key_id: string | null; policy: string;
  rule: string; phase: string; action: string; severity: string; match_count: number;
  excerpt: string | null; details: Record<string, unknown>;
}
export interface GuardrailPolicy { name: string; enabled: boolean; rules: string[] }
export interface CacheStats { enabled: boolean; available: boolean; hits: number | null; misses: number | null; entries: number | null }
export interface SystemInfo {
  version: string; uptime_seconds: number; providers: string[]; features: Record<string, boolean>;
  database_connected: boolean; redis_connected: boolean;
}
export interface ErrorEnvelope { error: { message: string; type: string; code: string } }

export type KeyUpdateRequest = Partial<KeyCreateRequest>
export interface TeamCreateRequest { name: string; description?: string | null; max_budget_usd?: number | null; budget_period?: string; metadata?: Record<string, unknown> }
export type TeamUpdateRequest = Partial<TeamCreateRequest>
export interface LogoutResponse { success: boolean }
export interface HealthCheckResponse { deployment_id: string; healthy: boolean }
export interface ConfigReloadResponse { models: string[]; deployment_count: number }
export interface CacheInvalidateRequest { key?: string | null; all_entries?: boolean }
export interface CacheInvalidateResponse { invalidated: number }
export interface PlaygroundRequest { model: string; messages: Record<string, unknown>[]; stream?: boolean; max_tokens?: number | null; temperature?: number | null }
export interface PlaygroundResponse {
  response: Record<string, unknown>; routing_decision: Record<string, unknown> | null;
  stage_timings: Record<string, number>; cost_usd: number; cache_hit: boolean;
}
