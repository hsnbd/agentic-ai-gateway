export interface LogRow {
  id: string;
  request_id: string;
  created_at: string;
  virtual_key_id: string | null;
  team_id: string | null;
  model: string;
  resolved_model: string | null;
  provider: string | null;
  deployment_id: string | null;
  status: string;
  status_code: number | null;
  error_code: string | null;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost_usd: number;
  latency_ms: number | null;
  attempt_count: number;
  fallback_count: number;
  cache_similarity: number | null;
  stage_timings: Record<string, unknown>;
  routing_reason: string | null;
  cache_hit: boolean;
  guardrail_flagged: boolean;
  stream: boolean;
}

export interface RequestAttempt {
  deployment_id: string;
  provider: string | null;
  outcome: 'success' | 'error' | 'unknown';
  latency_ms: number | null;
  error: string | null;
}

export interface LogDetail extends LogRow {
  routing_strategy: string | null;
  guardrail_results: Record<string, unknown>;
  attempts: RequestAttempt[];
  body_redacted: boolean;
  request_body: Record<string, unknown> | null;
  response_body: Record<string, unknown> | null;
}

export interface ApiPage<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface UsageRow {
  group: string;
  requests: number;
  success_count: number;
  error_count: number;
  cache_hit_count: number;
  total_tokens: number;
  cost_usd: number;
}

export interface UsagePage {
  group_by: 'model' | 'provider' | 'key' | 'team' | 'day' | 'hour';
  rows: UsageRow[];
  total: number;
  limit: number;
  offset: number;
}

export interface CacheStats {
  enabled: boolean;
  available: boolean;
  hits: number | null;
  misses: number | null;
  entries: number | null;
  similarity_threshold: number;
  index_size_bytes: number | null;
  estimated_cost_saved_usd: number | null;
  estimated_latency_saved_ms: number | null;
}

export interface CacheEntry {
  key: string;
  model: string | null;
  namespace: string | null;
  hit_count: number | null;
  age_seconds: number | null;
  ttl_remaining_seconds: number | null;
  cached_prompt: string | null;
}

export interface DeploymentHealth {
  id: string;
  model: string;
  provider: string;
  enabled: boolean;
  health_state: string;
  consecutive_failures: number;
  failure_rate: number;
  ewma_latency_ms: number;
}

export interface KeyBudget {
  id: string;
  name: string;
  key_prefix: string;
  team_id: string | null;
  max_budget_usd: number | null;
  spend_usd: number;
  budget_duration: string;
}

export interface TimeSeriesPoint {
  timestamp: string;
  value: number;
}

export interface TimeSeriesResponse {
  metric: string;
  interval: 'hour' | 'day';
  points: TimeSeriesPoint[];
}

export interface DashboardSummary {
  requests: number;
  success_rate: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  total_cost_usd: number;
  total_tokens: number;
  cache_hit_ratio: number;
  fallback_count: number;
  active_requests: number;
  previous_requests: number;
  previous_success_rate: number;
  previous_p50_latency_ms: number;
  previous_p95_latency_ms: number;
  previous_total_cost_usd: number;
  previous_total_tokens: number;
  previous_cache_hit_ratio: number;
  previous_fallback_count: number;
}
