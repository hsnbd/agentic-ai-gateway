import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '../client';
import type { ApiPage, LogDetail, LogRow } from '../../features/observability/types';

export interface LogFilters {
  limit: number;
  offset: number;
  start?: string;
  end?: string;
  virtual_key_id?: string;
  model?: string;
  provider?: string;
  status?: string;
  cache_hit?: boolean;
  min_latency_ms?: number;
  search?: string;
}

export function paramsFromFilters(filters: LogFilters): URLSearchParams {
  const params = new URLSearchParams({ limit: String(filters.limit), offset: String(filters.offset) });
  for (const key of ['start', 'end', 'virtual_key_id', 'model', 'provider', 'status', 'search'] as const) {
    const value = filters[key];
    if (value) params.set(key, value);
  }
  if (filters.cache_hit !== undefined) params.set('cache_hit', String(filters.cache_hit));
  if (filters.min_latency_ms !== undefined) params.set('min_latency_ms', String(filters.min_latency_ms));
  return params;
}

/** The CSV export URL for the same filters (paging does not apply). */
export function logsExportPath(filters: LogFilters): string {
  const params = paramsFromFilters(filters);
  params.delete('limit'); params.delete('offset');
  return `/admin/api/logs/export?${params.toString()}`;
}

export function useFilteredLogs(filters: LogFilters) {
  const params = paramsFromFilters(filters);
  return useQuery({ queryKey: ['logs', 'filtered', params.toString()],
    queryFn: () => apiRequest<ApiPage<LogRow>>(`/admin/api/logs?${params.toString()}`),
    staleTime: 15_000 });
}

export function useLogDetail(requestId: string | null) {
  return useQuery({ queryKey: ['logs', 'detail', requestId],
    queryFn: () => apiRequest<LogDetail>(`/admin/api/logs/${encodeURIComponent(requestId ?? '')}`),
    enabled: requestId !== null });
}
