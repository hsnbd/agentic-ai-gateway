import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '../client';
import type { ApiPage, CacheEntry, CacheStats, DashboardSummary, DeploymentHealth, KeyBudget, TimeSeriesResponse, UsagePage } from '../../features/observability/types';

export function useUsagePage(groupBy: UsagePage['group_by'], window: string, limit: number, offset: number) {
  const params = new URLSearchParams({ group_by: groupBy, window, limit: String(limit), offset: String(offset) });
  return useQuery({ queryKey: ['usage', groupBy, window, limit, offset],
    queryFn: () => apiRequest<UsagePage>(`/admin/api/usage?${params.toString()}`),
    staleTime: 60_000 });
}

export function useKeyBudgets() {
  const params = new URLSearchParams({ limit: '200', offset: '0' });
  return useQuery({ queryKey: ['usage', 'key-budgets'], queryFn: () => apiRequest<ApiPage<KeyBudget>>(`/admin/api/keys?${params.toString()}`), staleTime: 30_000 });
}

export function useCacheStats() {
  return useQuery({ queryKey: ['cache', 'stats'], queryFn: () => apiRequest<CacheStats>('/admin/api/cache/stats'), staleTime: 30_000, refetchInterval: 60_000 });
}

export function useDeployments() {
  const params = new URLSearchParams({ limit: '200', offset: '0' });
  return useQuery({ queryKey: ['dashboard', 'deployments'], queryFn: () => apiRequest<ApiPage<DeploymentHealth>>(`/admin/api/deployments?${params.toString()}`), staleTime: 30_000 });
}

export function useMetricSeries(window: string, metric: 'requests' | 'successes' | 'errors' | 'tokens' | 'cost' | 'cache_hits') {
  const interval = window === '30d' ? 'day' : 'hour';
  const params = new URLSearchParams({ window, metric, interval });
  return useQuery({ queryKey: ['dashboard', 'timeseries', window, metric, interval],
    queryFn: () => apiRequest<TimeSeriesResponse>(`/admin/api/dashboard/timeseries?${params.toString()}`), staleTime: 30_000 });
}

export function useDashboardOverview(window: string) {
  const params = new URLSearchParams({ window });
  return useQuery({ queryKey: ['dashboard', 'summary', window],
    queryFn: () => apiRequest<DashboardSummary>(`/admin/api/dashboard/summary?${params.toString()}`), staleTime: 30_000 });
}

export function useCacheEntries(limit: number, offset: number) {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return useQuery({ queryKey: ['cache', 'entries', limit, offset],
    queryFn: () => apiRequest<ApiPage<CacheEntry>>(`/admin/api/cache/entries?${params.toString()}`),
    staleTime: 30_000 });
}
