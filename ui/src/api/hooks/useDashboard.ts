import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '../client';
import type { DashboardSummary, TimeSeriesResponse } from '../types';

export const dashboardKeys = {
  all: ['dashboard'] as const,
  summary: (window: string) => ['dashboard', 'summary', window] as const,
  timeseries: (window: string, metric: string, interval: string) => ['dashboard', 'timeseries', window, metric, interval] as const,
};
export function useDashboardSummary(window = '24h') {
  return useQuery({ queryKey: dashboardKeys.summary(window), queryFn: () => apiRequest<DashboardSummary>(`/admin/api/dashboard/summary?window=${window}`), staleTime: 30_000 });
}
export function useDashboardTimeseries(window = '24h', metric = 'requests', interval = 'hour') {
  const params = new URLSearchParams({ window, metric, interval });
  return useQuery({ queryKey: dashboardKeys.timeseries(window, metric, interval), queryFn: () => apiRequest<TimeSeriesResponse>(`/admin/api/dashboard/timeseries?${params}`), staleTime: 30_000 });
}
