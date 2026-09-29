import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '../client';
import type { UsageResponse } from '../types';
export const usageKeys = { all: ['usage'] as const, list: (group: string, window: string) => ['usage', group, window] as const };
export function useUsage(groupBy: UsageResponse['group_by'] = 'model', window = '30d') {
  const params = new URLSearchParams({ group_by: groupBy, window });
  return useQuery({ queryKey: usageKeys.list(groupBy, window), queryFn: () => apiRequest<UsageResponse>(`/admin/api/usage?${params}`), staleTime: 60_000 });
}
