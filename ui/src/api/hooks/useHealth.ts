import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '../client';
import type { HealthResponse } from '../types';
export const healthKeys = { all: ['health'] as const };
export function useHealth() {
  return useQuery({ queryKey: healthKeys.all, queryFn: () => apiRequest<HealthResponse>('/healthz'), staleTime: 15_000, refetchInterval: 30_000 });
}
