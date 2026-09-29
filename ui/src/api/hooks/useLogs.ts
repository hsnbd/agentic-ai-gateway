import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '../client';
import type { Page, RequestLog } from '../types';
export const logKeys = { all: ['logs'] as const, list: (limit: number, offset: number) => ['logs', limit, offset] as const };
export function useLogs(limit = 50, offset = 0) {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return useQuery({ queryKey: logKeys.list(limit, offset), queryFn: () => apiRequest<Page<RequestLog>>(`/admin/api/logs?${params}`), staleTime: 15_000 });
}
