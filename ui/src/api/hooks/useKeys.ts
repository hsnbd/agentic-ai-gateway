import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '../client';
import type { Page, VirtualKey } from '../types';
export const keyKeys = { all: ['keys'] as const, list: (limit: number, offset: number) => ['keys', limit, offset] as const };
export function useKeys(limit = 50, offset = 0) {
  return useQuery({ queryKey: keyKeys.list(limit, offset), queryFn: () => apiRequest<Page<VirtualKey>>(`/admin/api/keys?limit=${limit}&offset=${offset}`), staleTime: 30_000 });
}
