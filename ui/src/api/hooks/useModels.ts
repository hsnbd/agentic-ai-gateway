import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '../client';
import type { Model, Page } from '../types';
export const modelKeys = { all: ['models'] as const, list: (limit: number, offset: number) => ['models', limit, offset] as const };
export function useModels(limit = 50, offset = 0) {
  return useQuery({ queryKey: modelKeys.list(limit, offset), queryFn: () => apiRequest<Page<Model>>(`/admin/api/models?limit=${limit}&offset=${offset}`), staleTime: 60_000 });
}
