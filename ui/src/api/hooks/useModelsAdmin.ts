import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '../client';
import type { Model, Page } from '../types';
export const adminModelKeys = { all: ['admin', 'models'] as const };
export function useAdminModels(limit = 100, offset = 0) { return useQuery({ queryKey: [...adminModelKeys.all, limit, offset], queryFn: () => apiRequest<Page<Model>>(`/admin/api/models?limit=${limit}&offset=${offset}`), staleTime: 60_000 }); }
