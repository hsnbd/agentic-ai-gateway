import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '../client';
import type { Page, Team } from '../types';
export const teamKeys = { all: ['admin', 'teams'] as const };
export function useTeams(limit = 100, offset = 0) { return useQuery({ queryKey: [...teamKeys.all, limit, offset], queryFn: () => apiRequest<Page<Team>>(`/admin/api/teams?limit=${limit}&offset=${offset}`), staleTime: 60_000 }); }
