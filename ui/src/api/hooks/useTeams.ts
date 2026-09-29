import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiRequest } from '../client';
import type { Page, Team, TeamCreateRequest, TeamUpdateRequest, TeamUsage } from '../types';
export const teamKeys = { all: ['admin', 'teams'] as const, usage: (id: string) => ['admin', 'teams', id, 'usage'] as const };
export function useTeams(limit = 100, offset = 0) { return useQuery({ queryKey: [...teamKeys.all, limit, offset], queryFn: () => apiRequest<Page<Team>>(`/admin/api/teams?limit=${limit}&offset=${offset}`), staleTime: 60_000 }); }
export function useTeamUsage(id: string | undefined, window = '30d') { return useQuery({ queryKey: [...teamKeys.usage(id ?? ''), window], queryFn: () => apiRequest<TeamUsage>(`/admin/api/teams/${id}/usage?window=${window}`), enabled: Boolean(id) }); }
export function useCreateTeam() { const queryClient = useQueryClient(); return useMutation({ mutationFn: (body: TeamCreateRequest) => apiRequest<Team>('/admin/api/teams', { method: 'POST', body }), onSuccess: () => { void queryClient.invalidateQueries({ queryKey: teamKeys.all }); } }); }
export function useUpdateTeam() { const queryClient = useQueryClient(); return useMutation({ mutationFn: ({ id, body }: { id: string; body: TeamUpdateRequest }) => apiRequest<Team>(`/admin/api/teams/${id}`, { method: 'PATCH', body }), onSuccess: () => { void queryClient.invalidateQueries({ queryKey: teamKeys.all }); } }); }
export function useDeleteTeam() { const queryClient = useQueryClient(); return useMutation({ mutationFn: (id: string) => apiRequest<{ success: boolean }>(`/admin/api/teams/${id}`, { method: 'DELETE' }), onSuccess: () => { void queryClient.invalidateQueries({ queryKey: teamKeys.all }); void queryClient.invalidateQueries({ queryKey: ['keys'] }); } }); }
