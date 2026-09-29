import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiRequest } from '../client';
import type { AdminUser, Page, ProviderStatus } from '../types';
export const userKeys = { all: ['admin', 'users'] as const, providers: ['admin', 'providers-status'] as const };
export interface UserCreate { email: string; password: string; role: AdminUser['role'] }
export interface UserUpdate { role?: AdminUser['role']; is_active?: boolean }
export function useUsers() { return useQuery({ queryKey: userKeys.all, queryFn: () => apiRequest<Page<AdminUser>>('/admin/api/users?limit=200') }); }
export function useCreateUser() { const queryClient = useQueryClient(); return useMutation({ mutationFn: (body: UserCreate) => apiRequest<AdminUser>('/admin/api/users', { method: 'POST', body }), onSuccess: () => { void queryClient.invalidateQueries({ queryKey: userKeys.all }); } }); }
export function useUpdateUser() { const queryClient = useQueryClient(); return useMutation({ mutationFn: ({ id, body }: { id: string; body: UserUpdate }) => apiRequest<AdminUser>(`/admin/api/users/${id}`, { method: 'PATCH', body }), onSuccess: () => { void queryClient.invalidateQueries({ queryKey: userKeys.all }); } }); }
export function useDeleteUser() { const queryClient = useQueryClient(); return useMutation({ mutationFn: (id: string) => apiRequest<{ success: boolean }>(`/admin/api/users/${id}`, { method: 'DELETE' }), onSuccess: () => { void queryClient.invalidateQueries({ queryKey: userKeys.all }); } }); }
export function useProviderStatus() { return useQuery({ queryKey: userKeys.providers, queryFn: () => apiRequest<Page<ProviderStatus>>('/admin/api/providers/status?limit=100'), staleTime: 30_000 }); }
