import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiRequest } from '../client';
import { keyKeys } from './useKeys';
import type { KeyCreateRequest, KeyUpdateRequest, VirtualKey } from '../types';
type AdminKeyUpdate = KeyUpdateRequest & { enabled?: boolean };
export function useCreateKey() { const queryClient = useQueryClient(); return useMutation({ mutationFn: (body: KeyCreateRequest) => apiRequest<VirtualKey>('/admin/api/keys', { method: 'POST', body }), onSuccess: () => { void queryClient.invalidateQueries({ queryKey: keyKeys.all }); } }); }
export function useUpdateKey() { const queryClient = useQueryClient(); return useMutation({ mutationFn: ({ id, body }: { id: string; body: AdminKeyUpdate }) => apiRequest<VirtualKey>(`/admin/api/keys/${id}`, { method: 'PATCH', body }), onSuccess: () => { void queryClient.invalidateQueries({ queryKey: keyKeys.all }); } }); }
export function useRegenerateKey() { const queryClient = useQueryClient(); return useMutation({ mutationFn: (id: string) => apiRequest<VirtualKey>(`/admin/api/keys/${id}/regenerate`, { method: 'POST' }), onSuccess: () => { void queryClient.invalidateQueries({ queryKey: keyKeys.all }); } }); }
export function useDeleteKey() { const queryClient = useQueryClient(); return useMutation({ mutationFn: (id: string) => apiRequest<{ success: boolean }>(`/admin/api/keys/${id}`, { method: 'DELETE' }), onSuccess: () => { void queryClient.invalidateQueries({ queryKey: keyKeys.all }); } }); }
