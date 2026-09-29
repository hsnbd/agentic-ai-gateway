import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiRequest } from '../client';
import type { ConfigReloadResponse, Deployment, HealthCheckResponse, Page, SystemInfo } from '../types';
export const adminKeys = { deployments: (limit: number, offset: number) => ['admin', 'deployments', limit, offset] as const, systemInfo: ['admin', 'system-info'] as const };
export function useDeployments(limit = 100, offset = 0) { return useQuery({ queryKey: adminKeys.deployments(limit, offset), queryFn: () => apiRequest<Page<Deployment>>(`/admin/api/deployments?limit=${limit}&offset=${offset}`), staleTime: 30_000 }); }
export function useSystemInfo() { return useQuery({ queryKey: adminKeys.systemInfo, queryFn: () => apiRequest<SystemInfo>('/admin/api/system/info'), staleTime: 30_000 }); }
export function useDeploymentHealthCheck() { const queryClient = useQueryClient(); return useMutation({ mutationFn: (deploymentId: string) => apiRequest<HealthCheckResponse>(`/admin/api/deployments/${encodeURIComponent(deploymentId)}/health-check`, { method: 'POST' }), onSuccess: () => { void queryClient.invalidateQueries({ queryKey: ['admin', 'deployments'] }); } }); }
export function useReloadConfig() { const queryClient = useQueryClient(); return useMutation({ mutationFn: () => apiRequest<ConfigReloadResponse>('/admin/api/config/reload', { method: 'POST' }), onSuccess: () => { void queryClient.invalidateQueries({ queryKey: ['admin'] }); void queryClient.invalidateQueries({ queryKey: ['models'] }); } }); }
