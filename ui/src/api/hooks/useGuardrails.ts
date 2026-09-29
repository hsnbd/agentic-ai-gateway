import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '../client';
import type { GuardrailPolicy, GuardrailViolation, Page } from '../types';
export const guardrailKeys = { policies: ['admin', 'guardrails', 'policies'] as const, violations: (limit: number, offset: number) => ['admin', 'guardrails', 'violations', limit, offset] as const };
export function useGuardrailPolicies(limit = 100, offset = 0) { return useQuery({ queryKey: [...guardrailKeys.policies, limit, offset], queryFn: () => apiRequest<Page<GuardrailPolicy>>(`/admin/api/guardrails/policies?limit=${limit}&offset=${offset}`), staleTime: 60_000 }); }
export function useGuardrailViolations(limit = 25, offset = 0) { return useQuery({ queryKey: guardrailKeys.violations(limit, offset), queryFn: () => apiRequest<Page<GuardrailViolation>>(`/admin/api/guardrails/violations?limit=${limit}&offset=${offset}`), staleTime: 15_000 }); }
