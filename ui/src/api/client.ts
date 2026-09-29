import type { ApiErrorEnvelope } from './types';

const TOKEN_KEY = 'aigateway.console.token';
let unauthorizedHandler: (() => void) | undefined;

export class ApiError extends Error {
  readonly status: number;
  readonly code: string | undefined;
  readonly type: string | undefined;
  readonly details: unknown;
  constructor(message: string, status: number, details?: unknown, code?: string, type?: string) {
    super(message); this.name = 'ApiError'; this.status = status; this.details = details; this.code = code; this.type = type;
  }
}

export function setUnauthorizedHandler(handler: () => void): void { unauthorizedHandler = handler; }
export interface RequestOptions extends Omit<RequestInit, 'body'> { body?: unknown }

function errorFromPayload(status: number, payload: unknown): ApiError {
  if (typeof payload === 'object' && payload !== null) {
    const data = payload as Partial<ApiErrorEnvelope> & { detail?: unknown };
    const envelope = data.error;
    if (envelope && typeof envelope.message === 'string') return new ApiError(envelope.message, status, payload, envelope.code, envelope.type);
    if (typeof data.detail === 'string') return new ApiError(data.detail, status, payload);
    if (typeof data.detail === 'object' && data.detail !== null) {
      const detail = data.detail as { message?: unknown; code?: unknown; type?: unknown };
      if (typeof detail.message === 'string') return new ApiError(detail.message, status, payload,
        typeof detail.code === 'string' ? detail.code : undefined, typeof detail.type === 'string' ? detail.type : undefined);
    }
  }
  return new ApiError(`Request failed with status ${status}`, status, payload);
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers = new Headers(options.headers);
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const isForm = typeof FormData !== 'undefined' && options.body instanceof FormData;
  if (options.body !== undefined && !isForm) headers.set('Content-Type', 'application/json');
  const response = await fetch(path, {
    ...options, headers,
    body: options.body === undefined ? undefined : isForm ? options.body as FormData : JSON.stringify(options.body),
  });
  if (response.status === 204) return undefined as T;
  const contentType = response.headers.get('content-type') ?? '';
  const payload: unknown = contentType.includes('application/json') ? await response.json() : await response.text();
  if (!response.ok) {
    const error = errorFromPayload(response.status, payload);
    if (response.status === 401) { unauthorizedHandler?.(); if (window.location.pathname !== '/ui/login') window.location.assign('/ui/login'); }
    throw error;
  }
  return payload as T;
}
