import type { ApiErrorEnvelope } from './types';

const TOKEN_KEY = 'aigateway.console.token';
const REFRESH_KEY = 'aigateway.console.refresh';
let unauthorizedHandler: (() => void) | undefined;
const sessionListeners = new Set<(token: string | null) => void>();

/** Console session storage; the auth provider subscribes to changes. */
export const session = {
  token: (): string | null => localStorage.getItem(TOKEN_KEY),
  refreshToken: (): string | null => localStorage.getItem(REFRESH_KEY),
  set(access: string, refresh?: string | null): void {
    localStorage.setItem(TOKEN_KEY, access);
    if (refresh) localStorage.setItem(REFRESH_KEY, refresh);
    sessionListeners.forEach((listener) => listener(access));
  },
  clear(): void {
    localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(REFRESH_KEY);
    sessionListeners.forEach((listener) => listener(null));
  },
  subscribe(listener: (token: string | null) => void): () => void { sessionListeners.add(listener); return () => { sessionListeners.delete(listener); }; },
};

let refreshing: Promise<boolean> | null = null;

/** Exchange the refresh token for a new pair. Concurrent callers share one request. */
export function refreshSession(): Promise<boolean> {
  const refresh = session.refreshToken();
  if (!refresh) return Promise.resolve(false);
  refreshing ??= fetch('/admin/api/auth/refresh', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ refresh_token: refresh }),
  }).then(async (response) => {
    if (!response.ok) return false;
    const data = await response.json() as { access_token: string; refresh_token?: string | null };
    session.set(data.access_token, data.refresh_token);
    return true;
  }).catch(() => false).finally(() => { refreshing = null; });
  return refreshing;
}

const AUTH_PATHS = ['/admin/api/auth/login', '/admin/api/auth/refresh'];

function signedOut(): void {
  unauthorizedHandler?.();
  if (window.location.pathname !== '/ui/login') window.location.assign('/ui/login');
}

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

async function send(path: string, options: RequestOptions): Promise<Response> {
  const headers = new Headers(options.headers);
  const token = session.token();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const isForm = typeof FormData !== 'undefined' && options.body instanceof FormData;
  if (options.body !== undefined && !isForm) headers.set('Content-Type', 'application/json');
  return fetch(path, {
    ...options, headers,
    body: options.body === undefined ? undefined : isForm ? options.body as FormData : JSON.stringify(options.body),
  });
}

/** Send, renewing an expired access token once before giving up on the session. */
async function authorized(path: string, options: RequestOptions): Promise<Response> {
  let response = await send(path, options);
  if (response.status === 401 && !AUTH_PATHS.includes(path) && await refreshSession()) response = await send(path, options);
  return response;
}

async function payloadOf(response: Response): Promise<unknown> {
  const contentType = response.headers.get('content-type') ?? '';
  return contentType.includes('application/json') ? response.json() : response.text();
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await authorized(path, options);
  if (response.status === 204) return undefined as T;
  const payload = await payloadOf(response);
  if (!response.ok) {
    const error = errorFromPayload(response.status, payload);
    if (response.status === 401 && !AUTH_PATHS.includes(path)) signedOut();
    throw error;
  }
  return payload as T;
}

/** Fetch a file (e.g. a CSV export) with the console session and save it. */
export async function apiDownload(path: string, filename: string): Promise<void> {
  const response = await authorized(path, {});
  if (!response.ok) {
    if (response.status === 401) signedOut();
    throw errorFromPayload(response.status, await payloadOf(response));
  }
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = url; link.download = filename; link.click();
  URL.revokeObjectURL(url);
}
