import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, apiDownload, apiRequest, refreshSession, session, setUnauthorizedHandler } from './client';

type Handler = (url: string, init: RequestInit) => Response | Promise<Response>;

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

function mockFetch(handler: Handler) {
  const spy = vi.fn((input: RequestInfo | URL, init?: RequestInit) => Promise.resolve(handler(String(input), init ?? {})));
  vi.stubGlobal('fetch', spy);
  return spy;
}

const assign = vi.fn();

beforeEach(() => {
  vi.stubGlobal('location', { pathname: '/ui/logs', assign });
  assign.mockReset();
  setUnauthorizedHandler(() => undefined);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('session', () => {
  it('stores tokens and notifies subscribers', () => {
    const seen: Array<string | null> = [];
    const unsubscribe = session.subscribe((token) => seen.push(token));
    session.set('access-1', 'refresh-1');
    session.set('access-2');
    expect(session.token()).toBe('access-2');
    expect(session.refreshToken()).toBe('refresh-1');
    session.clear();
    unsubscribe();
    session.set('ignored');
    expect(seen).toEqual(['access-1', 'access-2', null]);
    expect(session.refreshToken()).toBeNull();
  });
});

describe('refreshSession', () => {
  it('returns false without a refresh token', async () => {
    const spy = mockFetch(() => json({}));
    await expect(refreshSession()).resolves.toBe(false);
    expect(spy).not.toHaveBeenCalled();
  });

  it('exchanges the refresh token once for concurrent callers', async () => {
    session.set('old', 'refresh');
    const spy = mockFetch(() => json({ access_token: 'new', refresh_token: 'rotated' }));
    const [first, second] = await Promise.all([refreshSession(), refreshSession()]);
    expect(first && second).toBe(true);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(session.token()).toBe('new');
    expect(session.refreshToken()).toBe('rotated');
  });

  it('fails on a rejected or unreachable refresh', async () => {
    session.set('old', 'refresh');
    mockFetch(() => json({}, 401));
    await expect(refreshSession()).resolves.toBe(false);
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('offline'))));
    await expect(refreshSession()).resolves.toBe(false);
  });
});

describe('apiRequest', () => {
  it('sends the bearer token and JSON bodies', async () => {
    session.set('token');
    const spy = mockFetch(() => json({ ok: true }));
    await expect(apiRequest('/admin/api/x', { method: 'POST', body: { a: 1 } })).resolves.toEqual({ ok: true });
    const init = spy.mock.calls[0][1] as RequestInit;
    const headers = new Headers(init.headers);
    expect(headers.get('Authorization')).toBe('Bearer token');
    expect(headers.get('Content-Type')).toBe('application/json');
    expect(init.body).toBe('{"a":1}');
  });

  it('passes form data through without a JSON content type', async () => {
    const spy = mockFetch(() => new Response('plain text', { status: 200 }));
    const form = new FormData();
    form.append('file', 'x');
    await expect(apiRequest('/v1/rag', { method: 'POST', body: form })).resolves.toBe('plain text');
    const init = spy.mock.calls[0][1] as RequestInit;
    expect(new Headers(init.headers).has('Content-Type')).toBe(false);
    expect(new Headers(init.headers).has('Authorization')).toBe(false);
    expect(init.body).toBe(form);
  });

  it('returns undefined for 204 responses', async () => {
    mockFetch(() => new Response(null, { status: 204 }));
    await expect(apiRequest('/admin/api/x', { method: 'DELETE' })).resolves.toBeUndefined();
  });

  it.each([
    [{ error: { message: 'envelope', code: 'bad', type: 'invalid' } }, 'envelope', 'bad', 'invalid'],
    [{ detail: 'plain detail' }, 'plain detail', undefined, undefined],
    [{ detail: { message: 'nested', code: 'c', type: 't' } }, 'nested', 'c', 't'],
    [{ detail: { message: 'nested-no-code', code: 1, type: 2 } }, 'nested-no-code', undefined, undefined],
    [{ detail: { other: true } }, 'Request failed with status 400', undefined, undefined],
    [{ detail: null }, 'Request failed with status 400', undefined, undefined],
    [{ error: { message: 5 } }, 'Request failed with status 400', undefined, undefined],
  ])('maps error payload %j', async (payload, message, code, type) => {
    mockFetch(() => json(payload, 400));
    const error = await apiRequest('/admin/api/x').catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ message, status: 400, code, type, details: payload, name: 'ApiError' });
  });

  it('treats a body-less error as an empty text payload', async () => {
    mockFetch(() => new Response(null, { status: 500 }));
    await expect(apiRequest('/admin/api/x')).rejects.toMatchObject({ status: 500, details: '' });
  });

  it('reports a non-JSON error body with a generic message', async () => {
    mockFetch(() => new Response('gateway exploded', { status: 502 }));
    await expect(apiRequest('/admin/api/x')).rejects.toMatchObject({ message: 'Request failed with status 502', details: 'gateway exploded' });
  });

  it('retries once after a successful silent refresh', async () => {
    session.set('expired', 'refresh');
    let calls = 0;
    const spy = mockFetch((url) => {
      if (url === '/admin/api/auth/refresh') return json({ access_token: 'fresh' });
      calls += 1;
      return calls === 1 ? json({ detail: 'expired' }, 401) : json({ ok: true });
    });
    await expect(apiRequest('/admin/api/me')).resolves.toEqual({ ok: true });
    const retried = spy.mock.calls.at(-1)?.[1] as RequestInit;
    expect(new Headers(retried.headers).get('Authorization')).toBe('Bearer fresh');
  });

  it('signs out when the session cannot be renewed', async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    mockFetch(() => json({ detail: 'expired' }, 401));
    await expect(apiRequest('/admin/api/me')).rejects.toMatchObject({ status: 401 });
    expect(handler).toHaveBeenCalled();
    expect(assign).toHaveBeenCalledWith('/ui/login');
  });

  it('does not redirect when already on the login page or for auth endpoints', async () => {
    vi.stubGlobal('location', { pathname: '/ui/login', assign });
    mockFetch(() => json({ detail: 'expired' }, 401));
    await expect(apiRequest('/admin/api/me')).rejects.toBeInstanceOf(ApiError);
    expect(assign).not.toHaveBeenCalled();

    vi.stubGlobal('location', { pathname: '/ui/logs', assign });
    await expect(apiRequest('/admin/api/auth/login', { method: 'POST', body: {} })).rejects.toMatchObject({ message: 'expired' });
    expect(assign).not.toHaveBeenCalled();
  });
});

describe('apiDownload', () => {
  it('saves the response body under the given filename', async () => {
    mockFetch(() => new Response('a,b\n1,2', { status: 200 }));
    const createObjectURL = vi.fn(() => 'blob:csv');
    const revokeObjectURL = vi.fn();
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL, revokeObjectURL }));
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
    await apiDownload('/admin/api/logs/export', 'logs.csv');
    expect(click).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:csv');
  });

  it('raises and signs out on failures', async () => {
    mockFetch(() => json({ detail: 'nope' }, 401));
    await expect(apiDownload('/admin/api/logs/export', 'logs.csv')).rejects.toMatchObject({ message: 'nope' });
    expect(assign).toHaveBeenCalledWith('/ui/login');

    assign.mockReset();
    mockFetch(() => json({ detail: 'forbidden' }, 403));
    await expect(apiDownload('/admin/api/logs/export', 'logs.csv')).rejects.toMatchObject({ status: 403 });
    expect(assign).not.toHaveBeenCalled();
  });
});
