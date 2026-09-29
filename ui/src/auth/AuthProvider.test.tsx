import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { session } from '../api/client';
import { AuthProvider, useAuth } from './AuthProvider';
import { RequireAuth, RequireRole } from './RouteGuards';

function jwt(claims: Record<string, unknown>): string {
  const encode = (value: unknown) => btoa(JSON.stringify(value)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  return `${encode({ alg: 'HS256' })}.${encode(claims)}.signature`;
}

const ADMIN = { id: '1', email: 'admin@example.test', role: 'admin', full_name: 'Ada Admin' };

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

function Probe() {
  const { user, ready, token, login, logout } = useAuth();
  return <div>
    <span data-testid="ready">{String(ready)}</span>
    <span data-testid="user">{user ? `${user.username}:${user.role}:${user.fullName ?? ''}` : 'none'}</span>
    <span data-testid="token">{token ?? 'none'}</span>
    <button onClick={() => void login('admin@example.test', 'pw')}>login</button>
    <button onClick={logout}>logout</button>
  </div>;
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith('/auth/me')) return Promise.resolve(json(ADMIN));
    if (url.endsWith('/auth/login')) return Promise.resolve(json({ access_token: 'new-access', refresh_token: 'new-refresh', user: ADMIN }));
    return Promise.resolve(json({ success: true }));
  });
  vi.stubGlobal('fetch', fetchMock);
  vi.stubGlobal('location', { pathname: '/ui/', assign: vi.fn() });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('AuthProvider', () => {
  it('starts signed out and signs in', async () => {
    render(<AuthProvider><Probe /></AuthProvider>);
    await waitFor(() => expect(screen.getByTestId('ready')).toHaveTextContent('true'));
    expect(screen.getByTestId('user')).toHaveTextContent('none');
    await userEvent.click(screen.getByText('login'));
    await waitFor(() => expect(screen.getByTestId('user')).toHaveTextContent('admin@example.test:admin:Ada Admin'));
    expect(session.token()).toBe('new-access');
  });

  it('restores a saved session and refreshes the profile', async () => {
    session.set(jwt({ email: 'admin@example.test', role: 'admin' }), 'refresh');
    render(<AuthProvider><Probe /></AuthProvider>);
    expect(screen.getByTestId('user')).toHaveTextContent('admin@example.test:admin:');
    await waitFor(() => expect(screen.getByTestId('user')).toHaveTextContent('Ada Admin'));
  });

  it.each([
    ['not-a-jwt'],
    [jwt({ email: 'x@example.test', role: 'owner' })],
    [`header.${btoa('{not json')}.sig`],
  ])('clears an unusable saved token %s', async (token) => {
    session.set(token);
    render(<AuthProvider><Probe /></AuthProvider>);
    await waitFor(() => expect(screen.getByTestId('ready')).toHaveTextContent('true'));
    expect(screen.getByTestId('user')).toHaveTextContent('none');
    expect(session.token()).toBeNull();
  });

  it('logs profile failures other than 401', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    fetchMock.mockImplementation(() => Promise.resolve(json({ detail: 'boom' }, 500)));
    session.set(jwt({ email: 'admin@example.test', role: 'viewer' }));
    render(<AuthProvider><Probe /></AuthProvider>);
    await waitFor(() => expect(screen.getByTestId('ready')).toHaveTextContent('true'));
    expect(consoleError).toHaveBeenCalledWith('Unable to refresh console user profile', expect.anything());
  });

  it('stays quiet when the profile call is unauthorised', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    fetchMock.mockImplementation(() => Promise.resolve(json({ detail: 'expired' }, 401)));
    session.set(jwt({ email: 'admin@example.test', role: 'admin' }));
    render(<AuthProvider><Probe /></AuthProvider>);
    await waitFor(() => expect(screen.getByTestId('user')).toHaveTextContent('none'));
    expect(consoleError).not.toHaveBeenCalled();
  });

  it('follows silent token refreshes and ignores late profile answers', async () => {
    let answer: (response: Response) => void = () => undefined;
    fetchMock.mockImplementation(() => new Promise<Response>((resolve) => { answer = resolve; }));
    session.set(jwt({ email: 'admin@example.test', role: 'admin' }));
    const { unmount } = render(<AuthProvider><Probe /></AuthProvider>);
    act(() => session.set('rotated'));
    expect(screen.getByTestId('token')).toHaveTextContent('rotated');
    unmount();
    answer(json(ADMIN));
  });

  it('logs out locally even when the server call fails', async () => {
    fetchMock.mockImplementation((input: RequestInfo | URL) => String(input).endsWith('/logout')
      ? Promise.reject(new Error('offline')) : Promise.resolve(json(ADMIN)));
    session.set(jwt({ email: 'admin@example.test', role: 'admin' }), 'refresh');
    render(<AuthProvider><Probe /></AuthProvider>);
    await userEvent.click(screen.getByText('logout'));
    expect(screen.getByTestId('user')).toHaveTextContent('none');
    expect(session.token()).toBeNull();
    const logoutCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/logout'));
    expect(logoutCall).toBeDefined();
  });

  it('skips the server call when logging out without a session', async () => {
    render(<AuthProvider><Probe /></AuthProvider>);
    await userEvent.click(screen.getByText('logout'));
    expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith('/logout'))).toBe(false);
  });

  it('refuses to be used outside the provider', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    expect(() => render(<Probe />)).toThrow('useAuth must be used inside AuthProvider');
  });
});

describe('route guards', () => {
  function app(path: string) {
    return render(<AuthProvider><MemoryRouter initialEntries={[path]}><Routes>
      <Route path="/login" element={<p>login page</p>} />
      <Route element={<RequireAuth />}>
        <Route path="/" element={<p>home</p>} />
        <Route element={<RequireRole role="admin" />}><Route path="/users" element={<p>users</p>} /></Route>
      </Route>
    </Routes></MemoryRouter></AuthProvider>);
  }

  it('shows a spinner while a saved session is being checked', () => {
    fetchMock.mockImplementation(() => new Promise<Response>(() => undefined));
    session.set(jwt({ email: 'admin@example.test', role: 'admin' }));
    app('/');
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  it('redirects guests to the login page', async () => {
    app('/');
    expect(await screen.findByText('login page')).toBeInTheDocument();
  });

  it('lets admins in and sends viewers home from admin pages', async () => {
    session.set(jwt({ email: 'admin@example.test', role: 'admin' }));
    const { unmount } = app('/users');
    expect(await screen.findByText('users')).toBeInTheDocument();
    unmount();

    fetchMock.mockImplementation(() => Promise.resolve(json({ ...ADMIN, role: 'viewer' })));
    session.set(jwt({ email: 'viewer@example.test', role: 'viewer' }));
    app('/users');
    expect(await screen.findByText('home')).toBeInTheDocument();
  });
});
