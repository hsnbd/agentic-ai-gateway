import { createContext, useCallback, useContext, useEffect, useMemo, useState, type PropsWithChildren } from 'react';
import { apiRequest, setUnauthorizedHandler } from '../api/client';
import type { AdminUser, LoginResponse } from '../api/types';

const TOKEN_KEY = 'aigateway.console.token';
export interface AuthUser { username: string; role: 'admin' | 'viewer'; fullName: string | null }
interface AuthValue { token: string | null; user: AuthUser | null; ready: boolean; login: (email: string, password: string) => Promise<void>; logout: () => void }
const AuthContext = createContext<AuthValue | undefined>(undefined);
const toAuthUser = (user: AdminUser): AuthUser => ({ username: user.email, role: user.role, fullName: user.full_name });

function decodeUser(token: string): AuthUser | null {
  try {
    const part = token.split('.')[1]; if (!part) return null;
    const claims = JSON.parse(atob(part.replace(/-/g, '+').replace(/_/g, '/'))) as { email?: unknown; role?: unknown; exp?: unknown };
    if (typeof claims.email !== 'string' || (claims.role !== 'admin' && claims.role !== 'viewer')) return null;
    if (typeof claims.exp === 'number' && claims.exp * 1000 <= Date.now()) return null;
    return { username: claims.email, role: claims.role, fullName: null };
  } catch { return null; }
}

export function AuthProvider({ children }: PropsWithChildren) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_KEY));
  const [user, setUser] = useState<AuthUser | null>(() => { const saved = localStorage.getItem(TOKEN_KEY); return saved ? decodeUser(saved) : null; });
  const [ready, setReady] = useState(false);
  const clearSession = useCallback(() => { localStorage.removeItem(TOKEN_KEY); setToken(null); setUser(null); }, []);
  useEffect(() => { setUnauthorizedHandler(clearSession); return () => setUnauthorizedHandler(() => undefined); }, [clearSession]);
  useEffect(() => {
    if (!token || !user?.username) { if (token) clearSession(); setReady(true); return; }
    let active = true;
    apiRequest<AdminUser>('/admin/api/auth/me').then((current) => { if (active) setUser(toAuthUser(current)); })
      .catch((error: unknown) => { if (active && !(error instanceof Error && 'status' in error && error.status === 401)) console.error('Unable to refresh console user profile', error); })
      .finally(() => { if (active) setReady(true); });
    return () => { active = false; };
  }, [token, user?.username, clearSession]);
  const login = useCallback(async (email: string, password: string) => {
    const result = await apiRequest<LoginResponse>('/admin/api/auth/login', { method: 'POST', body: { email, password } });
    localStorage.setItem(TOKEN_KEY, result.access_token); setToken(result.access_token); setUser(toAuthUser(result.user)); setReady(true);
  }, []);
  const value = useMemo(() => ({ token, user, ready, login, logout: clearSession }), [token, user, ready, login, clearSession]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
export function useAuth(): AuthValue { const value = useContext(AuthContext); if (!value) throw new Error('useAuth must be used inside AuthProvider'); return value; }
