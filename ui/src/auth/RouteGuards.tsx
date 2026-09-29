import { CircularProgress, Stack } from '@mui/material';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from './AuthProvider';

export function RequireAuth() {
  const { user, ready } = useAuth(); const location = useLocation();
  if (!ready) return <Stack minHeight="100vh" alignItems="center" justifyContent="center"><CircularProgress /></Stack>;
  return user ? <Outlet /> : <Navigate to="/login" replace state={{ from: location.pathname }} />;
}
export function RequireRole({ role }: { role: 'admin' }) {
  const { user } = useAuth(); return user?.role === role ? <Outlet /> : <Navigate to="/" replace />;
}
