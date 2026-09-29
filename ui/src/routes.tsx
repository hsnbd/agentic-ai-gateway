import { Suspense, lazy } from 'react';
import { CircularProgress, Stack } from '@mui/material';
import { Navigate, Route, Routes } from 'react-router-dom';
import { RequireAuth, RequireRole } from './auth/RouteGuards';
import { AppShell } from './components/AppShell';
import { ErrorBoundary } from './components/ErrorBoundary';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Keys = lazy(() => import('./pages/Keys'));
const Teams = lazy(() => import('./pages/Teams'));
const Models = lazy(() => import('./pages/Models'));
const Logs = lazy(() => import('./pages/Logs'));
const Usage = lazy(() => import('./pages/Usage'));
const Guardrails = lazy(() => import('./pages/Guardrails'));
const Cache = lazy(() => import('./pages/Cache'));
const RagCollections = lazy(() => import('./pages/RagCollections'));
const McpServers = lazy(() => import('./pages/McpServers'));
const Playground = lazy(() => import('./pages/Playground'));
const Settings = lazy(() => import('./pages/Settings'));
const Login = lazy(() => import('./pages/Login'));
const NotFound = lazy(() => import('./pages/NotFound'));
const fallback = <Stack minHeight="60vh" alignItems="center" justifyContent="center"><CircularProgress /></Stack>;

export default function AppRoutes() {
  return <ErrorBoundary><Suspense fallback={fallback}><Routes>
    <Route path="/login" element={<Login />} />
    <Route element={<RequireAuth />}>
      <Route element={<AppShell />}>
        <Route index element={<Dashboard />} />

        {/* Readable by viewers. Each page hides its own mutating controls by
            role, and the backend enforces the same rules independently. */}
        <Route path="teams" element={<Teams />} />
        <Route path="models" element={<Models />} />
        <Route path="logs" element={<Logs />} />
        <Route path="usage" element={<Usage />} />
        <Route path="guardrails" element={<Guardrails />} />
        <Route path="cache" element={<Cache />} />
        <Route path="rag" element={<RagCollections />} />
        <Route path="mcp" element={<McpServers />} />

        {/* Admin only. Keys and Settings expose credentials and account
            management; the playground spends real money against providers and
            is rejected for viewers by the admin API regardless of this guard. */}
        <Route element={<RequireRole role="admin" />}>
          <Route path="keys" element={<Keys />} />
          <Route path="settings" element={<Settings />} />
          <Route path="playground" element={<Playground />} />
        </Route>

        <Route path="404" element={<NotFound />} />
        <Route path="*" element={<Navigate to="/404" replace />} />
      </Route>
    </Route>
    <Route path="*" element={<Navigate to="/" replace />} />
  </Routes></Suspense></ErrorBoundary>;
}
