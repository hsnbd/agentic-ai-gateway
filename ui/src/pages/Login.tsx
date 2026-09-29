import { useState, type FormEvent } from 'react';
import { Alert, Box, Button, Paper, Stack, TextField, Typography } from '@mui/material';
import { Navigate, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthProvider';

export default function Login() {
  const { login, user, ready } = useAuth();
  const [email, setEmail] = useState(''); const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null); const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate(); const location = useLocation();
  if (ready && user) return <Navigate to="/" replace />;
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError(null); setSubmitting(true);
    try {
      await login(email, password);
      const from = (location.state as { from?: string } | null)?.from ?? '/';
      navigate(from, { replace: true });
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Unable to sign in.'); }
    finally { setSubmitting(false); }
  }
  return <Box sx={{ minHeight: '100vh', display: 'grid', placeItems: 'center', p: 2, bgcolor: 'background.default' }}>
    <Paper sx={{ width: '100%', maxWidth: 420, p: { xs: 3, sm: 4 } }}>
      <Stack component="form" onSubmit={submit} spacing={2.5}>
        <Box><Typography variant="h2">Welcome back</Typography><Typography color="text.secondary" sx={{ mt: 1 }}>Sign in to your AI Gateway Console.</Typography></Box>
        {error && <Alert severity="error">{error}</Alert>}
        <TextField label="Email" type="email" autoComplete="username" required fullWidth value={email} onChange={(event) => setEmail(event.target.value)} />
        <TextField label="Password" type="password" autoComplete="current-password" required fullWidth value={password} onChange={(event) => setPassword(event.target.value)} />
        <Button type="submit" variant="contained" size="large" disabled={submitting}>{submitting ? 'Signing in...' : 'Sign in'}</Button>
      </Stack>
    </Paper>
  </Box>;
}
