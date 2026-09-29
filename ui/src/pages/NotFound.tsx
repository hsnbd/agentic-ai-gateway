import { Button, Stack, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';
export default function NotFound() {
  const navigate = useNavigate();
  return <Stack alignItems="center" justifyContent="center" spacing={2} sx={{ minHeight: '70vh', textAlign: 'center' }}>
    <Typography variant="h1">404</Typography><Typography variant="h3">Page not found</Typography>
    <Typography color="text.secondary">The page you requested does not exist.</Typography><Button variant="contained" onClick={() => navigate('/')}>Back to dashboard</Button>
  </Stack>;
}
