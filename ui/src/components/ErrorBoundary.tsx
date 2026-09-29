import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Alert, Box, Button, Typography } from '@mui/material';
interface Props { children: ReactNode }
interface State { error: Error | null }
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };
  static getDerivedStateFromError(error: Error): State { return { error }; }
  componentDidCatch(error: Error, info: ErrorInfo): void { console.error('Console rendering failed', error, info.componentStack); }
  render() {
    if (this.state.error) return <Box sx={{ maxWidth: 600, mx: 'auto', mt: 8 }}><Alert severity="error"><Typography variant="h3">Something went wrong</Typography><Typography sx={{ mt: 1 }}>{this.state.error.message}</Typography><Button sx={{ mt: 2 }} onClick={() => window.location.assign('/ui/')}>Reload console</Button></Alert></Box>;
    return this.props.children;
  }
}
