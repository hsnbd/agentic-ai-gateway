import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider, QueryCache, MutationCache } from '@tanstack/react-query';
import { AuthProvider } from './auth/AuthProvider';
import { ColorModeProvider } from './theme/ColorModeContext';
import AppRoutes from './routes';

const reportQueryError = (error: Error) => { console.error('Gateway API query failed', error); };
const queryClient = new QueryClient({
  queryCache: new QueryCache({ onError: reportQueryError }),
  mutationCache: new MutationCache({ onError: reportQueryError }),
  defaultOptions: { queries: { retry: (count, error) => !(error instanceof Error && 'status' in error && error.status === 401) && count < 1, refetchOnWindowFocus: false } },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><QueryClientProvider client={queryClient}><ColorModeProvider><AuthProvider><BrowserRouter basename="/ui"><AppRoutes /></BrowserRouter></AuthProvider></ColorModeProvider></QueryClientProvider></React.StrictMode>,
);
