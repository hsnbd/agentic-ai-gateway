import { useMemo, useState } from 'react';
import { Alert, Box, Button, Grid, Stack, TextField, Typography, Tooltip } from '@mui/material';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip as ChartTooltip, XAxis, YAxis } from 'recharts';
import DeleteSweepRounded from '@mui/icons-material/DeleteSweepRounded';
import { DataGrid } from '@mui/x-data-grid';
import type { GridColDef, GridPaginationModel } from '@mui/x-data-grid';
import { useAuth } from '../auth/AuthProvider';
import { apiRequest } from '../api/client';
import { useCacheStats, useCacheEntries, useMetricSeries } from '../api/hooks/useUsageObservability';
import { ConfirmDialog, EmptyState, ErrorState, LoadingState, PageHeader, StatCard } from '../components/Shared';
import { ChartPanel, WindowSelect, useChartColors, type WindowValue } from '../features/observability/Observability';
import { formatMoney, formatNumber, formatWindow } from '../features/observability/format';
import type { CacheEntry, CacheStats } from '../features/observability/types';

interface InvalidateResponse { invalidated: number }
type Invalidation = { key: string } | { namespace: string } | { all_entries: true };

export default function Cache() {
  const [window, setWindow] = useState<WindowValue>('24h');
  const [entryKey, setEntryKey] = useState('');
  const [namespace, setNamespace] = useState('');
  const [pageModel, setPageModel] = useState<GridPaginationModel>({ page: 0, pageSize: 25 });
  const [confirmation, setConfirmation] = useState<Invalidation | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const colors = useChartColors();
  const stats = useCacheStats();
  const entries = useCacheEntries(pageModel.pageSize, pageModel.page * pageModel.pageSize);
  const requests = useMetricSeries(window, 'requests');
  const hits = useMetricSeries(window, 'cache_hits');
  const invalidate = useMutation({
    mutationFn: (body: Invalidation) => apiRequest<InvalidateResponse>('/admin/api/cache/invalidate', { method: 'POST', body }),
    onError: () => setConfirmation(null),
    onSuccess: async (result) => {
      setSuccess(`${formatNumber(result.invalidated)} cache ${result.invalidated === 1 ? 'entry' : 'entries'} invalidated.`);
      setConfirmation(null);
      setEntryKey('');
      setNamespace('');
      await queryClient.invalidateQueries({ queryKey: ['cache'] });
    },
  });
  const chartData = useMemo(() => {
    const requestPoints = new Map((requests.data?.points ?? []).map((point) => [point.timestamp, point.value]));
    return (hits.data?.points ?? []).map((point) => {
      const total = requestPoints.get(point.timestamp);
      return { timestamp: point.timestamp, label: new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', hour: '2-digit' }).format(new Date(point.timestamp)),
        ratio: total && total > 0 ? point.value / total * 100 : null };
    }).filter((point) => point.ratio !== null);
  }, [hits.data, requests.data]);
  const ratio = (value: CacheStats | undefined): string => value?.hits !== null && value?.hits !== undefined && value.misses !== null && value.misses !== undefined && value.hits + value.misses > 0
    ? `${formatNumber(value.hits / (value.hits + value.misses) * 100, 1)}%` : 'Not recorded';
  const admin = user?.role === 'admin';
  const columns: GridColDef<CacheEntry>[] = [
    { field: 'key', headerName: 'Entry key', minWidth: 205, flex: 1, renderCell: ({ value }) => <Tooltip title={String(value)}><span>{String(value)}</span></Tooltip> },
    { field: 'model', headerName: 'Model', minWidth: 115, flex: 0.6, valueFormatter: (value) => value == null ? 'Not recorded' : String(value) },
    { field: 'namespace', headerName: 'Namespace', minWidth: 165, flex: 0.8, renderCell: ({ value }) => value === null ? 'Not recorded' : <Tooltip title={String(value)}><span>{String(value)}</span></Tooltip> },
    { field: 'hit_count', headerName: 'Hits', minWidth: 100, valueFormatter: (value) => value === null ? 'Not recorded' : formatNumber(Number(value)) },
    { field: 'age_seconds', headerName: 'Age', minWidth: 125, valueFormatter: (value) => value === null ? 'Not recorded' : `${formatNumber(Number(value), 0)} s` },
    { field: 'ttl_remaining_seconds', headerName: 'TTL remaining', minWidth: 125, valueFormatter: (value) => value === null ? 'Not recorded' : `${formatNumber(Number(value))} s` },
    { field: 'cached_prompt', headerName: 'Cached prompt (redacted)', minWidth: 170, flex: 1, renderCell: ({ value }) => value === null ? 'Not recorded' : <Tooltip title={String(value)}><span>{String(value)}</span></Tooltip> },
    { field: 'actions', headerName: 'Action', minWidth: 120, sortable: false, renderCell: ({ row }) => admin ? <Button size="small" color="error" onClick={() => { setSuccess(null); setConfirmation({ key: row.key }); }}>Invalidate</Button> : null },
  ];
  const validNamespace = /^[a-f0-9]{64}$/.test(namespace.trim());

  return <>
    <PageHeader title="Semantic cache" description="Inspect cache effectiveness and manage stored semantic responses." action={<WindowSelect value={window} onChange={setWindow} />} />
    {stats.isPending ? <LoadingState label="Loading cache statistics" /> : stats.isError ? <ErrorState error={stats.error} onRetry={() => void stats.refetch()} /> : <>
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6} lg={3}><StatCard label="All-time hit ratio" value={ratio(stats.data)} hint="Computed from cumulative cache counters; trend below uses the selected window." /></Grid>
        <Grid item xs={12} sm={6} lg={3}><StatCard label="Total hits" value={formatNumber(stats.data.hits)} hint="All-time cache hit counter" /></Grid>
        <Grid item xs={12} sm={6} lg={3}><StatCard label="Total misses" value={formatNumber(stats.data.misses)} hint="All-time cache miss counter" /></Grid>
        <Grid item xs={12} sm={6} lg={3}><StatCard label="Entry count" value={formatNumber(stats.data.entries)} hint="Current cache entries, when provided" /></Grid>
        <Grid item xs={12} sm={6} lg={3}><StatCard label="Estimated cost saved" value={stats.data.estimated_cost_saved_usd === null ? "Not recorded" : formatMoney(stats.data.estimated_cost_saved_usd)} hint="Estimated cumulative cost avoided" /></Grid>
        <Grid item xs={12} sm={6} lg={3}><StatCard label="Estimated latency saved" value={stats.data.estimated_latency_saved_ms === null ? "Not recorded" : `${formatNumber(stats.data.estimated_latency_saved_ms)} ms`} hint="Per-hit avoided latency is not recorded" /></Grid>
        <Grid item xs={12} sm={6} lg={3}><StatCard label="Vector index size" value={stats.data.index_size_bytes === null ? "Not recorded" : `${formatNumber(stats.data.index_size_bytes)} bytes`} hint="Redis Search index size when reported" /></Grid>
        <Grid item xs={12} sm={6} lg={3}><StatCard label="Cache state" value={!stats.data.enabled ? 'Disabled' : stats.data.available ? 'Available' : 'Unavailable'} hint="Enabled configuration and Redis availability" /></Grid>
      </Grid>
      <Grid container spacing={2} sx={{ mt: 0.25 }}>
        <Grid item xs={12}><ChartPanel title="Cache hit ratio over time" description={`Hits divided by requests · ${formatWindow(window)}`}>
          {requests.isPending || hits.isPending ? <LoadingState label="Loading cache hit trend" /> : requests.isError ? <ErrorState error={requests.error} onRetry={() => void requests.refetch()} /> : hits.isError ? <ErrorState error={hits.error} onRetry={() => void hits.refetch()} /> : chartData.length === 0 ? <EmptyState title="No cache trend data" description="The API has no cache-hit points for this time window." /> :
            <ResponsiveContainer width="100%" height="100%"><AreaChart data={chartData}><CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
              <XAxis dataKey="label" tick={{ fill: colors.text, fontSize: 11 }} /><YAxis domain={[0, 100]} tickFormatter={(value: number) => `${value}%`} tick={{ fill: colors.text, fontSize: 11 }} />
              <ChartTooltip formatter={(value) => `${formatNumber(Number(value), 1)}%`} /><Area type="monotone" dataKey="ratio" name="Hit ratio" stroke={colors.secondary} fill={colors.secondary} fillOpacity={0.25} /></AreaChart></ResponsiveContainer>}
        </ChartPanel></Grid>
        <Grid item xs={12}>
          <Typography variant="h3" sx={{ mb: 0.75 }}>Semantic threshold</Typography>
          <Typography variant="body2" color="text.secondary">A semantic cache hit is a vector-similarity match, not an exact prompt match. The gateway compares the request embedding with entries in the same isolated namespace and serves a nearby cached response only when its similarity reaches the configured threshold. The configured threshold is {formatNumber(stats.data.similarity_threshold * 100, 1)}% similarity.</Typography>
        </Grid>
        <Grid item xs={12}>
          <Typography variant="h3" sx={{ mb: 1 }}>Entry inspector</Typography>
          {entries.isPending ? <LoadingState label="Loading cached entries" /> : entries.isError ? <ErrorState error={entries.error} onRetry={() => void entries.refetch()} /> : entries.data.items.length === 0 && entries.data.total === 0 ? <EmptyState title="No cached entries" description="No semantic responses are currently cached." /> : <Box sx={{ height: 480, width: '100%' }}>
            <DataGrid rows={entries.data.items} getRowId={(row) => row.key} columns={columns} rowCount={entries.data.total} loading={entries.isFetching}
              paginationMode="server" paginationModel={pageModel} onPaginationModelChange={setPageModel} pageSizeOptions={[25, 50]} disableRowSelectionOnClick />
          </Box>}
          <Typography variant="caption" color="text.secondary">Prompts are redacted by the API; per-entry hit count or prompt may not be recorded.</Typography>
        </Grid>
        <Grid item xs={12}>
          <Typography variant="h3" sx={{ mb: 1 }}>Invalidation</Typography>
          {admin ? <Stack spacing={2}>
            {!stats.data.available && <Alert severity="warning">Redis is unavailable. Cache invalidation may fail until the cache backend is restored.</Alert>}
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-start', flexWrap: 'wrap' }}>
              <TextField size="small" label="Exact Redis entry key" value={entryKey} onChange={(event) => setEntryKey(event.target.value)} sx={{ minWidth: 300, flex: 1 }} helperText="Choose an entry above or paste its exact Redis key." />
              <Button variant="outlined" color="error" disabled={!entryKey.trim() || invalidate.isPending} onClick={() => { setSuccess(null); setConfirmation({ key: entryKey.trim() }); }}>Invalidate one entry</Button>
            </Box>
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-start', flexWrap: 'wrap' }}>
              <TextField size="small" label="Namespace" value={namespace} onChange={(event) => setNamespace(event.target.value)} sx={{ minWidth: 300, flex: 1 }} helperText="A namespace is a 64-character hexadecimal hash; every entry in it will be deleted." />
              <Button variant="outlined" color="error" disabled={!validNamespace || invalidate.isPending} onClick={() => { setSuccess(null); setConfirmation({ namespace: namespace.trim() }); }}>Invalidate namespace</Button>
            </Box>
            <Button variant="contained" color="error" startIcon={<DeleteSweepRounded />} disabled={invalidate.isPending || !stats.data.available} sx={{ alignSelf: 'flex-start' }} onClick={() => { setSuccess(null); setConfirmation({ all_entries: true }); }}>Flush all cache entries</Button>
            {invalidate.isError && <ErrorState error={invalidate.error} />}
            {success && <Alert severity="success" onClose={() => setSuccess(null)}>{success}</Alert>}
          </Stack> : <Alert severity="info">You have read-only viewer access. Cache invalidation controls are available to administrators only.</Alert>}
        </Grid>
      </Grid>
    </>}
    {admin && <ConfirmDialog open={confirmation !== null && !invalidate.isPending}
      title={confirmation && 'all_entries' in confirmation ? 'Flush all semantic cache entries?' : confirmation && 'namespace' in confirmation ? `Invalidate namespace ${confirmation.namespace}?` : 'Invalidate this cache entry?'}
      description={confirmation && 'all_entries' in confirmation ? 'This permanently deletes all semantic response entries (aigw:cache:entry:*) across every namespace. It does not drop the Redis Search index or the separate statistics keys. This cannot be undone.' : confirmation && 'namespace' in confirmation ? `This permanently deletes every cached response in namespace “${confirmation.namespace}”. This cannot be undone.` : `This permanently deletes the exact Redis key “${confirmation && 'key' in confirmation ? confirmation.key : ''}”. This cannot be undone.`}
      confirmLabel={confirmation && 'all_entries' in confirmation ? 'Flush all entries' : confirmation && 'namespace' in confirmation ? 'Invalidate namespace' : 'Invalidate entry'} onClose={() => setConfirmation(null)}
      onConfirm={() => { if (confirmation) invalidate.mutate(confirmation); }} />}
  </>;
}
