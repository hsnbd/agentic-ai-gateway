import { useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Box, Button, Chip, Divider, Drawer, FormControl, Grid, InputLabel, LinearProgress, MenuItem, Select, Stack, TextField, Tooltip, Typography } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import type { GridColDef, GridPaginationModel } from '@mui/x-data-grid';
import { useSearchParams } from 'react-router-dom';
import { useAuth } from '../auth/AuthProvider';
import { ApiError, apiRequest } from '../api/client';
import { useFilteredLogs, useLogDetail } from '../api/hooks/useLogsObservability';
import { ErrorState, EmptyState, JsonViewer, LoadingState, PageHeader } from '../components/Shared';
import { formatMoney, formatNumber, formatTimestamp } from '../features/observability/format';
import type { LogDetail, LogRow } from '../features/observability/types';
import { windowStart } from '../features/observability/format';

const PAGE_SIZES = [25, 50, 100];
const RANGE_VALUES = ['1h', '24h', '7d', '30d', 'all'] as const;
type RangeValue = typeof RANGE_VALUES[number];
function validRange(value: string | null): RangeValue { return RANGE_VALUES.includes(value as RangeValue) ? value as RangeValue : '24h'; }
function positiveNumber(value: string | null, fallback: number): number { const number = Number(value); return Number.isInteger(number) && number > 0 ? number : fallback; }
function stageEntries(detail: LogDetail): Array<{ name: string; milliseconds: number }> {
  return Object.entries(detail.stage_timings).flatMap(([name, raw]) => {
    const milliseconds = typeof raw === 'number' ? raw : typeof raw === 'string' ? Number(raw) : Number.NaN;
    return Number.isFinite(milliseconds) && milliseconds >= 0 ? [{ name, milliseconds }] : [];
  });
}
function statusIsSuccess(status: string): boolean { return ['success', 'succeeded', 'ok'].includes(status.toLowerCase()); }

function LogDetails({ detail }: { detail: LogDetail }) {
  const { user } = useAuth();
  const [revealedDetail, setRevealedDetail] = useState<LogDetail | null>(null);
  const [revealError, setRevealError] = useState<unknown>(null);
  const [revealing, setRevealing] = useState(false);
  const [forbidden, setForbidden] = useState(false);
  const revealController = useRef<AbortController | null>(null);
  useEffect(() => () => revealController.current?.abort(), []);
  useEffect(() => {
    if (user?.role !== 'admin') {
      revealController.current?.abort();
      setRevealedDetail(null);
    }
  }, [user?.role]);
  const revealBodies = async () => {
    if (user?.role !== 'admin' || revealing) return;
    const controller = new AbortController();
    revealController.current = controller;
    setRevealing(true);
    setRevealError(null);
    try {
      const result = await apiRequest<LogDetail>(`/admin/api/logs/${encodeURIComponent(detail.request_id)}?reveal=true`, { signal: controller.signal, cache: 'no-store' });
      if (!controller.signal.aborted) setRevealedDetail(result);
    } catch (error: unknown) {
      if (!controller.signal.aborted) {
        setRevealedDetail(null);
        if (error instanceof ApiError && error.status === 403) setForbidden(true);
        setRevealError(error);
      }
    } finally {
      if (!controller.signal.aborted) setRevealing(false);
    }
  };
  const bodyDetail = user?.role === 'admin' ? revealedDetail ?? detail : detail;
  const timings = stageEntries(detail);
  const maxTiming = Math.max(...timings.map((entry) => entry.milliseconds), 0);
  return <Stack spacing={2.5}>
    <Box><Typography variant="h3">Request {detail.request_id}</Typography>
      <Typography color="text.secondary" variant="body2">{formatTimestamp(detail.created_at).relative} · {formatTimestamp(detail.created_at).absolute}</Typography></Box>
    <Box><Typography variant="h3" sx={{ mb: 1 }}>Stage timings</Typography>
      {timings.length === 0 ? <Typography color="text.secondary">No stage timing details were recorded.</Typography> : <Stack spacing={1.25}>{timings.map((timing) => <Stack key={timing.name} direction="row" spacing={1} alignItems="center">
        <Typography variant="body2" sx={{ width: 130, overflowWrap: 'anywhere' }}>{timing.name}</Typography>
        <LinearProgress variant="determinate" value={maxTiming > 0 ? timing.milliseconds / maxTiming * 100 : 0} sx={{ flex: 1, height: 8, borderRadius: 4 }} />
        <Typography variant="caption" sx={{ minWidth: 70, textAlign: 'right' }}>{formatNumber(timing.milliseconds, 2)} ms</Typography>
      </Stack>)}</Stack>}
    </Box>
    <Divider />
    <Box><Typography variant="h3" sx={{ mb: 1 }}>Routing decision</Typography>
      <Typography variant="body2">Strategy: {detail.routing_strategy ?? 'Not available'}</Typography>
      <Typography variant="body2">Deployment: {detail.deployment_id ?? 'Not available'}</Typography>
      <Typography variant="body2">Chosen provider/model: {detail.provider ?? 'Not available'} · {detail.resolved_model ?? detail.model}</Typography>
      <Typography variant="body2" sx={{ mt: 0.5 }}>Reason: {detail.routing_reason ?? 'Not recorded'}</Typography>
    </Box>
    <Box><Typography variant="h3" sx={{ mb: 1 }}>Cache verdict</Typography>
      <Chip size="small" color={detail.cache_hit ? 'success' : 'default'} label={detail.cache_hit ? 'Semantic cache hit' : 'Cache miss'} />
      <Typography variant="body2" sx={{ mt: 1 }}>Similarity score: {detail.cache_similarity === null ? 'Not recorded' : `${formatNumber(detail.cache_similarity * 100, 1)}%`}</Typography>
    </Box>
    <Box><Typography variant="h3" sx={{ mb: 1 }}>Guardrail results</Typography><JsonViewer value={detail.guardrail_results} /></Box>
    <Box><Typography variant="h3" sx={{ mb: 1 }}>Attempt history</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>{formatNumber(detail.attempt_count)} attempts · {formatNumber(detail.fallback_count)} fallbacks</Typography>
      {detail.attempts.length === 0 ? <Typography variant="body2" color="text.secondary">No per-attempt history was recorded for this request.</Typography> :
        <Box component="ol" sx={{ pl: 3, m: 0, borderLeft: 2, borderColor: 'divider', ml: 1 }}>
          {detail.attempts.map((attempt, index) => <Box component="li" key={`${attempt.deployment_id}-${index}`} sx={{ pl: 1, mb: 2, '&::marker': { color: attempt.outcome === 'success' ? 'success.main' : 'error.main' } }}>
            <Stack direction="row" alignItems="center" spacing={1} flexWrap="wrap">
              <Typography variant="subtitle2">{attempt.provider ?? 'Provider not recorded'} · {attempt.deployment_id}</Typography>
              <Chip size="small" label={attempt.outcome} color={attempt.outcome === 'success' ? 'success' : attempt.outcome === 'error' ? 'error' : 'default'} />
            </Stack>
            <Typography variant="body2" color="text.secondary">Latency: {attempt.latency_ms === null ? 'Not recorded' : `${formatNumber(attempt.latency_ms, 1)} ms`}</Typography>
            {attempt.error && <Typography variant="body2" color="error.main" sx={{ overflowWrap: 'anywhere' }}>Error: {attempt.error}</Typography>}
          </Box>)}
        </Box>}
      {detail.error_code && <Typography variant="body2">Final error code: {detail.error_code}</Typography>}
    </Box>
    <Box><Typography variant="h3" sx={{ mb: 1 }}>Tokens and cost</Typography>
      <Typography variant="body2">Prompt: {formatNumber(detail.prompt_tokens)} tokens · Completion: {formatNumber(detail.completion_tokens)} tokens · Total: {formatNumber(detail.total_tokens)} tokens</Typography>
      <Typography variant="body2">Cost: {formatMoney(detail.cost_usd)} · Latency: {detail.latency_ms === null ? 'Not available' : `${formatNumber(detail.latency_ms, 2)} ms`}</Typography>
      <Typography variant="body2">Status: {detail.status}{detail.status_code === null ? '' : ` (${detail.status_code})`}</Typography>
    </Box>
    <Box>
      <Typography variant="h3" sx={{ mb: 1 }}>Prompt and response bodies</Typography>
      {bodyDetail.body_redacted ? <>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>Bodies are hidden — reveal to view.</Typography>
        {user?.role === 'admin' && !forbidden && <>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>Revealing is recorded in the audit log. Stored content may already be redacted.</Typography>
          <Button variant="outlined" disabled={revealing} onClick={() => void revealBodies()}>{revealing ? 'Revealing…' : 'Reveal bodies'}</Button>
        </>}
      </> : bodyDetail.request_body === null && bodyDetail.response_body === null ?
        <Typography variant="body2" color="text.secondary">No body was stored.</Typography> :
        <Stack spacing={1.5}>
          <Alert severity="warning">Stored content may have been redacted by the gateway.</Alert>
          <Box><Typography variant="subtitle2" sx={{ mb: 0.5 }}>Request body</Typography>{bodyDetail.request_body === null ? <Typography color="text.secondary">No request body was stored.</Typography> : <JsonViewer value={bodyDetail.request_body} />}</Box>
          <Box><Typography variant="subtitle2" sx={{ mb: 0.5 }}>Response body</Typography>{bodyDetail.response_body === null ? <Typography color="text.secondary">No response body was stored.</Typography> : <JsonViewer value={bodyDetail.response_body} />}</Box>
          {revealedDetail && <Button variant="text" onClick={() => setRevealedDetail(null)}>Hide bodies</Button>}
        </Stack>}
      {revealError !== null && <Box sx={{ mt: 1 }}><ErrorState error={revealError} /></Box>}
      {forbidden && <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>Your role no longer permits revealing bodies.</Typography>}
    </Box>
  </Stack>;
}

export default function Logs() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedRequest, setSelectedRequest] = useState<string | null>(null);
  const { user } = useAuth();
  const range = validRange(searchParams.get('range'));
  const pageSize = positiveNumber(searchParams.get('limit'), 25);
  const offset = Math.max(0, Number(searchParams.get('offset')) || 0);
  const filters = useMemo(() => ({
    limit: PAGE_SIZES.includes(pageSize) ? pageSize : 25,
    offset,
    ...(range === 'all' ? {} : { start: windowStart(range), end: new Date().toISOString() }),
    virtual_key_id: searchParams.get('virtual_key_id') || undefined,
    model: searchParams.get('model') || undefined,
    provider: searchParams.get('provider') || undefined,
    status: searchParams.get('status') || undefined,
    cache_hit: searchParams.get('cache_hit') === 'true' ? true : searchParams.get('cache_hit') === 'false' ? false : undefined,
    min_latency_ms: searchParams.has('min_latency_ms') && Number.isFinite(Number(searchParams.get('min_latency_ms'))) ? Number(searchParams.get('min_latency_ms')) : undefined,
    search: searchParams.get('search') || undefined,
  }), [pageSize, offset, range, searchParams]);
  const logs = useFilteredLogs(filters);
  const detail = useLogDetail(selectedRequest);

  useEffect(() => {
    if (!searchParams.has('range')) setSearchParams((current) => { const next = new URLSearchParams(current); next.set('range', '24h'); return next; }, { replace: true });
  }, [searchParams, setSearchParams]);

  const updateParam = (name: string, value: string, resetPage = true) => {
    setSearchParams((current) => {
      const next = new URLSearchParams(current);
      if (value) next.set(name, value); else next.delete(name);
      if (resetPage && name !== 'offset') next.delete('offset');
      return next;
    }, { replace: true });
  };
  const columns: GridColDef<LogRow>[] = [
    { field: 'created_at', headerName: 'Time', minWidth: 175, flex: 1, renderCell: ({ value }) => {
      const formatted = formatTimestamp(String(value));
      return <Tooltip title={formatted.absolute}><span>{formatted.relative}</span></Tooltip>;
    } },
    { field: 'request_id', headerName: 'Request ID', minWidth: 175, flex: 1 },
    { field: 'virtual_key_id', headerName: 'Key', minWidth: 115, flex: 0.8, renderCell: ({ value }) => value ? String(value).slice(0, 10) : 'Unassigned' },
    { field: 'model', headerName: 'Model', minWidth: 125, flex: 1 },
    { field: 'provider', headerName: 'Provider', minWidth: 105, flex: 0.7 },
    { field: 'status', headerName: 'Status', minWidth: 110, renderCell: ({ value }) => <Chip size="small" color={statusIsSuccess(String(value)) ? 'success' : 'error'} label={String(value)} /> },
    { field: 'latency_ms', headerName: 'Latency', minWidth: 100, valueFormatter: (value) => value == null ? 'Not available' : `${formatNumber(Number(value), 1)} ms` },
    { field: 'prompt_tokens', headerName: 'Tokens in', minWidth: 95, valueFormatter: (value) => formatNumber(Number(value)) },
    { field: 'completion_tokens', headerName: 'Tokens out', minWidth: 100, valueFormatter: (value) => formatNumber(Number(value)) },
    { field: 'cost_usd', headerName: 'Cost', minWidth: 95, valueFormatter: (value) => formatMoney(Number(value)) },
    { field: 'cache_hit', headerName: 'Cache', minWidth: 115, renderCell: ({ row }) => <Tooltip title={row.cache_similarity === null ? 'Similarity not recorded' : `Similarity: ${formatNumber(row.cache_similarity * 100, 1)}%`}><span>{row.cache_hit ? 'Hit' : 'Miss'}{row.cache_similarity === null ? '' : ` · ${formatNumber(row.cache_similarity * 100, 0)}%`}</span></Tooltip> },
    { field: 'attempt_count', headerName: 'Attempts / fallbacks', minWidth: 155, sortable: false, renderCell: ({ row }) => row.attempt_count > 1 || row.fallback_count > 0 ? <Chip size="small" color="warning" variant="outlined" label={`${row.attempt_count} attempts · ${row.fallback_count} fallbacks`} /> : `${row.attempt_count} attempt` },
  ];

  const pageModel: GridPaginationModel = { page: Math.floor(offset / filters.limit), pageSize: filters.limit };
  return <>
    <PageHeader title="Request logs" description="Search request-level activity. Results are fetched in bounded server-side pages." />
    <Grid container spacing={1.5} sx={{ mb: 2 }}>
      <Grid item xs={12} sm={4} md={2.4}><FormControl fullWidth size="small"><InputLabel id="log-range-label">Time range</InputLabel><Select labelId="log-range-label" label="Time range" value={range} onChange={(event) => updateParam('range', event.target.value)}>
        <MenuItem value="1h">Last hour</MenuItem><MenuItem value="24h">Last 24 hours</MenuItem><MenuItem value="7d">Last 7 days</MenuItem><MenuItem value="30d">Last 30 days</MenuItem><MenuItem value="all">All time</MenuItem>
      </Select></FormControl></Grid>
      <Grid item xs={12} sm={4} md={2.4}><TextField fullWidth size="small" label="Virtual key ID" value={searchParams.get('virtual_key_id') ?? ''} onChange={(event) => updateParam('virtual_key_id', event.target.value)} /></Grid>
      <Grid item xs={12} sm={4} md={2.4}><TextField fullWidth size="small" label="Model" value={searchParams.get('model') ?? ''} onChange={(event) => updateParam('model', event.target.value)} /></Grid>
      <Grid item xs={12} sm={4} md={2.4}><TextField fullWidth size="small" label="Provider" value={searchParams.get('provider') ?? ''} onChange={(event) => updateParam('provider', event.target.value)} /></Grid>
      <Grid item xs={12} sm={4} md={2.4}><TextField fullWidth size="small" label="Status / outcome" value={searchParams.get('status') ?? ''} onChange={(event) => updateParam('status', event.target.value)} /></Grid>
      <Grid item xs={12} sm={4} md={2.4}><FormControl fullWidth size="small"><InputLabel id="log-cache-label">Cache verdict</InputLabel><Select labelId="log-cache-label" label="Cache verdict" value={searchParams.get('cache_hit') ?? ''} onChange={(event) => updateParam('cache_hit', event.target.value)}>
        <MenuItem value="">Any</MenuItem><MenuItem value="true">Hit</MenuItem><MenuItem value="false">Miss</MenuItem>
      </Select></FormControl></Grid>
      <Grid item xs={12} sm={4} md={2.4}><TextField fullWidth size="small" type="number" label="Min latency (ms)" inputProps={{ min: 0 }} value={searchParams.get('min_latency_ms') ?? ''} onChange={(event) => updateParam('min_latency_ms', event.target.value)} /></Grid>
      <Grid item xs={12} sm={4} md={2.4}><TextField fullWidth size="small" label="Request ID search" value={searchParams.get('search') ?? ''} onChange={(event) => updateParam('search', event.target.value)} /></Grid>
      <Grid item xs={12}><Button onClick={() => setSearchParams({ range: '24h' }, { replace: true })}>Clear filters</Button></Grid>
    </Grid>
    {logs.isPending ? <LoadingState label="Loading request logs" /> : logs.isError ? <ErrorState error={logs.error} onRetry={() => void logs.refetch()} /> : logs.data.items.length === 0 && logs.data.total === 0 ? <EmptyState title="No request logs found" description="Try widening the time range or clearing one or more filters." /> : <Box sx={{ height: 620, width: '100%' }}>
      <DataGrid rows={logs.data.items} columns={columns} rowCount={logs.data.total} loading={logs.isFetching}
        paginationMode="server" paginationModel={pageModel} pageSizeOptions={PAGE_SIZES}
        onPaginationModelChange={(model) => setSearchParams((current) => {
          const next = new URLSearchParams(current);
          next.set('limit', String(model.pageSize));
          if (model.page === 0) next.delete('offset'); else next.set('offset', String(model.page * model.pageSize));
          return next;
        }, { replace: true })}
        onRowClick={(params) => setSelectedRequest(params.row.request_id)} disableRowSelectionOnClick />
    </Box>}
    <Drawer anchor="right" open={selectedRequest !== null} onClose={() => setSelectedRequest(null)} PaperProps={{ sx: { width: { xs: '100%', md: 640 }, p: 3, overflow: 'auto' } }}>
      {detail.isPending ? <LoadingState label="Loading request detail" /> : detail.isError ? <ErrorState error={detail.error} onRetry={() => void detail.refetch()} /> : detail.data ? <LogDetails key={detail.data.request_id} detail={detail.data} /> : <Alert severity="info">Select a request to inspect its details.</Alert>}
      <Button sx={{ mt: 2 }} onClick={() => setSelectedRequest(null)}>Close</Button>
    </Drawer>
    {user?.role === 'viewer' && <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>Prompt and response bodies are available only to administrators.</Typography>}
  </>;
}
