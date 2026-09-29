import { useMemo, useState } from 'react';
import { Box, Chip, Grid, Stack, Typography } from '@mui/material';
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Area, AreaChart } from 'recharts';
import { useMetricSeries, useDeployments, useUsagePage, useDashboardOverview } from '../api/hooks/useUsageObservability';
import { ErrorState, LoadingState, PageHeader, StatCard, EmptyState } from '../components/Shared';
import { ChartPanel, MetricStatCard, UnavailableChart, UnavailableNote, useChartColors, WindowSelect, type WindowValue } from '../features/observability/Observability';
import { formatMoney, formatNumber, formatWindow } from '../features/observability/format';
import type { TimeSeriesPoint, UsageRow } from '../features/observability/types';

function stampLabel(timestamp: string): string {
  const date = new Date(timestamp);
  return Number.isFinite(date.getTime()) ? new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(date) : timestamp;
}

function mergeSeries(first: TimeSeriesPoint[] | undefined, second: TimeSeriesPoint[] | undefined, firstName: string, secondName: string) {
  const values = new Map<string, { timestamp: string; [key: string]: number | string }>();
  for (const point of first ?? []) values.set(point.timestamp, { timestamp: point.timestamp, [firstName]: point.value });
  for (const point of second ?? []) {
    const existing = values.get(point.timestamp) ?? { timestamp: point.timestamp };
    existing[secondName] = point.value;
    values.set(point.timestamp, existing);
  }
  return [...values.values()].sort((a, b) => a.timestamp.localeCompare(b.timestamp)).map((point) => ({ ...point, label: stampLabel(point.timestamp) }));
}

function Breakdown({ title, rows, loading, error, retry, color, window }: { title: string; rows: UsageRow[] | undefined; loading: boolean; error: unknown; retry: () => void; color: string; window: string }) {
  const chartRows = useMemo(() => [...(rows ?? [])].sort((a, b) => b.requests - a.requests).slice(0, 8), [rows]);
  const chartColors = useChartColors();
  return <ChartPanel title={title} description={`Requests per group in ${formatWindow(window)}`}>
    {loading ? <LoadingState label="Loading breakdown" /> : error ? <ErrorState error={error} onRetry={retry} /> : chartRows.length === 0 ? <EmptyState title="No breakdown data" /> :
      <ResponsiveContainer width="100%" height="100%"><BarChart data={chartRows} layout="vertical" margin={{ left: 10, right: 18 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} horizontal={false} />
        <XAxis type="number" tick={{ fill: chartColors.text, fontSize: 11 }} />
        <YAxis type="category" dataKey="group" width={105} tick={{ fill: chartColors.text, fontSize: 11 }} />
        <Tooltip /><Bar dataKey="requests" fill={color} radius={[0, 5, 5, 0]} />
      </BarChart></ResponsiveContainer>}
  </ChartPanel>;
}

export default function Dashboard() {
  const [window, setWindow] = useState<WindowValue>('24h');
  const colors = useChartColors();
  const summary = useDashboardOverview(window);
  const requests = useMetricSeries(window, 'requests');
  const successes = useMetricSeries(window, 'successes');
  const errors = useMetricSeries(window, 'errors');
  const bucket = window === '1h' || window === '24h' ? 'hour' : 'day';
  const trend = useUsagePage(bucket, window, 200, 0);
  const modelRows = useUsagePage('model', window, 200, 0);
  const providerRows = useUsagePage('provider', window, 200, 0);
  const keyRows = useUsagePage('key', window, 200, 0);
  const deployments = useDeployments();
  const requestData = useMemo(() => mergeSeries(requests.data?.points, successes.data?.points, 'requests', 'successes').map((point) => {
    const matchingError = errors.data?.points.find((errorPoint) => errorPoint.timestamp === point.timestamp);
    return { ...point, errors: matchingError?.value ?? 0 };
  }), [requests.data, successes.data, errors.data]);
  const trendData = useMemo(() => trend.data?.rows.map((row) => ({ ...row, label: row.group })) ?? [], [trend.data]);
  const requestsError = requests.error ?? successes.error ?? errors.error;

  return <>
    <PageHeader title="Dashboard" description="Gateway traffic, performance, spend, and provider health." action={<WindowSelect value={window} onChange={setWindow} />} />
    {summary.isPending ? <LoadingState label="Loading gateway metrics" /> : summary.isError ? <ErrorState error={summary.error} onRetry={() => void summary.refetch()} /> : <>
      <Grid container spacing={2}>
        {([
          { label: 'Request success rate', value: `${(summary.data.success_rate * 100).toFixed(1)}%`, current: summary.data.success_rate, previous: summary.data.previous_success_rate, favorable: 'up', unit: 'percentage points', format: formatNumber },
          { label: 'p50 latency', value: `${formatNumber(summary.data.p50_latency_ms, 1)} ms`, current: summary.data.p50_latency_ms, previous: summary.data.previous_p50_latency_ms, favorable: 'down', unit: 'ms', format: (value: number) => formatNumber(value, 1) },
          { label: 'p95 latency', value: `${formatNumber(summary.data.p95_latency_ms, 1)} ms`, current: summary.data.p95_latency_ms, previous: summary.data.previous_p95_latency_ms, favorable: 'down', unit: 'ms', format: (value: number) => formatNumber(value, 1) },
          { label: 'Cache hit ratio', value: `${(summary.data.cache_hit_ratio * 100).toFixed(1)}%`, current: summary.data.cache_hit_ratio, previous: summary.data.previous_cache_hit_ratio, favorable: 'up', unit: 'percentage points', format: formatNumber },
          { label: 'Total spend', value: formatMoney(summary.data.total_cost_usd), current: summary.data.total_cost_usd, previous: summary.data.previous_total_cost_usd, favorable: 'down', format: formatMoney },
          { label: 'Total tokens', value: formatNumber(summary.data.total_tokens), current: summary.data.total_tokens, previous: summary.data.previous_total_tokens, favorable: 'neutral', format: formatNumber },
          { label: 'Failover count', value: formatNumber(summary.data.fallback_count), current: summary.data.fallback_count, previous: summary.data.previous_fallback_count, favorable: 'down', format: formatNumber },
        ] as const).map((metric) => <Grid item xs={12} sm={6} lg={3} key={metric.label}>
          <MetricStatCard label={metric.label} value={metric.value} hint={formatWindow(window)} current={metric.current} previous={metric.previous} unit={'unit' in metric ? metric.unit : undefined} favorable={metric.favorable} format={metric.format} />
        </Grid>)}
        <Grid item xs={12} sm={6} lg={3}><StatCard label="Active requests" value={formatNumber(summary.data.active_requests)} hint="Current in-flight requests · live gauge" /></Grid>
      </Grid>
      <Grid container spacing={2} sx={{ mt: 0.25 }}>
        <Grid item xs={12} lg={6}><ChartPanel title="Requests by status" description={`Requests, successes, and errors · ${formatWindow(window)}`}>
          {requests.isPending || successes.isPending || errors.isPending ? <LoadingState label="Loading request series" /> : requestsError ? <ErrorState error={requestsError} onRetry={() => { void requests.refetch(); void successes.refetch(); void errors.refetch(); }} /> : requestData.length === 0 ? <EmptyState title="No request data" /> :
            <ResponsiveContainer width="100%" height="100%"><AreaChart data={requestData}><CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
              <XAxis dataKey="label" tick={{ fill: colors.text, fontSize: 11 }} /><YAxis tick={{ fill: colors.text, fontSize: 11 }} />
              <Tooltip /><Legend /><Area type="monotone" dataKey="successes" stackId="status" stroke={colors.success} fill={colors.success} fillOpacity={0.32} />
              <Area type="monotone" dataKey="errors" stackId="status" stroke={colors.error} fill={colors.error} fillOpacity={0.32} /></AreaChart></ResponsiveContainer>}
        </ChartPanel></Grid>
        <Grid item xs={12} lg={6}><ChartPanel title="Latency percentiles" description={`Latency percentiles over ${formatWindow(window)}`}>
          <UnavailableChart message="The API provides p50/p95 only for the whole selected window, not as a time series." />
        </ChartPanel></Grid>
        <Grid item xs={12} lg={6}><ChartPanel title="Spend over time" description={`${formatWindow(window)} · ${bucket} buckets`}>
          {trend.isPending ? <LoadingState /> : trend.isError ? <ErrorState error={trend.error} onRetry={() => void trend.refetch()} /> : trendData.length === 0 ? <EmptyState title="No spend data" /> :
            <ResponsiveContainer width="100%" height="100%"><LineChart data={trendData}><CartesianGrid strokeDasharray="3 3" stroke={colors.grid} /><XAxis dataKey="label" tick={{ fill: colors.text, fontSize: 11 }} /><YAxis tick={{ fill: colors.text, fontSize: 11 }} tickFormatter={(value: number) => formatMoney(value)} /><Tooltip formatter={(value) => formatMoney(Number(value))} />
              <Line type="monotone" dataKey="cost_usd" name="Spend" stroke={colors.primary} strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer>}
        </ChartPanel></Grid>
        <Grid item xs={12} lg={6}><ChartPanel title="Tokens over time" description={`${formatWindow(window)} · ${bucket} buckets`}>
          {trend.isPending ? <LoadingState /> : trend.isError ? <ErrorState error={trend.error} onRetry={() => void trend.refetch()} /> : trendData.length === 0 ? <EmptyState title="No token data" /> :
            <ResponsiveContainer width="100%" height="100%"><LineChart data={trendData}><CartesianGrid strokeDasharray="3 3" stroke={colors.grid} /><XAxis dataKey="label" tick={{ fill: colors.text, fontSize: 11 }} /><YAxis tick={{ fill: colors.text, fontSize: 11 }} /><Tooltip formatter={(value) => formatNumber(Number(value))} />
              <Line type="monotone" dataKey="total_tokens" name="Tokens" stroke={colors.secondary} strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer>}
        </ChartPanel></Grid>
        <Grid item xs={12} lg={4}><Breakdown title="Requests by model" rows={modelRows.data?.rows} loading={modelRows.isPending} error={modelRows.error} retry={() => void modelRows.refetch()} color={colors.primary} window={window} /></Grid>
        <Grid item xs={12} lg={4}><Breakdown title="Requests by provider" rows={providerRows.data?.rows} loading={providerRows.isPending} error={providerRows.error} retry={() => void providerRows.refetch()} color={colors.secondary} window={window} /></Grid>
        <Grid item xs={12} lg={4}><Breakdown title="Requests by key" rows={keyRows.data?.rows} loading={keyRows.isPending} error={keyRows.error} retry={() => void keyRows.refetch()} color={colors.warning} window={window} /></Grid>
      </Grid>
      <Box sx={{ mt: 3 }}>
        <Typography variant="h3" sx={{ mb: 1.5 }}>Provider health</Typography>
        {deployments.isPending ? <LoadingState label="Loading provider health" /> : deployments.isError ? <ErrorState error={deployments.error} onRetry={() => void deployments.refetch()} /> : deployments.data.items.length === 0 ? <EmptyState title="No deployments configured" /> :
          <Stack direction="row" useFlexGap flexWrap="wrap" gap={1}>{deployments.data.items.map((deployment) => {
            const state = deployment.health_state.toLowerCase();
            const color = state === 'closed' ? 'success' : state === 'open' ? 'error' : 'warning';
            return <Chip key={deployment.id} color={color} variant="outlined" label={`${deployment.provider} · ${deployment.model} · ${deployment.health_state}`} title={`${deployment.id} · ${deployment.consecutive_failures} consecutive failures`} />;
          })}</Stack>}
        {deployments.data && deployments.data.total > deployments.data.items.length && <UnavailableNote>Showing {deployments.data.items.length} of {deployments.data.total} deployments.</UnavailableNote>}
      </Box>
    </>}
  </>;
}
