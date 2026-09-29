import { useMemo, useState } from 'react';
import { Box, Button, FormControl, Grid, InputLabel, LinearProgress, MenuItem, Select, Stack, Typography } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import type { GridColDef, GridPaginationModel } from '@mui/x-data-grid';
import { Area, CartesianGrid, ComposedChart, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import FileDownloadRounded from '@mui/icons-material/FileDownloadRounded';
import { useKeyBudgets, useUsagePage } from '../api/hooks/useUsageObservability';
import { EmptyState, ErrorState, LoadingState, PageHeader } from '../components/Shared';
import { ChartPanel, SectionHeading, UnavailableNote, WindowSelect, useChartColors, type WindowValue } from '../features/observability/Observability';
import { formatMoney, formatNumber, formatWindow } from '../features/observability/format';
import type { KeyBudget, UsagePage, UsageRow } from '../features/observability/types';

const PAGE_SIZES = [25, 50];
const GROUPS: Array<{ value: UsagePage['group_by'] | 'day'; label: string }> = [
  { value: 'key', label: 'Key' }, { value: 'model', label: 'Model' }, { value: 'provider', label: 'Provider' },
  { value: 'team', label: 'Team' }, { value: 'day', label: 'Day' },
];
function csvCell(value: string | number): string { return `"${String(value).replaceAll('"', '""')}"`; }
function exportRows(rows: UsageRow[], group: string) {
  const headings = ['group', 'requests', 'successful requests', 'errors', 'cache hits', 'total tokens', 'cost USD'];
  const lines = [headings, ...rows.map((row) => [row.group, row.requests, row.success_count, row.error_count, row.cache_hit_count, row.total_tokens, row.cost_usd])]
    .map((line) => line.map(csvCell).join(','));
  const blob = new Blob([`\uFEFF${lines.join('\r\n')}`], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `gateway-usage-${group}-${new Date().toISOString().slice(0, 10)}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}
function budgetUtilization(key: KeyBudget): number | undefined {
  return key.max_budget_usd !== null && key.max_budget_usd > 0 ? key.spend_usd / key.max_budget_usd * 100 : undefined;
}

export default function Usage() {
  const [window, setWindow] = useState<WindowValue>('30d');
  const colors = useChartColors();
  const [group, setGroup] = useState<UsagePage['group_by'] | 'day'>('key');
  const [pageModel, setPageModel] = useState<GridPaginationModel>({ page: 0, pageSize: 25 });
  const usage = useUsagePage(group, window, pageModel.pageSize, pageModel.page * pageModel.pageSize);
  const bucket = window === '1h' || window === '24h' ? 'hour' : 'day';
  const trend = useUsagePage(bucket, window, 200, 0);
  const budgets = useKeyBudgets();
  const timeseries = useMemo(() => trend.data?.rows.map((row) => ({ ...row, label: row.group })) ?? [], [trend.data]);
  const columns: GridColDef<UsageRow>[] = [
    { field: 'group', headerName: group === 'key' ? 'Key ID' : group === 'team' ? 'Team ID' : group, flex: 1, minWidth: 180 },
    { field: 'requests', headerName: 'Requests', type: 'number', minWidth: 110, valueFormatter: (value) => formatNumber(Number(value)) },
    { field: 'success_count', headerName: 'Successes', type: 'number', minWidth: 110, valueFormatter: (value) => formatNumber(Number(value)) },
    { field: 'error_count', headerName: 'Errors', type: 'number', minWidth: 90, valueFormatter: (value) => formatNumber(Number(value)) },
    { field: 'cache_hit_count', headerName: 'Cache hits', type: 'number', minWidth: 110, valueFormatter: (value) => formatNumber(Number(value)) },
    { field: 'total_tokens', headerName: 'Tokens', type: 'number', minWidth: 110, valueFormatter: (value) => formatNumber(Number(value)) },
    { field: 'cost_usd', headerName: 'Spend', type: 'number', minWidth: 115, valueFormatter: (value) => formatMoney(Number(value)) },
  ];
  const usageRows = (usage.data?.rows ?? []).map((row, index) => ({ ...row, id: `${row.group}-${index}` }));
  const budgetRows = useMemo(() => [...(budgets.data?.items ?? [])].sort((a, b) => (budgetUtilization(b) ?? -1) - (budgetUtilization(a) ?? -1)), [budgets.data]);

  return <>
    <PageHeader title="Usage and costs" description="Inspect spend and token usage by dimension and report key-budget utilization." action={<WindowSelect value={window} onChange={(next) => { setWindow(next); setPageModel((current) => ({ ...current, page: 0 })); }} />} />
    <Grid container spacing={2}>
      <Grid item xs={12}><ChartPanel title="Spend and tokens over time" description={`Spend (USD) and tokens · ${formatWindow(window)} · ${bucket} buckets`}>
        {trend.isPending ? <LoadingState label="Loading spend and token series" /> : trend.isError ? <ErrorState error={trend.error} onRetry={() => void trend.refetch()} /> : timeseries.length === 0 ? <EmptyState title="No usage series available" /> :
          <ResponsiveContainer width="100%" height="100%"><ComposedChart data={timeseries}>
            <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
            <XAxis dataKey="label" tick={{ fill: colors.text, fontSize: 11 }} />
            <YAxis yAxisId="spend" tick={{ fill: colors.text, fontSize: 11 }} tickFormatter={(value: number) => formatMoney(value)} />
            <YAxis yAxisId="tokens" orientation="right" tick={{ fill: colors.text, fontSize: 11 }} />
            <Tooltip formatter={(value, name) => name === 'Spend' ? formatMoney(Number(value)) : formatNumber(Number(value))} /><Legend />
            <Area yAxisId="spend" type="monotone" dataKey="cost_usd" name="Spend" stackId="spend" stroke={colors.primary} fill={colors.primary} fillOpacity={0.25} />
            <Area yAxisId="tokens" type="monotone" dataKey="total_tokens" name="Tokens" stackId="tokens" stroke={colors.secondary} fill={colors.secondary} fillOpacity={0.25} />
          </ComposedChart></ResponsiveContainer>}
      </ChartPanel></Grid>
      <Grid item xs={12}>
        <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ xs: 'stretch', sm: 'center' }} spacing={1.5} sx={{ mb: 1.5 }}>
          <Box><SectionHeading title="Usage breakdown" subtitle={`Aggregated results for ${formatWindow(window)}.`} /></Box>
          <Stack direction="row" spacing={1}>
            <FormControl size="small" sx={{ minWidth: 190 }}><InputLabel id="usage-group-label">Group by</InputLabel><Select labelId="usage-group-label" label="Group by" value={group} onChange={(event) => { setGroup(event.target.value as typeof group); setPageModel((current) => ({ ...current, page: 0 })); }}>
              {GROUPS.map((item) => <MenuItem key={item.value} value={item.value} >{item.label}</MenuItem>)}
            </Select></FormControl>
            <Button variant="outlined" startIcon={<FileDownloadRounded />} disabled={!usage.data?.rows.length} onClick={() => usage.data && exportRows(usage.data.rows, group)}>Export CSV</Button>
          </Stack>
        </Stack>
        {usage.isPending ? <LoadingState label="Loading usage aggregates" /> : usage.isError ? <ErrorState error={usage.error} onRetry={() => void usage.refetch()} /> : usage.data.rows.length === 0 ? <EmptyState title="No usage data" description="There is no aggregate usage for this window." /> : <Box sx={{ height: 560 }}>
          <DataGrid rows={usageRows} columns={columns} rowCount={usage.data.total} loading={usage.isFetching} paginationMode="server" paginationModel={pageModel} onPaginationModelChange={setPageModel} pageSizeOptions={PAGE_SIZES} disableRowSelectionOnClick />
        </Box>}
      </Grid>
      <Grid item xs={12}>
        <SectionHeading title="Key budgets" subtitle="Spend and budget values as reported by the key API. Utilization above 80% is highlighted." />
        {budgets.isPending ? <LoadingState label="Loading key budgets" /> : budgets.isError ? <ErrorState error={budgets.error} onRetry={() => void budgets.refetch()} /> : budgetRows.length === 0 ? <EmptyState title="No virtual keys" description="Key budgets will appear here when keys are configured." /> : <Stack spacing={1.25}>
          {budgetRows.map((key) => {
            const utilization = budgetUtilization(key);
            const clamped = utilization === undefined ? 0 : Math.min(Math.max(utilization, 0), 100);
            return <Box key={key.id} sx={{ p: 1.5, border: 1, borderColor: utilization !== undefined && utilization > 80 ? 'warning.main' : 'divider', borderRadius: 2 }}>
              <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" spacing={0.5} sx={{ mb: 0.75 }}>
                <Typography variant="body2" fontWeight={600}>{key.name} <Typography component="span" variant="caption" color="text.secondary">({key.key_prefix})</Typography></Typography>
                <Typography variant="body2" color={utilization !== undefined && utilization > 80 ? 'warning.main' : 'text.secondary'}>
                  {formatMoney(key.spend_usd)} spent of {key.max_budget_usd === null ? 'No budget configured' : `${formatMoney(key.max_budget_usd)} ${key.budget_duration} budget`}
                  {utilization === undefined ? ' · utilization not available' : ` · ${formatNumber(utilization, 1)}% utilized`}
                </Typography>
              </Stack>
              {utilization === undefined ? <Typography variant="caption" color="text.secondary">A configured non-zero budget is required to calculate utilization.</Typography> : <LinearProgress variant="determinate" value={clamped} color={utilization > 80 ? 'warning' : 'primary'} />}
            </Box>;
          })}
          {budgets.data.total > budgets.data.items.length && <UnavailableNote>Showing {budgets.data.items.length} of {budgets.data.total} keys; additional keys are not included in this page.</UnavailableNote>}
        </Stack>}
      </Grid>
    </Grid>
  </>;
}
