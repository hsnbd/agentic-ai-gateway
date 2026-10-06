import { useState } from 'react';
import { Chip, MenuItem, Paper, Stack, TextField, Typography } from '@mui/material';
import type { GridColDef } from '@mui/x-data-grid';
import { useToolCalls, type ToolCallLog } from '../../api/hooks/useMcpServers';
import { DataTable, ErrorState } from '../../components/Shared';

const STATUSES = ['', 'ok', 'tool_error', 'failed', 'denied', 'blocked', 'invalid', 'unavailable'];

const STATUS_COLOR: Record<string, 'success' | 'warning' | 'error' | 'default'> = {
  ok: 'success',
  tool_error: 'warning',
  invalid: 'warning',
  failed: 'error',
  denied: 'error',
  blocked: 'error',
  unavailable: 'default',
};

const columns: GridColDef<ToolCallLog>[] = [
  { field: 'created_at', headerName: 'Time', width: 170, valueGetter: (_, row) => new Date(row.created_at).toLocaleString() },
  { field: 'tool', headerName: 'Tool', flex: 1, minWidth: 160 },
  { field: 'status', headerName: 'Status', width: 120, renderCell: ({ row }) => <Chip size="small" label={row.status} color={STATUS_COLOR[row.status] ?? 'default'} variant="outlined" /> },
  { field: 'source', headerName: 'Source', width: 90 },
  { field: 'virtual_key_id', headerName: 'Key', width: 120, valueGetter: (_, row) => row.virtual_key_id?.slice(0, 8) ?? 'master' },
  { field: 'duration_ms', headerName: 'Duration', width: 100, valueGetter: (_, row) => `${row.duration_ms.toFixed(0)} ms` },
  { field: 'result_chars', headerName: 'Result', width: 120, valueGetter: (_, row) => `${row.result_chars} chars${row.truncated ? ' (cut)' : ''}` },
  { field: 'guardrail', headerName: 'Guardrail', width: 150, valueGetter: (_, row) => row.guardrail ?? '—' },
  { field: 'error', headerName: 'Detail', flex: 1, minWidth: 180, valueGetter: (_, row) => row.error ?? '' },
];

/** Every MCP tool call the gateway made: who called what, the outcome, and any guardrail. */
export function ToolCallsPanel() {
  const [status, setStatus] = useState('');
  const calls = useToolCalls(status, 100);
  return <Paper variant="outlined" sx={{ mt: 3, p: 2 }}>
    <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ sm: 'center' }} spacing={2} sx={{ mb: 2 }}>
      <Stack>
        <Typography variant="h3">Tool calls</Typography>
        <Typography variant="body2" color="text.secondary">Audit log of MCP tool calls from agent loops and direct calls. Arguments are stored only as a hash.</Typography>
      </Stack>
      <TextField select size="small" label="Status" value={status} onChange={(event) => setStatus(event.target.value)} sx={{ minWidth: 160 }}>
        {STATUSES.map((value) => <MenuItem key={value || 'all'} value={value}>{value || 'All statuses'}</MenuItem>)}
      </TextField>
    </Stack>
    {calls.isError ? <ErrorState error={calls.error} onRetry={() => void calls.refetch()} /> : <DataTable rows={calls.data?.items ?? []} columns={columns} loading={calls.isLoading} />}
  </Paper>;
}
