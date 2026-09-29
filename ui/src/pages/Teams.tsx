import { useState } from 'react';
import { Button, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, MenuItem, Stack, TextField, Typography } from '@mui/material';
import AddRounded from '@mui/icons-material/AddRounded';
import DeleteOutlineRounded from '@mui/icons-material/DeleteOutlineRounded';
import EditRounded from '@mui/icons-material/EditRounded';
import InsightsRounded from '@mui/icons-material/InsightsRounded';
import type { GridColDef } from '@mui/x-data-grid';
import { ConfirmDialog, DataTable, EmptyState, ErrorState, LoadingState, PageHeader } from '../components/Shared';
import { useAuth } from '../auth/AuthProvider';
import { useCreateTeam, useDeleteTeam, useTeamUsage, useTeams, useUpdateTeam } from '../api/hooks/useTeams';
import { BudgetBar, formatDate, useAdminMutationFeedback } from '../features/admin/AdminPrimitives';
import type { Team, TeamCreateRequest } from '../api/types';

const PERIODS = ['daily', 'weekly', 'monthly'];
const emptyForm: TeamCreateRequest = { name: '', description: '', max_budget_usd: null, budget_period: 'monthly' };

/** Teams group virtual keys under a shared budget. Viewers can browse; admins manage. */
export default function Teams() {
  const canMutate = useAuth().user?.role === 'admin';
  const teams = useTeams(); const create = useCreateTeam(); const update = useUpdateTeam(); const remove = useDeleteTeam(); const feedback = useAdminMutationFeedback();
  const [editing, setEditing] = useState<{ id: string | null; form: TeamCreateRequest } | null>(null);
  const [deleting, setDeleting] = useState<Team | null>(null);
  const [usageFor, setUsageFor] = useState<Team | null>(null);
  const save = () => {
    if (!editing) return;
    const body = { ...editing.form, description: editing.form.description || null };
    const done = { onSuccess: () => { feedback.notify(editing.id ? 'Team updated.' : `Team ${body.name} created.`); setEditing(null); }, onError: feedback.fail };
    if (editing.id) update.mutate({ id: editing.id, body }, done); else create.mutate(body, done);
  };
  const columns: GridColDef<Team>[] = [
    { field: 'name', headerName: 'Name', flex: 1, minWidth: 160 },
    { field: 'description', headerName: 'Description', flex: 1, minWidth: 180, valueGetter: (_, row) => row.description || '—' },
    { field: 'budget', headerName: 'Budget / spend', minWidth: 180, flex: 1, renderCell: ({ row }) => <BudgetBar spent={row.spend_usd} budget={row.max_budget_usd} /> },
    { field: 'budget_period', headerName: 'Period', width: 110 },
    { field: 'budget_reset_at', headerName: 'Next reset', width: 170, valueGetter: (_, row) => row.budget_reset_at ? formatDate(row.budget_reset_at) : '—' },
    { field: 'actions', headerName: 'Actions', width: 170, sortable: false, renderCell: ({ row }) => <Stack direction="row">
      <IconButton aria-label={`Usage for ${row.name}`} onClick={() => setUsageFor(row)}><InsightsRounded fontSize="small" /></IconButton>
      {canMutate && <IconButton aria-label={`Edit ${row.name}`} onClick={() => setEditing({ id: row.id, form: { name: row.name, description: row.description ?? '', max_budget_usd: row.max_budget_usd, budget_period: row.budget_period } })}><EditRounded fontSize="small" /></IconButton>}
      {canMutate && <IconButton aria-label={`Delete ${row.name}`} color="error" onClick={() => setDeleting(row)}><DeleteOutlineRounded fontSize="small" /></IconButton>}
    </Stack> },
  ];
  const header = <PageHeader title="Teams" description="Group virtual keys under a shared budget and see their combined usage." action={canMutate ? <Button variant="contained" startIcon={<AddRounded />} onClick={() => setEditing({ id: null, form: emptyForm })}>Create team</Button> : undefined} />;
  if (teams.isLoading) return <LoadingState label="Loading teams" />;
  if (teams.isError) return <ErrorState error={teams.error} onRetry={() => void teams.refetch()} />;
  return <>
    {header}
    {teams.data?.items.length ? <DataTable rows={teams.data.items} columns={columns} rowCount={teams.data.total} loading={teams.isFetching} /> : <EmptyState title="No teams yet" description={canMutate ? 'Create a team, then assign keys to it from the Keys page.' : 'An administrator has not created any teams.'} />}
    <TeamDialog value={editing} pending={create.isPending || update.isPending} onChange={setEditing} onClose={() => setEditing(null)} onSave={save} />
    <UsageDialog team={usageFor} onClose={() => setUsageFor(null)} />
    <ConfirmDialog open={Boolean(deleting)} title="Delete team?" description={`${deleting?.name ?? ''} will be removed. Its keys keep working but no longer share its budget.`} confirmLabel="Delete team" onClose={() => setDeleting(null)} onConfirm={() => { const target = deleting; setDeleting(null); if (target) remove.mutate(target.id, { onSuccess: () => feedback.notify(`Deleted ${target.name}.`), onError: feedback.fail }); }} />
    {feedback.node}
  </>;
}

function TeamDialog({ value, pending, onChange, onClose, onSave }: { value: { id: string | null; form: TeamCreateRequest } | null; pending: boolean; onChange: (value: { id: string | null; form: TeamCreateRequest }) => void; onClose: () => void; onSave: () => void }) {
  const form = value?.form ?? emptyForm;
  const set = (patch: Partial<TeamCreateRequest>) => value && onChange({ ...value, form: { ...form, ...patch } });
  return <Dialog open={Boolean(value)} onClose={onClose} fullWidth maxWidth="xs"><DialogTitle>{value?.id ? 'Edit team' : 'Create team'}</DialogTitle><DialogContent><Stack spacing={2} sx={{ pt: 1 }}>
    <TextField label="Team name" required value={form.name} onChange={(event) => set({ name: event.target.value })} />
    <TextField label="Description" value={form.description ?? ''} onChange={(event) => set({ description: event.target.value })} />
    <TextField label="Max budget (USD)" type="number" value={form.max_budget_usd ?? ''} helperText="Leave empty for no team budget" onChange={(event) => set({ max_budget_usd: event.target.value ? Number(event.target.value) : null })} />
    <TextField label="Budget period" select value={form.budget_period ?? 'monthly'} onChange={(event) => set({ budget_period: event.target.value })}>{PERIODS.map((period) => <MenuItem key={period} value={period}>{period}</MenuItem>)}</TextField>
  </Stack></DialogContent><DialogActions><Button onClick={onClose}>Cancel</Button><Button variant="contained" disabled={pending || !form.name.trim()} onClick={onSave}>{value?.id ? 'Save team' : 'Create team'}</Button></DialogActions></Dialog>;
}

function UsageDialog({ team, onClose }: { team: Team | null; onClose: () => void }) {
  const usage = useTeamUsage(team?.id);
  return <Dialog open={Boolean(team)} onClose={onClose} fullWidth maxWidth="xs"><DialogTitle>{team?.name} · last 30 days</DialogTitle><DialogContent>
    {usage.isLoading ? <LoadingState label="Loading usage" /> : usage.isError ? <ErrorState error={usage.error} /> : <Stack spacing={1}>
      <Typography><strong>Requests:</strong> {usage.data?.requests ?? 0} ({usage.data?.error_count ?? 0} errors)</Typography>
      <Typography><strong>Tokens:</strong> {(usage.data?.total_tokens ?? 0).toLocaleString()}</Typography>
      <Typography><strong>Cost:</strong> ${(usage.data?.cost_usd ?? 0).toFixed(6)}</Typography>
    </Stack>}
  </DialogContent><DialogActions><Button onClick={onClose}>Close</Button></DialogActions></Dialog>;
}
