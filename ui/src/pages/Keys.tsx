import { useId, useMemo, useState } from 'react';
import { Button, Dialog, DialogActions, DialogContent, DialogTitle, Divider, FormControl, FormControlLabel, FormHelperText, IconButton, InputLabel, MenuItem, Select, Stack, Switch, TextField, Typography } from '@mui/material';
import DeleteOutlineRounded from '@mui/icons-material/DeleteOutlineRounded';
import EditRounded from '@mui/icons-material/EditRounded';
import AddRounded from '@mui/icons-material/AddRounded';
import AutorenewRounded from '@mui/icons-material/AutorenewRounded';
import { DataTable, EmptyState, ErrorState, PageHeader, LoadingState, ConfirmDialog, CopyButton } from '../components/Shared';
import { useAuth } from '../auth/AuthProvider';
import { useKeys } from '../api/hooks/useKeys';
import { useAdminModels } from '../api/hooks/useModelsAdmin';
import { useTeams } from '../api/hooks/useTeams';
import { useMcpServers } from '../api/hooks/useMcpServers';
import { useGuardrailPolicies } from '../api/hooks/useGuardrails';
import { useCreateKey, useDeleteKey, useRegenerateKey, useUpdateKey } from '../api/hooks/useKeysAdmin';
import { BudgetBar, formatDate, useAdminMutationFeedback } from '../features/admin/AdminPrimitives';
import type { KeyCreateRequest, VirtualKey } from '../api/types';
import type { GridColDef } from '@mui/x-data-grid';

/** Every setting a key has; the create and edit dialogs share it, so they cannot drift apart. */
type KeyForm = Required<Pick<KeyCreateRequest, 'name' | 'team_id' | 'allowed_models' | 'allowed_mcp_servers' | 'allowed_tools' | 'max_budget_usd' | 'budget_duration' | 'rpm_limit' | 'tpm_limit' | 'max_parallel_requests' | 'guardrail_policy' | 'allowed_routes' | 'expires_at' | 'enabled'>>;

const emptyForm: KeyForm = { name: '', team_id: null, allowed_models: [], allowed_mcp_servers: [], allowed_tools: [], max_budget_usd: null, budget_duration: 'monthly', rpm_limit: null, tpm_limit: null, max_parallel_requests: null, guardrail_policy: null, allowed_routes: [], expires_at: null, enabled: true };

const BUDGET_PERIODS = ['daily', 'weekly', 'monthly'];

function formFromKey(key: VirtualKey): KeyForm {
  return { name: key.name, team_id: key.team_id, allowed_models: key.allowed_models, allowed_mcp_servers: key.allowed_mcp_servers ?? [], allowed_tools: key.allowed_tools ?? [], max_budget_usd: key.max_budget_usd, budget_duration: key.budget_duration, rpm_limit: key.rpm_limit, tpm_limit: key.tpm_limit, max_parallel_requests: key.max_parallel_requests, guardrail_policy: key.guardrail_policy, allowed_routes: key.allowed_routes, expires_at: key.expires_at, enabled: key.enabled };
}

export default function Keys() {
  const { user } = useAuth(); const canMutate = user?.role === 'admin';
  const [page, setPage] = useState(0); const [createOpen, setCreateOpen] = useState(false); const [form, setForm] = useState<KeyForm>(emptyForm);
  const [edit, setEdit] = useState<{ id: string; form: KeyForm } | null>(null);
  const [confirm, setConfirm] = useState<{ kind: 'delete' | 'regenerate'; key: VirtualKey } | null>(null); const [plaintext, setPlaintext] = useState<string | null>(null);
  const keys = useKeys(25, page * 25); const models = useAdminModels(); const teams = useTeams(); const mcpServers = useMcpServers(); const policies = useGuardrailPolicies();
  const create = useCreateKey(); const update = useUpdateKey(); const regenerate = useRegenerateKey(); const remove = useDeleteKey(); const feedback = useAdminMutationFeedback();
  const teamName = useMemo(() => new Map((teams.data?.items ?? []).map((team) => [team.id, team.name])), [teams.data]);
  const options: KeyFormOptions = { models: models.data?.items.map((model) => model.name) ?? [], teams: teams.data?.items ?? [], mcpServers: mcpServers.data ?? [], policies: policies.data?.items.map((policy) => policy.name) ?? [] };
  const submitCreate = () => { create.mutate(form, { onSuccess: (key) => { setCreateOpen(false); setForm(emptyForm); setPlaintext(key.key ?? null); feedback.notify('Virtual key created.'); }, onError: feedback.fail }); };
  const submitEdit = () => { if (!edit) return; update.mutate({ id: edit.id, body: edit.form }, { onSuccess: () => { setEdit(null); feedback.notify('Key updated.'); }, onError: feedback.fail }); };
  const runConfirm = () => { if (!confirm) return; if (confirm.kind === 'delete') { remove.mutate(confirm.key.id, { onSuccess: () => { setConfirm(null); feedback.notify('Key deleted.'); }, onError: feedback.fail }); } else { regenerate.mutate(confirm.key.id, { onSuccess: (key) => { setConfirm(null); setPlaintext(key.key ?? null); feedback.notify('Key regenerated.'); }, onError: feedback.fail }); } };
  const columns: GridColDef<VirtualKey>[] = [{ field: 'name', headerName: 'Name', flex: 1, minWidth: 160 }, { field: 'key_prefix', headerName: 'Key prefix', width: 130 }, { field: 'team_id', headerName: 'Team', width: 130, valueGetter: (_, row) => row.team_id ? teamName.get(row.team_id) ?? row.team_id : '—' }, { field: 'allowed_models', headerName: 'Models allowed', flex: 1, minWidth: 160, valueGetter: (_, row) => row.allowed_models.length ? row.allowed_models.join(', ') : 'All models' }, { field: 'budget', headerName: 'Budget / spend', minWidth: 180, flex: 1, renderCell: ({ row }) => <BudgetBar spent={row.spend_usd} budget={row.max_budget_usd} /> }, { field: 'limits', headerName: 'RPM / TPM', width: 120, valueGetter: (_, row) => `${row.rpm_limit ?? '∞'} / ${row.tpm_limit ?? '∞'}` }, { field: 'status', headerName: 'Status', width: 105, renderCell: ({ row }) => row.enabled ? 'Active' : 'Disabled' }, { field: 'created_at', headerName: 'Created', width: 160, valueGetter: (_, row) => formatDate(row.created_at) }, { field: 'last_used_at', headerName: 'Last used', width: 160, valueGetter: (_, row) => formatDate(row.last_used_at) }, { field: 'actions', headerName: 'Actions', width: 180, sortable: false, renderCell: ({ row }) => canMutate ? <Stack direction="row"><IconButton aria-label="Edit key" onClick={() => setEdit({ id: row.id, form: formFromKey(row) })}><EditRounded fontSize="small" /></IconButton><IconButton aria-label="Regenerate key" onClick={() => setConfirm({ kind: 'regenerate', key: row })}><AutorenewRounded fontSize="small" /></IconButton><IconButton aria-label={row.enabled ? 'Disable key' : 'Enable key'} onClick={() => update.mutate({ id: row.id, body: { enabled: !row.enabled } }, { onSuccess: () => feedback.notify(row.enabled ? 'Key disabled.' : 'Key enabled.'), onError: feedback.fail })}><Typography variant="caption">{row.enabled ? 'Disable' : 'Enable'}</Typography></IconButton><IconButton aria-label="Delete key" color="error" onClick={() => setConfirm({ kind: 'delete', key: row })}><DeleteOutlineRounded fontSize="small" /></IconButton></Stack> : null }];
  if (keys.isLoading || models.isLoading || teams.isLoading) return <LoadingState label="Loading keys" />;
  if (keys.isError) return <ErrorState error={keys.error} onRetry={() => void keys.refetch()} />;
  return <>
    <PageHeader title="Virtual keys" description="Manage budgets, access, and lifecycle for gateway credentials." action={canMutate ? <Button variant="contained" startIcon={<AddRounded />} onClick={() => setCreateOpen(true)}>Create key</Button> : undefined} />
    {keys.data?.items.length ? <><DataTable rows={keys.data.items} columns={columns} rowCount={keys.data.total} loading={keys.isFetching} /><Stack direction="row" justifyContent="flex-end"><Button disabled={page === 0} onClick={() => setPage((value) => value - 1)}>Previous</Button><Button disabled={(page + 1) * 25 >= keys.data.total} onClick={() => setPage((value) => value + 1)}>Next</Button></Stack></> : <EmptyState title="No virtual keys" description="Create a key to start issuing scoped gateway credentials." />}
    {/* Both dialogs render in every state: with no keys yet, "Create key" must still open one. Keys remount them per opening. */}
    <KeyDialog key={createOpen ? 'create-open' : 'create-closed'} mode="create" open={createOpen} form={form} setForm={setForm} options={options} onClose={() => setCreateOpen(false)} onSubmit={submitCreate} submitting={create.isPending} />
    <KeyDialog key={edit?.id ?? 'edit-closed'} mode="edit" open={Boolean(edit)} form={edit?.form ?? emptyForm} setForm={(next) => setEdit((current) => current && { ...current, form: next })} options={options} onClose={() => setEdit(null)} onSubmit={submitEdit} submitting={update.isPending} />
    <ConfirmDialog open={Boolean(confirm)} title={confirm?.kind === 'delete' ? 'Delete virtual key?' : 'Regenerate virtual key?'} description={confirm?.kind === 'delete' ? `Deleting ${confirm?.key.name} permanently removes this credential and its access.` : `Regenerating ${confirm?.key.name} immediately stops the old key from working.`} confirmLabel={confirm?.kind === 'delete' ? 'Delete key' : 'Regenerate'} onClose={() => setConfirm(null)} onConfirm={runConfirm} />
    <Dialog open={Boolean(plaintext)} onClose={() => setPlaintext(null)}><DialogTitle>Copy your new key now</DialogTitle><DialogContent><Typography color="warning.main" sx={{ mb: 2 }}>This plaintext key will not be shown again. Store it securely before closing.</Typography><Stack direction="row" alignItems="center" spacing={1}><Typography sx={{ wordBreak: 'break-all' }}>{plaintext}</Typography>{plaintext && <CopyButton value={plaintext} label="Copy key" />}</Stack></DialogContent><DialogActions><Button onClick={() => setPlaintext(null)}>Done</Button></DialogActions></Dialog>
    {feedback.node}
  </>;
}

interface KeyFormOptions { models: string[]; teams: { id: string; name: string }[]; mcpServers: { id: string; name: string }[]; policies: string[] }

function KeyDialog({ mode, open, form, setForm, options, onClose, onSubmit, submitting }: { mode: 'create' | 'edit'; open: boolean; form: KeyForm; setForm: (form: KeyForm) => void; options: KeyFormOptions; onClose: () => void; onSubmit: () => void; submitting: boolean }) {
  const [restrictModels, setRestrictModels] = useState(form.allowed_models.length > 0);
  const invalid = !form.name.trim() || (restrictModels && !form.allowed_models.length);
  const set = (patch: Partial<KeyForm>) => setForm({ ...form, ...patch });
  const number = (value: string) => (value ? Number(value) : null);
  return <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
    <DialogTitle>{mode === 'create' ? 'Create virtual key' : `Edit key “${form.name}”`}</DialogTitle>
    <DialogContent><Stack spacing={2} sx={{ pt: 1 }}>
      <TextField label="Name" value={form.name} onChange={(event) => set({ name: event.target.value })} required />
      <TextField select label="Team" value={form.team_id ?? ''} SelectProps={{ displayEmpty: true }} InputLabelProps={{ shrink: true }} onChange={(event) => set({ team_id: event.target.value || null })} helperText="Keys in a team share its budget and its RAG collections.">
        <MenuItem value="">No team</MenuItem>{options.teams.map((team) => <MenuItem key={team.id} value={team.id}>{team.name}</MenuItem>)}
      </TextField>

      <Section title="Access" />
      <ModelAccessField models={options.models} value={form.allowed_models} restricted={restrictModels} onRestrictedChange={setRestrictModels} onChange={(allowed) => set({ allowed_models: allowed })} />
      <McpAccessFields servers={options.mcpServers} value={form} onChange={(access) => set(access)} />
      <TextField select label="Guardrail policy" value={form.guardrail_policy ?? ''} SelectProps={{ displayEmpty: true }} InputLabelProps={{ shrink: true }} onChange={(event) => set({ guardrail_policy: event.target.value || null })}>
        <MenuItem value="">Default</MenuItem>{options.policies.map((policy) => <MenuItem key={policy} value={policy}>{policy}</MenuItem>)}
      </TextField>
      <TextField label="Allowed routes" value={form.allowed_routes.join(', ')} onChange={(event) => set({ allowed_routes: parseList(event.target.value) })} helperText="Comma-separated path prefixes, e.g. /v1/embeddings. Empty allows every route." />

      <Section title="Limits" />
      <Stack direction="row" spacing={2}>
        <TextField fullWidth label="Max budget (USD)" type="number" value={form.max_budget_usd ?? ''} onChange={(event) => set({ max_budget_usd: number(event.target.value) })} helperText="Empty means no budget." />
        <TextField fullWidth select label="Budget period" value={form.budget_duration} onChange={(event) => set({ budget_duration: event.target.value })} helperText="When spend resets.">
          {BUDGET_PERIODS.map((period) => <MenuItem key={period} value={period}>{period[0].toUpperCase() + period.slice(1)}</MenuItem>)}
        </TextField>
      </Stack>
      <Stack direction="row" spacing={2}>
        <TextField fullWidth label="RPM limit" type="number" value={form.rpm_limit ?? ''} onChange={(event) => set({ rpm_limit: number(event.target.value) })} helperText="Requests per minute" />
        <TextField fullWidth label="TPM limit" type="number" value={form.tpm_limit ?? ''} onChange={(event) => set({ tpm_limit: number(event.target.value) })} helperText="Tokens per minute" />
        <TextField fullWidth label="Max concurrent" type="number" value={form.max_parallel_requests ?? ''} onChange={(event) => set({ max_parallel_requests: number(event.target.value) })} helperText="Requests in flight" />
      </Stack>

      <Section title="Lifetime" />
      <TextField label="Expiry" type="datetime-local" InputLabelProps={{ shrink: true }} value={toLocalInput(form.expires_at)} onChange={(event) => set({ expires_at: event.target.value ? new Date(event.target.value).toISOString() : null })} helperText="Empty means the key never expires. Shown in your local time." />
      {mode === 'edit' && <FormControlLabel control={<Switch inputProps={{ role: 'switch' }} checked={form.enabled} onChange={(event) => set({ enabled: event.target.checked })} />} label={form.enabled ? 'Enabled' : 'Disabled (requests are refused)'} />}
    </Stack></DialogContent>
    <DialogActions><Button onClick={onClose}>Cancel</Button><Button variant="contained" onClick={onSubmit} disabled={invalid || submitting}>{mode === 'create' ? 'Create' : 'Save changes'}</Button></DialogActions>
  </Dialog>;
}

function Section({ title }: { title: string }) {
  return <><Divider /><Typography variant="subtitle2" color="text.secondary" sx={{ mt: -0.5 }}>{title}</Typography></>;
}

/** An ISO timestamp as a datetime-local value in the viewer's time zone. */
export function toLocalInput(iso: string | null): string {
  if (!iso) return '';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return new Date(date.getTime() - date.getTimezoneOffset() * 60_000).toISOString().slice(0, 16);
}

/** Which MCP servers and tools a key may use; empty lists mean no restriction. */
function McpAccessFields({ servers, value, onChange }: { servers: { id: string; name: string }[]; value: { allowed_mcp_servers?: string[]; allowed_tools?: string[] }; onChange: (access: { allowed_mcp_servers: string[]; allowed_tools: string[] }) => void }) {
  const allowedServers = value.allowed_mcp_servers ?? [];
  const allowedTools = value.allowed_tools ?? [];
  return <>
    <FormControl><InputLabel>Allowed MCP servers</InputLabel><Select multiple label="Allowed MCP servers" value={allowedServers} onChange={(event) => onChange({ allowed_mcp_servers: event.target.value as string[], allowed_tools: allowedTools })} renderValue={(selected) => selected.length ? selected.join(', ') : 'All servers'}>{servers.map((server) => <MenuItem key={server.id} value={server.name}>{server.name}</MenuItem>)}</Select></FormControl>
    <TextField label="Allowed tools" value={allowedTools.join(', ')} onChange={(event) => onChange({ allowed_mcp_servers: allowedServers, allowed_tools: parseList(event.target.value) })} helperText="Comma-separated server__tool names; * is a wildcard (e.g. files__read*). Empty allows every tool." />
  </>;
}

function parseList(text: string): string[] {
  return text.split(',').map((item) => item.trim()).filter(Boolean);
}

/** Allow every model (the default), or choose specific ones; an empty list means all models. */
function ModelAccessField({ models, value, restricted, onRestrictedChange, onChange }: { models: string[]; value: string[]; restricted: boolean; onRestrictedChange: (restricted: boolean) => void; onChange: (models: string[]) => void }) {
  const labelId = useId();
  return <Stack spacing={1}>
    <FormControlLabel control={<Switch inputProps={{ role: 'switch' }} checked={!restricted} onChange={(event) => { onRestrictedChange(!event.target.checked); if (event.target.checked) onChange([]); }} />} label="All models" />
    {restricted
      ? <FormControl error={value.length === 0}><InputLabel id={labelId}>Allowed models</InputLabel><Select multiple labelId={labelId} label="Allowed models" value={value} onChange={(event) => onChange(event.target.value as string[])}>{models.map((model) => <MenuItem key={model} value={model}>{model}</MenuItem>)}</Select><FormHelperText>{value.length ? `${value.length} model${value.length === 1 ? '' : 's'} allowed` : 'Choose at least one model, or turn on All models.'}</FormHelperText></FormControl>
      : <Typography variant="body2" color="text.secondary" sx={{ mt: -0.5 }}>This key can use every model, including models added later.</Typography>}
  </Stack>;
}
