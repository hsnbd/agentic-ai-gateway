import { Button, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, MenuItem, Stack, TextField, Typography } from '@mui/material';
import DeleteOutlineRounded from '@mui/icons-material/DeleteOutlineRounded';
import PersonAddRounded from '@mui/icons-material/PersonAddRounded';
import type { GridColDef } from '@mui/x-data-grid';
import { useState } from 'react';
import type { AdminUser } from '../../api/types';
import { useCreateUser, useDeleteUser, useUpdateUser, useUsers } from '../../api/hooks/useUsers';
import { ConfirmDialog, DataTable, ErrorState, LoadingState } from '../../components/Shared';
import { StatusChip, formatDate, useAdminMutationFeedback } from './AdminPrimitives';

const MIN_PASSWORD = 12;

/** Console users: invite, change role, deactivate, and remove. */
export default function UserManagement({ currentEmail }: { currentEmail?: string }) {
  const users = useUsers(); const create = useCreateUser(); const update = useUpdateUser(); const remove = useDeleteUser(); const feedback = useAdminMutationFeedback();
  const [adding, setAdding] = useState(false); const [deleting, setDeleting] = useState<AdminUser | null>(null);
  const columns: GridColDef<AdminUser>[] = [
    { field: 'email', headerName: 'Email', flex: 1, minWidth: 200 },
    { field: 'role', headerName: 'Role', width: 150, renderCell: ({ row }) => <TextField select size="small" variant="standard" value={row.role} disabled={row.email === currentEmail} inputProps={{ 'aria-label': `Role for ${row.email}` }} onChange={(event) => update.mutate({ id: row.id, body: { role: event.target.value as AdminUser['role'] } }, { onSuccess: () => feedback.notify(`${row.email} is now ${event.target.value}.`), onError: feedback.fail })}><MenuItem value="admin">admin</MenuItem><MenuItem value="viewer">viewer</MenuItem></TextField> },
    { field: 'is_active', headerName: 'Status', width: 120, renderCell: ({ row }) => <StatusChip value={row.is_active ? 'active' : 'disabled'} /> },
    { field: 'created_at', headerName: 'Created', width: 180, valueGetter: (_, row) => formatDate(row.created_at) },
    { field: 'actions', headerName: 'Actions', width: 200, sortable: false, renderCell: ({ row }) => row.email === currentEmail ? <Typography variant="caption" color="text.secondary">You</Typography> : <Stack direction="row"><Button size="small" aria-label={`${row.is_active ? 'Deactivate' : 'Activate'} ${row.email}`} onClick={() => update.mutate({ id: row.id, body: { is_active: !row.is_active } }, { onSuccess: () => feedback.notify(`${row.email} ${row.is_active ? 'deactivated' : 'activated'}.`), onError: feedback.fail })}>{row.is_active ? 'Deactivate' : 'Activate'}</Button><IconButton aria-label={`Delete ${row.email}`} color="error" onClick={() => setDeleting(row)}><DeleteOutlineRounded fontSize="small" /></IconButton></Stack> },
  ];
  if (users.isLoading) return <LoadingState label="Loading users" />;
  if (users.isError) return <ErrorState error={users.error} onRetry={() => void users.refetch()} />;
  return <>
    <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mt: 1, mb: 2 }}><Typography color="text.secondary">Admins can change the gateway; viewers can only read it.</Typography><Button variant="contained" startIcon={<PersonAddRounded />} onClick={() => setAdding(true)}>Add user</Button></Stack>
    <DataTable rows={users.data?.items ?? []} columns={columns} rowCount={users.data?.total} />
    <AddUserDialog open={adding} pending={create.isPending} onClose={() => setAdding(false)} onSubmit={(body) => create.mutate(body, { onSuccess: () => { setAdding(false); feedback.notify(`Added ${body.email}.`); }, onError: feedback.fail })} />
    <ConfirmDialog open={Boolean(deleting)} title="Delete console user?" description={`${deleting?.email ?? ''} will lose access to the console immediately.`} confirmLabel="Delete user" onClose={() => setDeleting(null)} onConfirm={() => { const target = deleting; setDeleting(null); if (target) remove.mutate(target.id, { onSuccess: () => feedback.notify(`Deleted ${target.email}.`), onError: feedback.fail }); }} />
    {feedback.node}
  </>;
}

function AddUserDialog({ open, pending, onClose, onSubmit }: { open: boolean; pending: boolean; onClose: () => void; onSubmit: (body: { email: string; password: string; role: AdminUser['role'] }) => void }) {
  const [email, setEmail] = useState(''); const [password, setPassword] = useState(''); const [role, setRole] = useState<AdminUser['role']>('viewer');
  const tooShort = password.length > 0 && password.length < MIN_PASSWORD;
  const close = () => { setEmail(''); setPassword(''); setRole('viewer'); onClose(); };
  return <Dialog open={open} onClose={close} fullWidth maxWidth="xs"><DialogTitle>Add console user</DialogTitle><DialogContent><Stack spacing={2} sx={{ pt: 1 }}><TextField label="Email" type="email" required value={email} onChange={(event) => setEmail(event.target.value)} /><TextField label="Initial password" type="password" required value={password} error={tooShort} helperText={`At least ${MIN_PASSWORD} characters`} onChange={(event) => setPassword(event.target.value)} /><TextField label="Role" select value={role} onChange={(event) => setRole(event.target.value as AdminUser['role'])}><MenuItem value="viewer">viewer</MenuItem><MenuItem value="admin">admin</MenuItem></TextField></Stack></DialogContent><DialogActions><Button onClick={close}>Cancel</Button><Button variant="contained" disabled={pending || !email || password.length < MIN_PASSWORD} onClick={() => { onSubmit({ email, password, role }); setEmail(''); setPassword(''); setRole('viewer'); }}>Add user</Button></DialogActions></Dialog>;
}
