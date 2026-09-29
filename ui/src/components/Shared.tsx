import { useState, type ReactNode } from 'react';
import { Alert, Box, Button, Card, CardContent, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, Stack, Typography, Tooltip } from '@mui/material';
import ContentCopyRounded from '@mui/icons-material/ContentCopyRounded';
import ErrorOutlineRounded from '@mui/icons-material/ErrorOutlineRounded';
import InboxOutlined from '@mui/icons-material/InboxOutlined';
import type { GridColDef, GridRowsProp } from '@mui/x-data-grid';
import { DataGrid } from '@mui/x-data-grid';

export function PageHeader({ title, description, action }: { title: string; description?: string; action?: ReactNode }) {
  return <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ xs: 'stretch', sm: 'center' }} spacing={2} sx={{ mb: 3 }}>
    <Box><Typography variant="h1">{title}</Typography>{description && <Typography color="text.secondary" sx={{ mt: 0.75 }}>{description}</Typography>}</Box>{action}
  </Stack>;
}
export function StatCard({ label, value, hint, icon }: { label: string; value: string; hint?: string; icon?: ReactNode }) {
  return <Card sx={{ height: '100%' }}><CardContent sx={{ p: 2.5, '&:last-child': { pb: 2.5 } }}>
    <Stack direction="row" justifyContent="space-between" alignItems="center"><Typography variant="body2" color="text.secondary">{label}</Typography>{icon}</Stack>
    <Typography variant="h2" sx={{ mt: 2, fontVariantNumeric: 'tabular-nums' }}>{value}</Typography>{hint && <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>{hint}</Typography>}
  </CardContent></Card>;
}
export function DataTable<T extends { id: string | number }>({ rows, columns, loading = false, rowCount }: { rows: GridRowsProp<T>; columns: GridColDef<T>[]; loading?: boolean; rowCount?: number }) {
  return <Box sx={{ width: '100%', height: 520 }}><DataGrid rows={rows} columns={columns} loading={loading} rowCount={rowCount}
    pageSizeOptions={[10, 25, 50]} initialState={{ pagination: { paginationModel: { pageSize: 10, page: 0 } } }}
    disableRowSelectionOnClick sx={{ border: 0, '& .MuiDataGrid-columnHeaders': { borderBottomColor: 'divider' } }} /></Box>;
}
export function EmptyState({ title, description, icon }: { title: string; description?: string; icon?: ReactNode }) {
  return <Stack alignItems="center" justifyContent="center" textAlign="center" sx={{ minHeight: 220, p: 4 }}>
    {icon ?? <InboxOutlined color="disabled" sx={{ fontSize: 42, mb: 1.5 }} />}<Typography variant="h3">{title}</Typography>
    {description && <Typography color="text.secondary" sx={{ mt: 1, maxWidth: 460 }}>{description}</Typography>}
  </Stack>;
}
export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof Error ? error.message : 'The request could not be completed.';
  return <Alert severity="error" icon={<ErrorOutlineRounded />} action={onRetry ? <Button color="inherit" size="small" onClick={onRetry}>Retry</Button> : undefined}>{message}</Alert>;
}
export function LoadingState({ label = 'Loading' }: { label?: string }) {
  return <Stack alignItems="center" justifyContent="center" spacing={1.5} sx={{ minHeight: 220 }}><CircularProgress size={28} /><Typography variant="body2" color="text.secondary">{label}</Typography></Stack>;
}
export function ConfirmDialog({ open, title, description, confirmLabel = 'Confirm', onClose, onConfirm }: { open: boolean; title: string; description: string; confirmLabel?: string; onClose: () => void; onConfirm: () => void }) {
  return <Dialog open={open} onClose={onClose}><DialogTitle>{title}</DialogTitle><DialogContent><Typography color="text.secondary">{description}</Typography></DialogContent>
    <DialogActions><Button onClick={onClose}>Cancel</Button><Button color="error" variant="contained" onClick={onConfirm}>{confirmLabel}</Button></DialogActions></Dialog>;
}
export function CopyButton({ value, label = 'Copy' }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => { await navigator.clipboard.writeText(value); setCopied(true); window.setTimeout(() => setCopied(false), 1400); };
  return <Tooltip title={copied ? 'Copied' : label}><IconButton size="small" aria-label={label} onClick={() => void copy()}><ContentCopyRounded fontSize="small" /></IconButton></Tooltip>;
}
export function JsonViewer({ value }: { value: unknown }) {
  return <Box component="pre" sx={{ overflow: 'auto', p: 2, borderRadius: 2, bgcolor: 'action.hover', fontSize: 13, lineHeight: 1.5 }}>
    {JSON.stringify(value, null, 2) ?? 'null'}</Box>;
}
