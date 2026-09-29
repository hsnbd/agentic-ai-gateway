import { useState, type FormEvent } from 'react';
import { Alert, Button, Dialog, DialogActions, DialogContent, DialogTitle, MenuItem, Stack, TextField, Typography } from '@mui/material';
import type { McpServer, McpTransport } from '../../api/hooks/useMcpServers';

type ServerFields = {
  name: string; transport: McpTransport; url: string; command: string; args: string;
  description: string; toolPrefix: string; headers: string; env: string; isActive: boolean;
};

function parseStringMap(value: string, label: string): Record<string, string> {
  let parsed: unknown;
  try { parsed = JSON.parse(value); } catch { throw new Error(`${label} must be valid JSON.`); }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed) || Object.entries(parsed).some(([key, entry]) => !key || typeof entry !== 'string')) {
    throw new Error(`${label} must be a JSON object with string values.`);
  }
  return parsed as Record<string, string>;
}

function parseArgs(value: string): string[] {
  let parsed: unknown;
  try { parsed = JSON.parse(value); } catch { throw new Error('Arguments must be a JSON array of strings.'); }
  if (!Array.isArray(parsed) || parsed.some((item) => typeof item !== 'string')) throw new Error('Arguments must be a JSON array of strings.');
  return parsed as string[];
}

export function McpServerForm({ server, onClose, onSave, pending, serverError }: {
  server?: McpServer;
  onClose: () => void;
  onSave: (payload: Record<string, unknown>) => void;
  pending: boolean;
  serverError: string | null;
}) {
  const [fields, setFields] = useState<ServerFields>(() => ({
    name: server?.name ?? '', transport: server?.transport ?? 'http', url: server?.url ?? '',
    command: server?.command ?? '', args: JSON.stringify(server?.args ?? []),
    description: server?.description ?? '', toolPrefix: server?.tool_prefix ?? '',
    headers: '', env: '', isActive: server?.is_active ?? true,
  }));
  const [localError, setLocalError] = useState<string | null>(null);
  const set = <K extends keyof ServerFields>(key: K, value: ServerFields[K]) => setFields((current) => ({ ...current, [key]: value }));
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLocalError(null);
    try {
      const stdio = fields.transport === 'stdio';
      if (!fields.name.trim()) throw new Error('A server name is required.');
      if (stdio ? !fields.command.trim() : !fields.url.trim()) throw new Error(stdio ? 'A command is required for stdio.' : 'An endpoint URL is required for HTTP.');
      const args = stdio ? parseArgs(fields.args) : [];
      const payload: Record<string, unknown> = {};
      if (!server || fields.name.trim() !== server.name) payload.name = fields.name.trim();
      if (!server || fields.transport !== server.transport) payload.transport = fields.transport;
      if (!server || fields.description.trim() !== (server.description ?? '')) payload.description = fields.description.trim() || null;
      if (!server || fields.toolPrefix.trim() !== (server.tool_prefix ?? '')) payload.tool_prefix = fields.toolPrefix.trim() || null;
      if (server && fields.isActive !== server.is_active) payload.is_active = fields.isActive;
      if (stdio) {
        if (!server || fields.command.trim() !== server.command) payload.command = fields.command.trim();
        if (!server || JSON.stringify(args) !== JSON.stringify(server.args)) payload.args = args;
        if (server && server.transport !== 'stdio') payload.url = null;
      } else {
        if (!server || fields.url.trim() !== server.url) payload.url = fields.url.trim();
        if (server?.transport === 'stdio') payload.command = null;
      }
      if (fields.headers.trim()) payload.headers = parseStringMap(fields.headers, 'Headers');
      else if (!server) payload.headers = {};
      if (fields.env.trim()) {
        const env = parseStringMap(fields.env, 'Environment');
        if (Object.values(env).some((value) => value === '***')) throw new Error('Enter actual environment values; *** is only a display mask.');
        payload.env = env;
      } else if (!server) payload.env = {};
      if (server && Object.keys(payload).length === 0) { onClose(); return; }
      onSave(payload);
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : 'Invalid server configuration.');
    }
  };
  return <Dialog open onClose={pending ? undefined : onClose} fullWidth maxWidth="sm">
    <Stack component="form" onSubmit={submit}>
      <DialogTitle>{server ? `Edit ${server.name}` : 'Add MCP server'}</DialogTitle>
      <DialogContent sx={{ display: 'grid', gap: 2, pt: '12px !important' }}>
        <TextField label="Server name" required autoFocus value={fields.name} onChange={(event) => set('name', event.target.value)} />
        <TextField select label="Transport" value={fields.transport} onChange={(event) => {
          const transport = event.target.value;
          if (transport === 'stdio' || transport === 'http' || transport === 'streamable-http') set('transport', transport);
        }}><MenuItem value="http">HTTP</MenuItem><MenuItem value="streamable-http">Streamable HTTP</MenuItem><MenuItem value="stdio">stdio</MenuItem></TextField>
        {fields.transport === 'stdio' ? <>
          <TextField label="Command" required value={fields.command} onChange={(event) => set('command', event.target.value)} />
          <TextField label="Arguments (JSON array of strings)" value={fields.args} onChange={(event) => set('args', event.target.value)} multiline minRows={2} />
          {server && Object.keys(server.env).length > 0 && <Typography variant="body2" color="text.secondary">Configured variables: {Object.keys(server.env).join(', ')}. Values are masked by the API; “***” is not the actual value.</Typography>}
          <TextField label="Environment (JSON object)" value={fields.env} onChange={(event) => set('env', event.target.value)} multiline minRows={2} placeholder={'{"API_KEY":"actual value"}'} helperText={server ? 'Leave blank to preserve all existing variables. If entered, this replaces the entire environment: re-enter values for every variable to retain.' : 'Optional names and values. Secret values will not be shown again.'} />
        </> : <>
          <TextField label="Endpoint URL" type="url" required value={fields.url} onChange={(event) => set('url', event.target.value)} />
          {server && Object.keys(server.headers).length > 0 && <Typography variant="body2" color="text.secondary">Headers configured: {Object.keys(server.headers).join(', ')}. Values are not shown here.</Typography>}
          <TextField label="Headers (JSON object)" value={fields.headers} onChange={(event) => set('headers', event.target.value)} multiline minRows={2} placeholder={'{"Authorization":"Bearer …"}'} helperText={server ? 'Leave blank to preserve headers. Entering new headers replaces the entire set.' : 'Optional request headers. Values are not displayed after saving.'} />
        </>}
        <TextField label="Description" multiline minRows={2} value={fields.description} onChange={(event) => set('description', event.target.value)} />
        <TextField label="Tool prefix (optional)" value={fields.toolPrefix} onChange={(event) => set('toolPrefix', event.target.value)} />
        {server && <TextField select label="Active" value={fields.isActive ? 'yes' : 'no'} onChange={(event) => set('isActive', event.target.value === 'yes')}><MenuItem value="yes">Active</MenuItem><MenuItem value="no">Inactive</MenuItem></TextField>}
        {(localError || serverError) && <Alert severity="error">{localError || serverError}</Alert>}
      </DialogContent>
      <DialogActions><Button onClick={onClose} disabled={pending}>Cancel</Button><Button variant="contained" type="submit" disabled={pending}>{pending ? 'Saving…' : server ? 'Save changes' : 'Add server'}</Button></DialogActions>
    </Stack>
  </Dialog>;
}
