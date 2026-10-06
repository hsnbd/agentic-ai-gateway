import { useMemo, useState, type FormEvent } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Alert, Box, Button, Card, CardContent, Chip, Divider, MenuItem, Paper, Stack, Switch, TextField, Typography } from '@mui/material';
import { apiRequest } from '../api/client';
import { mcpKeys, useMcpServers, useMcpTools, type McpServer, type McpTool, type McpToolCallResponse, type McpRefreshResponse } from '../api/hooks/useMcpServers';
import { useAuth } from '../auth/AuthProvider';
import { ConfirmDialog, EmptyState, ErrorState, JsonViewer, LoadingState, PageHeader } from '../components/Shared';
import { LongText } from '../features/knowledge/LongText';
import { McpServerForm } from '../features/knowledge/McpServerForm';
import { ToolCallsPanel } from '../features/knowledge/ToolCallsPanel';

interface SchemaProperty {
  type: string;
  description: string;
  enumValues: string[];
}

function schemaProperties(schema: Record<string, unknown>): Record<string, SchemaProperty> {
  const rawProperties = schema.properties;
  if (typeof rawProperties !== 'object' || rawProperties === null || Array.isArray(rawProperties)) return {};
  const output: Record<string, SchemaProperty> = {};
  for (const [name, value] of Object.entries(rawProperties)) {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) continue;
    const definition = value as Record<string, unknown>;
    const type = typeof definition.type === 'string' ? definition.type : 'object';
    const description = typeof definition.description === 'string' ? definition.description : '';
    const enumValues = Array.isArray(definition.enum) ? definition.enum.filter((item): item is string => typeof item === 'string') : [];
    output[name] = { type, description, enumValues };
  }
  return output;
}

function requiredProperties(schema: Record<string, unknown>): Set<string> {
  const required = schema.required;
  return new Set(Array.isArray(required) ? required.filter((name): name is string => typeof name === 'string') : []);
}

function schemaTypeLabel(type: string): string {
  if (type === 'integer' || type === 'number') return type;
  if (type === 'array' || type === 'object' || type === 'boolean' || type === 'string') return type;
  return type;
}

function displayDate(value: string | null): string {
  if (!value) return 'Never';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function JsonInput({ label, value, onChange, required, helperText }: { label: string; value: string; onChange: (value: string) => void; required: boolean; helperText?: string }) {
  return <TextField label={label} value={value} onChange={(event) => onChange(event.target.value)} required={required} multiline minRows={2} helperText={helperText} />;
}

function ToolParameterList({ schema }: { schema: Record<string, unknown> }) {
  const properties = schemaProperties(schema);
  const required = requiredProperties(schema);
  const entries = Object.entries(properties);
  if (entries.length === 0) return <Typography variant="body2" color="text.secondary">No parameters defined.</Typography>;
  return <Stack divider={<Divider flexItem />}>
    {entries.map(([name, property]) => <Box key={name} sx={{ py: 1.25, display: 'grid', gridTemplateColumns: { xs: 'minmax(0, 1fr)', sm: 'minmax(120px, 0.7fr) minmax(80px, 0.5fr) minmax(0, 2fr)' }, gap: 1, minWidth: 0 }}>
      <Stack direction="row" alignItems="center" gap={0.75} sx={{ minWidth: 0 }}><Typography variant="body2" sx={{ fontWeight: 700, overflowWrap: 'anywhere' }}>{name}</Typography>{required.has(name) && <Chip size="small" label="required" color="primary" />}</Stack>
      <Typography variant="caption" color="text.secondary">{schemaTypeLabel(property.type)}</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ overflowWrap: 'anywhere' }}>{property.description || 'No description'}</Typography>
    </Box>)}
  </Stack>;
}

function ToolTester({ tool, canInvoke }: { tool: McpTool; canInvoke: boolean }) {
  const schema = tool.function.parameters;
  const properties = useMemo(() => schemaProperties(schema), [schema]);
  const required = useMemo(() => requiredProperties(schema), [schema]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [lastRequest, setLastRequest] = useState<Record<string, unknown> | null>(null);
  const [callError, setCallError] = useState<string | null>(null);
  const [response, setResponse] = useState<McpToolCallResponse | null>(null);
  const callMutation = useMutation({
    mutationFn: (arguments_: Record<string, unknown>) => apiRequest<McpToolCallResponse>('/v1/mcp/tools/call', {
      method: 'POST',
      body: { name: tool.function.name, arguments: arguments_ },
    }),
    onSuccess: (result) => { setResponse(result); setCallError(null); },
    onError: (error: Error) => { setCallError(error.message); setResponse(null); },
  });

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setCallError(null);
    const args: Record<string, unknown> = {};
    try {
      for (const [name, property] of Object.entries(properties)) {
        const value = values[name] ?? '';
        if (!value && !required.has(name)) continue;
        if (!value && required.has(name)) throw new Error(`${name} is required`);
        if (property.type === 'boolean') args[name] = value === 'true';
        else if (property.type === 'number' || property.type === 'integer') {
          const parsed = Number(value);
          if (!Number.isFinite(parsed) || (property.type === 'integer' && !Number.isInteger(parsed))) throw new Error(`${name} must be a valid ${property.type}`);
          args[name] = parsed;
        } else if (property.type === 'object' || property.type === 'array') {
          const parsed: unknown = JSON.parse(value);
          if (property.type === 'object' && (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed))) throw new Error(`${name} must be a JSON object`);
          if (property.type === 'array' && !Array.isArray(parsed)) throw new Error(`${name} must be a JSON array`);
          args[name] = parsed;
        } else args[name] = value;
      }
    } catch (error) {
      setCallError(error instanceof Error ? error.message : 'Invalid tool arguments.');
      return;
    }
    setLastRequest(args);
    callMutation.mutate(args);
  };

  if (!canInvoke) return <Alert severity="info">Tool tester is available to administrators only.</Alert>;

  return <Box sx={{ mt: 2 }}>
    <Box component="form" onSubmit={submit} sx={{ display: 'grid', gap: 1.5 }}>
      {Object.entries(properties).map(([name, property]) => {
        const isRequired = required.has(name);
        if (property.type === 'boolean') return <TextField key={name} select label={name} value={values[name] ?? ''} required={isRequired} onChange={(event) => setValues((current) => ({ ...current, [name]: event.target.value }))} helperText={property.description}>
          <MenuItem value="true">true</MenuItem><MenuItem value="false">false</MenuItem>
        </TextField>;
        if (property.enumValues.length > 0) return <TextField key={name} select label={name} value={values[name] ?? ''} required={isRequired} onChange={(event) => setValues((current) => ({ ...current, [name]: event.target.value }))} helperText={property.description}>
          {property.enumValues.map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
        </TextField>;
        if (property.type === 'object' || property.type === 'array') return <JsonInput key={name} label={`${name} (${property.type}, JSON)`} value={values[name] ?? ''} onChange={(value) => setValues((current) => ({ ...current, [name]: value }))} required={isRequired} helperText={property.description || `Enter a JSON ${property.type}.`} />;
        return <TextField key={name} label={name} type={property.type === 'number' || property.type === 'integer' ? 'number' : 'text'} required={isRequired} value={values[name] ?? ''} onChange={(event) => setValues((current) => ({ ...current, [name]: event.target.value }))} helperText={property.description} />;
      })}
      {canInvoke && <Button type="submit" variant="contained" disabled={callMutation.isPending}>{callMutation.isPending ? 'Invoking…' : 'Invoke tool'}</Button>}
    </Box>
    {callError && <Alert severity="error" sx={{ mt: 1.5 }}>{callError}</Alert>}
    {(response || callMutation.isPending) && <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 1.5, mt: 2 }}>
      <Box><Typography variant="subtitle2" sx={{ mb: 0.75 }}>Request</Typography><JsonViewer value={{ name: tool.function.name, arguments: lastRequest ?? {} }} /></Box>
      <Box><Typography variant="subtitle2" sx={{ mb: 0.75 }}>Response</Typography>{response ? <JsonViewer value={response.message} /> : <LoadingState label="Waiting for tool response" />}</Box>
    </Box>}
  </Box>;
}

export default function McpServers() {
  const { user } = useAuth();
  const canManage = user?.role === 'admin';
  const queryClient = useQueryClient();
  const serversQuery = useMcpServers();
  const servers = serversQuery.data ?? [];
  const [selectedId, setSelectedId] = useState<string | undefined>();
  const selectedServer = servers.find((server) => server.id === selectedId) ?? servers[0];
  const toolsQuery = useMcpTools(selectedServer?.id);
  const tools = toolsQuery.data ?? [];
  const [createOpen, setCreateOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<McpServer | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<McpServer | null>(null);
  const [showRaw, setShowRaw] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [refreshResult, setRefreshResult] = useState<McpRefreshResponse | null>(null);
  const [healthResult, setHealthResult] = useState<boolean | null>(null);
  const [operationError, setOperationError] = useState<string | null>(null);

  const createMutation = useMutation({
    mutationFn: (payload: Record<string, unknown>) => apiRequest<McpServer>('/v1/mcp/servers', { method: 'POST', body: payload }),
    onSuccess: async (server) => {
      await queryClient.invalidateQueries({ queryKey: mcpKeys.servers });
      setSelectedId(server.id);
      await queryClient.invalidateQueries({ queryKey: mcpKeys.tools(server.id) });
      setCreateOpen(false);
      setCreateError(null);
    },
    onError: (error: Error) => setCreateError(error.message),
  });

  const editMutation = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: Record<string, unknown> }) => apiRequest<McpServer>(`/v1/mcp/servers/${encodeURIComponent(id)}`, { method: 'PATCH', body: payload }),
    onSuccess: async (server) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: mcpKeys.servers }),
        queryClient.invalidateQueries({ queryKey: mcpKeys.tools(server.id) }),
      ]);
      setEditTarget(null);
      setSelectedId(server.id);
      setCreateError(null);
      setRefreshResult(null);
    },
    onError: (error: Error) => setCreateError(error.message),
  });

  const deleteMutation = useMutation({
    mutationFn: (serverId: string) => apiRequest<void>(`/v1/mcp/servers/${encodeURIComponent(serverId)}`, { method: 'DELETE' }),
    onSuccess: async (_, serverId) => {
      await queryClient.invalidateQueries({ queryKey: mcpKeys.servers });
      setSelectedId((current) => current === serverId ? undefined : current);
      setDeleteTarget(null);
    },
  });

  const refreshMutation = useMutation({
    mutationFn: (serverId: string) => apiRequest<McpRefreshResponse>(`/v1/mcp/servers/${encodeURIComponent(serverId)}/refresh`, { method: 'POST' }),
    onSuccess: async (result) => {
      setRefreshResult(result);
      setHealthResult(result.healthy);
      setOperationError(null);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: mcpKeys.servers }),
        queryClient.invalidateQueries({ queryKey: mcpKeys.tools(selectedServer?.id) }),
      ]);
    },
    onError: (error: Error) => setOperationError(error.message),
  });

  const healthMutation = useMutation({
    mutationFn: () => apiRequest<Record<string, boolean>>('/v1/mcp/health'),
    onSuccess: (result) => {
      setHealthResult(selectedServer ? result[selectedServer.id] ?? false : false);
      setOperationError(null);
    },
    onError: (error: Error) => setOperationError(error.message),
  });

  const discoveredCount = useMemo(() => selectedServer?.discovered_tools.length ?? 0, [selectedServer]);

  return <Box>
    <PageHeader title="MCP servers" description="Manage connected servers, review discovered tool schemas, and test tool calls." action={canManage ? <Button variant="contained" onClick={() => { setCreateError(null); setCreateOpen(true); }}>Add server</Button> : null} />
    {serversQuery.isLoading ? <LoadingState label="Loading MCP servers" /> : serversQuery.isError ? <ErrorState error={serversQuery.error} onRetry={() => void serversQuery.refetch()} /> : servers.length === 0 ?
      <Paper><EmptyState title="No MCP servers configured" description={canManage ? 'Add a server to discover its tools.' : 'No MCP servers are available.'} /></Paper> :
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(260px, 0.85fr) minmax(0, 2.15fr)' }, gap: 2, alignItems: 'start' }}>
        <Stack spacing={1.5}>
          {servers.map((server) => <Card key={server.id} variant="outlined" sx={{ borderColor: server.id === selectedServer?.id ? 'primary.main' : 'divider' }}>
            <CardContent>
              <Stack direction="row" justifyContent="space-between" alignItems="flex-start" gap={1}>
                <Button onClick={() => { setSelectedId(server.id); setShowRaw(false); setRefreshResult(null); setHealthResult(null); setOperationError(null); }} sx={{ p: 0, minWidth: 0, textAlign: 'left', justifyContent: 'flex-start', fontSize: '1rem', fontWeight: 700, overflowWrap: 'anywhere' }}>{server.name}</Button>
                <Chip size="small" label={server.health_status || 'unknown'} color={server.health_status === 'healthy' ? 'success' : server.health_status === 'unhealthy' ? 'error' : 'default'} />
              </Stack>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1, overflowWrap: 'anywhere' }}>{server.transport} · {server.transport === 'stdio' ? `${server.command ?? 'Command unavailable'} ${server.args.join(' ')}` : server.url ?? 'Endpoint unavailable'}</Typography>
              <Stack direction="row" gap={1} flexWrap="wrap" sx={{ mt: 1.5 }}><Chip size="small" label={`${server.discovered_tools.length} tools`} /><Chip size="small" label={`Discovery ${displayDate(server.last_health_check_at)}`} /></Stack>
              {canManage && <Button size="small" color="error" sx={{ mt: 1 }} onClick={() => setDeleteTarget(server)}>Delete</Button>}
            </CardContent>
          </Card>)}
        </Stack>

        {selectedServer && <Paper sx={{ minWidth: 0, overflow: 'hidden' }}>
          <Box sx={{ p: { xs: 2, md: 3 } }}>
            <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ sm: 'center' }} gap={2}>
              <Box sx={{ minWidth: 0 }}><Typography variant="h2" sx={{ overflowWrap: 'anywhere' }}>{selectedServer.name}</Typography><Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>{selectedServer.transport} · {selectedServer.transport === 'stdio' ? `${selectedServer.command ?? 'Command unavailable'} ${selectedServer.args.join(' ')}` : selectedServer.url ?? 'No URL returned'}</Typography>{selectedServer.description && <LongText text={selectedServer.description} lines={3} />}</Box>
              <Stack direction="row" gap={1} flexWrap="wrap">
                <Button variant="outlined" onClick={() => healthMutation.mutate()} disabled={healthMutation.isPending}>{healthMutation.isPending ? 'Checking…' : 'Check health'}</Button>
                {canManage && <Button variant="contained" onClick={() => { setOperationError(null); refreshMutation.mutate(selectedServer.id); }} disabled={refreshMutation.isPending}>{refreshMutation.isPending ? 'Discovering…' : 'Rediscover tools'}</Button>}
                {canManage && <Button variant="outlined" onClick={() => { setCreateError(null); setEditTarget(selectedServer); }}>Edit</Button>}
                {canManage && <Button color="error" variant="outlined" onClick={() => setDeleteTarget(selectedServer)}>Delete</Button>}
              </Stack>
            </Stack>
            {selectedServer.transport === 'stdio' && Object.keys(selectedServer.env).length > 0 && <Typography variant="body2" sx={{ mt: 1 }}>Environment variables: {Object.keys(selectedServer.env).join(', ')} (values masked by server as ***, not literal values)</Typography>}
            <Stack direction="row" gap={1} flexWrap="wrap" sx={{ mt: 1.5 }}><Chip label={`Health: ${healthResult === null ? selectedServer.health_status : healthResult ? 'healthy' : 'unhealthy'}`} color={(healthResult ?? (selectedServer.health_status === 'healthy')) ? 'success' : 'error'} /><Chip label={`${discoveredCount} tools`} /><Chip label={`Last discovery: ${displayDate(selectedServer.last_health_check_at)}`} /></Stack>
            {healthResult !== null && <Alert severity={healthResult ? 'success' : 'warning'} sx={{ mt: 2 }}>Health check result for {selectedServer.name}: {healthResult ? 'healthy' : 'unhealthy'}. The health endpoint reports the registry’s current cached state.</Alert>}
            {refreshResult && <Alert severity={refreshResult.healthy ? 'success' : 'error'} sx={{ mt: 2 }}><Typography variant="subtitle2">{refreshResult.healthy ? 'Discovery succeeded' : `Discovery failed: ${refreshResult.failure_kind ?? 'unknown failure'}`}</Typography>{refreshResult.healthy ? `${refreshResult.tools.length} tool definitions returned.` : <LongText text={refreshResult.error ?? 'The server returned no diagnostic message.'} />}</Alert>}
            {operationError && <ErrorState error={operationError} />}
            {(healthMutation.isPending || refreshMutation.isPending) && <Box sx={{ mt: 1 }}><LoadingState label={healthMutation.isPending ? 'Checking current server health' : 'Connecting to server and discovering tools'} /></Box>}
          </Box>
          <Divider />
          <Box sx={{ p: { xs: 2, md: 3 } }}>
            <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ sm: 'center' }} gap={1} sx={{ mb: 1.5 }}>
              <Box><Typography variant="h3">Discovered tools</Typography><Typography variant="body2" color="text.secondary">Input schemas are presented as parameters; raw JSON is optional.</Typography></Box>
              <Stack direction="row" alignItems="center"><Typography variant="body2">Raw schema</Typography><Switch checked={showRaw} onChange={(event) => setShowRaw(event.target.checked)} inputProps={{ 'aria-label': 'Show raw input schema' }} /></Stack>
            </Stack>
            {toolsQuery.isLoading ? <LoadingState label="Loading discovered tools" /> : toolsQuery.isError ? <ErrorState error={toolsQuery.error} onRetry={() => void toolsQuery.refetch()} /> : tools.length === 0 ?
              <EmptyState title="No tools discovered" description="Run discovery again or check the server connection." /> :
              <Stack spacing={1.5}>
                {tools.map((tool) => <Card key={tool.function.name} variant="outlined"><CardContent>
                  <Typography variant="h3" sx={{ overflowWrap: 'anywhere' }}>{tool.function.name}</Typography>
                  {tool.function.description && <LongText text={tool.function.description} lines={3} />}
                  <Divider sx={{ my: 1.5 }} />
                  {showRaw ? <JsonViewer value={tool.function.parameters} /> : <ToolParameterList schema={tool.function.parameters} />}
                  <Divider sx={{ my: 1.5 }} />
                  <Typography variant="subtitle2">Tool tester</Typography>
                  <ToolTester key={`${selectedServer.id}-${tool.function.name}`} tool={tool} canInvoke={canManage} />
                </CardContent></Card>)}
              </Stack>}
          </Box>
        </Paper>}
      </Box>}

    {createOpen && <McpServerForm onClose={() => setCreateOpen(false)} onSave={(payload) => { setCreateError(null); createMutation.mutate(payload); }} pending={createMutation.isPending} serverError={createError} />}
    {editTarget && <McpServerForm key={editTarget.id} server={editTarget} onClose={() => setEditTarget(null)} onSave={(payload) => { setCreateError(null); editMutation.mutate({ id: editTarget.id, payload }); }} pending={editMutation.isPending} serverError={createError} />}
    <ConfirmDialog open={Boolean(deleteTarget)} title="Delete MCP server?" description={`Remove “${deleteTarget?.name ?? ''}” from the gateway registry? Its connection and discovered tools will no longer be available.`} confirmLabel={deleteMutation.isPending ? 'Deleting…' : 'Delete server'} onClose={() => setDeleteTarget(null)} onConfirm={() => { if (deleteTarget) deleteMutation.mutate(deleteTarget.id); }} />
    {deleteMutation.isError && <Alert severity="error" sx={{ mt: 2 }}>{deleteMutation.error.message}</Alert>}
    <ToolCallsPanel />
  </Box>;
}
