import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '../client';

export type McpTransport = 'stdio' | 'http' | 'streamable-http';

export interface McpServer {
  id: string;
  name: string;
  description: string | null;
  transport: McpTransport;
  command: string | null;
  args: string[];
  env: Record<string, string>;
  headers: Record<string, string>;
  metadata: Record<string, unknown>;
  url: string | null;
  is_active: boolean;
  health_status: string;
  last_health_check_at: string | null;
  discovered_tools: Record<string, unknown>[];
  tool_prefix: string | null;
}

export interface McpRefreshResponse {
  server_id: string;
  healthy: boolean;
  tools: McpTool[];
  error: string | null;
  failure_kind: 'connection' | 'timeout' | 'protocol_error' | 'authentication' | 'configuration' | null;
}

export interface McpFunction {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface McpTool {
  type: 'function';
  function: McpFunction;
}

export interface ToolCallLog {
  id: string;
  created_at: string;
  request_id: string | null;
  virtual_key_id: string | null;
  team_id: string | null;
  source: 'agent' | 'direct';
  server_id: string | null;
  tool: string;
  status: string;
  duration_ms: number;
  arguments_hash: string | null;
  result_chars: number;
  truncated: boolean;
  guardrail: string | null;
  error: string | null;
}

export interface ToolCallLogPage { items: ToolCallLog[]; total: number; limit: number; offset: number }

export interface McpToolCallResponse {
  message: Record<string, unknown>;
}

export const mcpKeys = {
  all: ['mcp'] as const,
  servers: ['mcp', 'servers'] as const,
  tools: (serverId: string | undefined) => ['mcp', 'tools', serverId ?? 'all'] as const,
  toolCalls: (status: string, offset: number) => ['mcp', 'tool-calls', status, offset] as const,
};

export function useMcpServers() {
  return useQuery({
    queryKey: mcpKeys.servers,
    queryFn: () => apiRequest<McpServer[]>('/v1/mcp/servers'),
    staleTime: 15_000,
  });
}

export function useMcpTools(serverId: string | undefined) {
  const query = new URLSearchParams();
  if (serverId) query.set('server_id', serverId);
  const suffix = query.size > 0 ? `?${query.toString()}` : '';
  return useQuery({
    queryKey: mcpKeys.tools(serverId),
    queryFn: () => apiRequest<McpTool[]>(`/v1/mcp/tools${suffix}`),
    enabled: Boolean(serverId),
    staleTime: 15_000,
  });
}

/** The tool-call audit log (console users only), optionally filtered by status. */
export function useToolCalls(status: string, limit = 25, offset = 0) {
  const query = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (status) query.set('status', status);
  return useQuery({
    queryKey: mcpKeys.toolCalls(status, offset),
    queryFn: () => apiRequest<ToolCallLogPage>(`/admin/api/tool-calls?${query.toString()}`),
    staleTime: 10_000,
  });
}
