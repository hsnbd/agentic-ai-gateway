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

export interface McpToolCallResponse {
  message: Record<string, unknown>;
}

export const mcpKeys = {
  all: ['mcp'] as const,
  servers: ['mcp', 'servers'] as const,
  tools: (serverId: string | undefined) => ['mcp', 'tools', serverId ?? 'all'] as const,
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
