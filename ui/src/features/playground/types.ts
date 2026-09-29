export interface PlaygroundMessage { id: string; role: 'system' | 'user' | 'assistant' | 'tool'; content: string; toolCalls?: ToolCall[]; toolCallId?: string }
export interface ToolCall { id: string; name: string; arguments: string; result?: string }
export interface InspectorData { requestId: string; latency: string; firstToken: string; inputTokens: string; outputTokens: string; cost: string; deployment: string; provider: string; cache: string; similarity: string; retries: string; fallbacks: string; guardrails: string }
export interface RerievalCitation { id: string; text: string; score: number; source: string | null; document_id: string | null }
export interface PlaygroundCollection { id: string; name: string; document_count: number }
export interface ReplayDetail { request_body: Record<string, unknown> | null; response_body: Record<string, unknown> | null; model: string; resolved_model: string | null; deployment_id: string | null; provider: string | null; stream: boolean }
