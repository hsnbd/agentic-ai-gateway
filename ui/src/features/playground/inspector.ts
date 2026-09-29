import type { InspectorData } from './types';

type JsonRecord = Record<string, unknown>;
const unavailable = 'Unavailable';
const isRecord = (value: unknown): value is JsonRecord => typeof value === 'object' && value !== null && !Array.isArray(value);
const pick = (sources: JsonRecord[], ...keys: string[]): unknown => {
  for (const source of sources) for (const key of keys) if (source[key] !== undefined && source[key] !== null) return source[key];
  return undefined;
};
const text = (value: unknown): string | undefined => typeof value === 'string' && value ? value : typeof value === 'number' && Number.isFinite(value) ? String(value) : undefined;
const numberValue = (value: unknown): number | undefined => typeof value === 'number' && Number.isFinite(value) ? value : typeof value === 'string' && value.trim() && Number.isFinite(Number(value)) ? Number(value) : undefined;
const ms = (value: unknown): string => { const parsed = numberValue(value); return parsed === undefined ? unavailable : `${parsed.toFixed(parsed < 10 ? 2 : 0)} ms`; };
const usd = (value: unknown): string => { const parsed = numberValue(value); return parsed === undefined ? unavailable : `$${parsed.toFixed(parsed < 0.01 ? 6 : 4)}`; };
const count = (value: unknown): string => { const parsed = numberValue(value); return parsed === undefined ? unavailable : String(parsed); };
const bool = (value: unknown): boolean | undefined => typeof value === 'boolean' ? value : value === 'hit' ? true : value === 'miss' ? false : undefined;

export const unavailableInspector: InspectorData = { requestId: unavailable, latency: unavailable, firstToken: unavailable, inputTokens: unavailable, outputTokens: unavailable, cost: unavailable, deployment: unavailable, provider: unavailable, cache: unavailable, similarity: unavailable, retries: unavailable, fallbacks: unavailable, guardrails: unavailable };

export function inspectorFromPlayground(payload: unknown, headers?: Headers): InspectorData {
  const root = isRecord(payload) ? payload : {};
  const metadata = isRecord(root.metadata) ? root.metadata : isRecord(root.inspector) ? root.inspector : {};
  const response = isRecord(root.response) ? root.response : {};
  const routing = isRecord(root.routing_decision) ? root.routing_decision : isRecord(metadata.routing_decision) ? metadata.routing_decision : {};
  const sources = [metadata, root, response, routing];
  const usage = pick(sources, 'usage');
  const usageRecord = isRecord(usage) ? usage : {};
  const cacheHit = bool(pick(sources, 'cache_hit')) ?? bool(headers?.get('X-Gateway-Cache'));
  const cacheSimilarity = numberValue(pick(sources, 'cache_similarity', 'similarity_score'));
  const attempts = numberValue(pick(sources, 'attempt_count'));
  const retries = pick(sources, 'retry_count', 'retries') ?? (attempts === undefined ? undefined : Math.max(0, attempts - 1));
  const fallbackCount = pick(sources, 'fallback_count', 'fallbacks_used');
  const fallbackUsed = pick(sources, 'fallback_used');
  const guardrails = pick(sources, 'guardrail_verdicts', 'guardrail_results', 'guardrails');
  return {
    requestId: text(pick(sources, 'request_id')) ?? text(headers?.get('X-Gateway-Request-Id')) ?? unavailable,
    latency: ms(pick(sources, 'latency_ms') ?? headers?.get('X-Gateway-Latency-Ms')),
    firstToken: ms(pick(sources, 'time_to_first_token_ms', 'ttft_ms', 'first_token_ms')),
    inputTokens: count(usageRecord.prompt_tokens ?? pick(sources, 'prompt_tokens', 'input_tokens')),
    outputTokens: count(usageRecord.completion_tokens ?? pick(sources, 'completion_tokens', 'output_tokens')),
    cost: usd(pick(sources, 'cost_usd') ?? headers?.get('X-Gateway-Cost-USD')),
    deployment: text(pick(sources, 'deployment_id', 'deployment')) ?? text(headers?.get('X-Gateway-Deployment')) ?? unavailable,
    provider: text(pick(sources, 'provider')) ?? text(headers?.get('X-Gateway-Provider')) ?? unavailable,
    cache: cacheHit === undefined ? unavailable : cacheHit ? 'hit' : 'miss',
    similarity: cacheSimilarity === undefined ? unavailable : cacheSimilarity.toFixed(3),
    retries: count(retries),
    fallbacks: fallbackCount !== undefined ? count(fallbackCount) : typeof fallbackUsed === 'boolean' ? fallbackUsed ? 'Used (count unavailable)' : '0' : unavailable,
    guardrails: guardrails === undefined ? unavailable : JSON.stringify(guardrails),
  };
}

export function mergeInspector(current: InspectorData, next: InspectorData): InspectorData {
  return Object.fromEntries(Object.entries(next).map(([key, value]) => [key, value === unavailable ? current[key as keyof InspectorData] : value])) as unknown as InspectorData;
}
