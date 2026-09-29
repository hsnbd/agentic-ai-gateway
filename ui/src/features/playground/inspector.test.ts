import { describe, expect, it } from 'vitest';
import { inspectorFromPlayground, mergeInspector, unavailableInspector } from './inspector';

describe('inspectorFromPlayground', () => {
  it('reads a unary playground response', () => {
    const data = inspectorFromPlayground({
      request_id: 'req-1',
      latency_ms: 1234.5,
      time_to_first_token_ms: 4.321,
      token_usage: { prompt_tokens: 3 },
      response: { usage: { prompt_tokens: 11, completion_tokens: 4 } },
      estimated_cost_usd: 1,
      cost_usd: 0.000123,
      routing_decision: { deployment_id: 'dep-1' },
      provider: 'openai',
      cache_hit: true,
      cache_similarity: 0.97654,
      retry_count: 1,
      fallback_count: 2,
      guardrail_results: { input: 'pass' },
    });
    expect(data).toEqual({
      requestId: 'req-1',
      latency: '1235 ms',
      firstToken: '4.32 ms',
      inputTokens: '11',
      outputTokens: '4',
      cost: '$0.000123',
      deployment: 'dep-1',
      provider: 'openai',
      cache: 'hit',
      similarity: '0.977',
      retries: '1',
      fallbacks: '2',
      guardrails: '{"input":"pass"}',
    });
  });

  it('falls back to gateway headers and derived values', () => {
    const headers = new Headers({
      'X-Gateway-Request-Id': 'req-h',
      'X-Gateway-Latency-Ms': '12',
      'X-Gateway-Cost-USD': '0.5',
      'X-Gateway-Deployment': 'dep-h',
      'X-Gateway-Provider': 'anthropic',
      'X-Gateway-Cache': 'miss',
    });
    const data = inspectorFromPlayground({ metadata: { attempt_count: '3', fallback_used: true, input_tokens: 7 } }, headers);
    expect(data).toMatchObject({
      requestId: 'req-h',
      latency: '12 ms',
      cost: '$0.5000',
      deployment: 'dep-h',
      provider: 'anthropic',
      cache: 'miss',
      retries: '2',
      fallbacks: 'Used (count unavailable)',
      inputTokens: '7',
    });
  });

  it('reads routing and cache state from the stream metadata trailer', () => {
    const data = inspectorFromPlayground({
      metadata: { routing_decision: { deployment_id: 'dep-m' }, cache_hit: 'hit' },
    });
    expect(data).toMatchObject({ deployment: 'dep-m', cache: 'hit' });
  });

  it('reads the inspector block and reports no fallback', () => {
    const data = inspectorFromPlayground({ inspector: { fallback_used: false, request_id: 42 } });
    expect(data.fallbacks).toBe('0');
    expect(data.requestId).toBe('42');
  });

  it('marks everything unavailable for unusable payloads', () => {
    expect(inspectorFromPlayground('not a record')).toEqual(unavailableInspector);
    expect(inspectorFromPlayground([1, 2])).toEqual(unavailableInspector);
    const odd = inspectorFromPlayground({ latency_ms: 'soon', request_id: '', cost_usd: Infinity, cache_hit: 'maybe' });
    expect(odd).toMatchObject({ latency: 'Unavailable', requestId: 'Unavailable', cost: 'Unavailable', cache: 'Unavailable' });
  });
});

describe('mergeInspector', () => {
  it('keeps known values when the next update does not have them', () => {
    const current = { ...unavailableInspector, requestId: 'req-1', provider: 'openai' };
    const next = { ...unavailableInspector, provider: 'anthropic', latency: '5 ms' };
    expect(mergeInspector(current, next)).toMatchObject({ requestId: 'req-1', provider: 'anthropic', latency: '5 ms' });
  });
});
