import { afterEach, describe, expect, it, vi } from 'vitest';
import { session } from '../../api/client';
import { StreamHttpError, streamChat } from './sse';

/** A streamed response whose body arrives in the given pieces. */
function streamed(pieces: string[], init: ResponseInit = {}): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const piece of pieces) controller.enqueue(encoder.encode(piece));
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' }, ...init });
}

function mockFetch(...responses: Response[]) {
  const spy = vi.fn(() => Promise.resolve(responses.shift() as Response));
  vi.stubGlobal('fetch', spy);
  return spy;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('streamChat', () => {
  it('collects text from chat deltas across split frames and stops at [DONE]', async () => {
    session.set('token');
    const spy = mockFetch(streamed([
      'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
      'data: {"choices":[{"delta":{"content":"lo"}}]',
      '}\n\nevent: metadata\ndata: {"request_id":"req-1"}\r\n\r\n',
      'data: [DONE]\n\ndata: {"choices":[{"delta":{"content":"ignored"}}]}\n\n',
    ]));
    const events: unknown[] = [];
    const result = await streamChat('/admin/api/playground/chat', { a: 1 }, (event) => events.push(event));
    expect(result.text).toBe('Hello');
    expect(events).toEqual([
      { event: 'message', data: { choices: [{ delta: { content: 'Hel' } }] } },
      { event: 'message', data: { choices: [{ delta: { content: 'lo' } }] } },
      { event: 'metadata', data: { request_id: 'req-1' } },
    ]);
    const init = spy.mock.calls[0] as unknown as [string, RequestInit];
    expect(new Headers(init[1].headers).get('Authorization')).toBe('Bearer token');
  });

  it('reads message content, plain text, and a trailing frame without a blank line', async () => {
    mockFetch(streamed([
      'data: {"content":"A"}\n\n',
      'data: {"choices":[{"message":{"content":"B"}}]}\n\n',
      'data: not json\n\n',
      'event:\ndata: {"choices":[]}\n\n',
      ': comment only\n\n',
      'data: {"choices":[null]}\n\n',
      'data: {"choices":[{"delta":{}}]}\n\n',
      'data: {"choices":[{"delta":{"content":"C"}}]}',
    ]));
    const result = await streamChat('/x', {}, () => undefined);
    expect(result.text).toBe('ABC');
    expect(result.events).toContain('not json');
  });

  it('ignores non-object payloads when extracting text', async () => {
    mockFetch(streamed(['data: 42\n\n', 'data: null\n\n']));
    const result = await streamChat('/x', {}, () => undefined);
    expect(result).toMatchObject({ text: '', events: [42, null] });
  });

  it('raises in-band error events', async () => {
    mockFetch(streamed(['data: {"content":"partial"}\n\n', 'event: error\ndata: {"error":{"message":"upstream died"}}\n\n']));
    const error = await streamChat('/x', {}, () => undefined).catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(StreamHttpError);
    expect(error).toMatchObject({ message: 'upstream died', status: 200, name: 'StreamHttpError' });
  });

  it('raises HTTP errors with JSON or text payloads', async () => {
    mockFetch(new Response(JSON.stringify({ error: { message: 'bad model' } }), {
      status: 404, headers: { 'content-type': 'application/json' },
    }));
    await expect(streamChat('/x', {}, () => undefined)).rejects.toMatchObject({ message: 'bad model', status: 404 });

    mockFetch(new Response('upstream text', { status: 502 }));
    await expect(streamChat('/x', {}, () => undefined)).rejects.toMatchObject({ message: 'upstream text' });

    // No body at all, so no content type either.
    mockFetch(new Response(null, { status: 503 }));
    await expect(streamChat('/x', {}, () => undefined)).rejects.toMatchObject({ status: 503, payload: '' });
  });

  it('renews an expired token once', async () => {
    session.set('expired', 'refresh');
    const refreshed = new Response(JSON.stringify({ access_token: 'fresh' }), { headers: { 'content-type': 'application/json' } });
    const spy = mockFetch(new Response('expired', { status: 401 }), refreshed, streamed(['data: {"content":"ok"}\n\n']));
    const result = await streamChat('/x', {}, () => undefined);
    expect(result.text).toBe('ok');
    expect(spy).toHaveBeenCalledTimes(3);
  });

  it('rejects a successful response without a body', async () => {
    mockFetch(new Response(null, { status: 200 }));
    await expect(streamChat('/x', {}, () => undefined)).rejects.toThrow('empty stream');
  });
});

describe('StreamHttpError', () => {
  it('derives messages from every payload shape', () => {
    const headers = new Headers();
    expect(new StreamHttpError(500, { error: {} }, headers).message).toBe('Gateway request failed');
    expect(new StreamHttpError(500, { other: 1 }, headers).message).toBe('Gateway request failed with status 500');
    expect(new StreamHttpError(500, null, headers).message).toBe('Gateway request failed with status 500');
  });
});
