const TOKEN_KEY = 'aigateway.console.token';

export interface StreamEvent {
  event: string;
  data: unknown;
}

export interface StreamResult {
  text: string;
  events: unknown[];
  headers: Headers;
}

function contentFromPayload(payload: unknown): string {
  if (typeof payload !== 'object' || payload === null) return '';
  const record = payload as Record<string, unknown>;
  if (typeof record.content === 'string') return record.content;
  const choices = record.choices;
  if (!Array.isArray(choices) || choices.length === 0 || typeof choices[0] !== 'object' || choices[0] === null) return '';
  const choice = choices[0] as Record<string, unknown>;
  const delta = choice.delta;
  if (typeof delta === 'object' && delta !== null && typeof (delta as Record<string, unknown>).content === 'string') {
    return (delta as Record<string, string>).content;
  }
  const message = choice.message;
  if (typeof message === 'object' && message !== null && typeof (message as Record<string, unknown>).content === 'string') {
    return (message as Record<string, string>).content;
  }
  return '';
}

export async function streamChat(
  path: string,
  body: unknown,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<StreamResult> {
  const headers = new Headers({ 'Content-Type': 'application/json', Accept: 'text/event-stream' });
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const response = await fetch(path, { method: 'POST', headers, body: JSON.stringify(body), signal });
  if (!response.ok) {
    const contentType = response.headers.get('content-type') ?? '';
    const payload: unknown = contentType.includes('application/json') ? await response.json() : await response.text();
    throw new StreamHttpError(response.status, payload, response.headers);
  }
  if (!response.body) throw new Error('The gateway returned an empty stream.');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let text = '';
  const events: unknown[] = [];
  const consume = (frame: string): boolean => {
    let eventName = 'message';
    const dataLines: string[] = [];
    for (const line of frame.split(/\r?\n/)) {
      if (line.startsWith('event:')) eventName = line.slice(6).trim() || 'message';
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
    }
    const data = dataLines.join('\n');
    if (!data) return false;
    if (eventName === 'message' && data === '[DONE]') return true;
    let payload: unknown;
    try { payload = JSON.parse(data) as unknown; } catch { payload = data; }
    events.push(payload);
    if (eventName === 'message') text += contentFromPayload(payload);
    onEvent({ event: eventName, data: payload });
    return false;
  };
  try {
    let done = false;
    while (!done) {
      const next = await reader.read();
      done = next.done;
      buffer += decoder.decode(next.value, { stream: !done });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? '';
      for (const frame of frames) if (consume(frame)) return { text, events, headers: response.headers };
    }
    if (buffer.trim()) consume(buffer);
    return { text, events, headers: response.headers };
  } finally {
    reader.releaseLock();
  }
}

export class StreamHttpError extends Error {
  readonly status: number;
  readonly payload: unknown;
  readonly headers: Headers;
  constructor(status: number, payload: unknown, headers: Headers) {
    const message = typeof payload === 'object' && payload !== null && 'error' in payload
      ? String((payload as { error?: { message?: unknown } }).error?.message ?? 'Gateway request failed')
      : typeof payload === 'string' ? payload : `Gateway request failed with status ${status}`;
    super(message);
    this.name = 'StreamHttpError'; this.status = status; this.payload = payload; this.headers = headers;
  }
}
