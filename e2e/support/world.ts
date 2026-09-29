import { IWorldOptions, World, setWorldConstructor, setDefaultTimeout } from '@cucumber/cucumber';
import type { Browser, BrowserContext, Page } from 'playwright';

export const config = {
  baseUrl: process.env.E2E_BASE_URL ?? 'http://localhost:18000',
  masterKey: process.env.E2E_MASTER_KEY ?? 'sk-e2e-master-key',
  adminEmail: process.env.E2E_ADMIN_EMAIL ?? 'admin@e2e.test',
  adminPassword: process.env.E2E_ADMIN_PASSWORD ?? 'e2e-admin-password',
  /** Where the gateway (inside Docker) reaches the fake MCP server. */
  mcpUrl: process.env.E2E_MCP_URL ?? 'http://fake-mcp:4200/mcp',
  headless: process.env.E2E_HEADED !== '1',
};

setDefaultTimeout(60_000);

export interface ApiResponse {
  status: number;
  headers: Headers;
  text: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  json: any;
}

export interface RequestOptions {
  body?: unknown;
  headers?: Record<string, string>;
  /** A virtual key, the master key, or a console JWT; sent as a Bearer token. */
  token?: string;
}

/** A process-wide browser, launched lazily by the first @ui scenario. */
export const shared: { browser?: Browser; adminToken?: string } = {};

/** Short unique suffix so scenarios never collide on the shared database. */
export function unique(prefix: string): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

export class GatewayWorld extends World {
  /** The last HTTP response, for Then steps to assert on. */
  response?: ApiResponse;
  /** Named values captured by steps: keys, ids, collection ids, request ids. */
  vars: Record<string, string> = {};
  /** The credential "I" am currently using for data-plane calls. */
  credential: string = config.masterKey;
  /** Streamed text, assembled by streaming steps. */
  streamed = '';

  context?: BrowserContext;
  page?: Page;

  constructor(options: IWorldOptions) {
    super(options);
  }

  async request(method: string, path: string, options: RequestOptions = {}): Promise<ApiResponse> {
    const headers: Record<string, string> = { ...(options.headers ?? {}) };
    const token = options.token;
    if (token) headers.Authorization = `Bearer ${token}`;
    let body: BodyInit | undefined;
    if (options.body instanceof FormData) {
      body = options.body;
    } else if (options.body !== undefined) {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(options.body);
    }
    const res = await fetch(`${config.baseUrl}${path}`, { method, headers, body });
    const text = await res.text();
    let json: unknown = undefined;
    try {
      json = text ? JSON.parse(text) : undefined;
    } catch {
      json = undefined;
    }
    this.response = { status: res.status, headers: res.headers, text, json };
    return this.response;
  }

  /** A console JWT for the bootstrap admin, cached for the whole run. */
  async adminToken(): Promise<string> {
    if (!shared.adminToken) {
      const res = await fetch(`${config.baseUrl}/admin/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: config.adminEmail, password: config.adminPassword }),
      });
      if (!res.ok) throw new Error(`admin login failed: ${res.status} ${await res.text()}`);
      shared.adminToken = ((await res.json()) as { access_token: string }).access_token;
    }
    return shared.adminToken;
  }

  async admin(method: string, path: string, body?: unknown): Promise<ApiResponse> {
    return this.request(method, `/admin/api${path}`, { body, token: await this.adminToken() });
  }

  /** Create a virtual key through the admin API and return its secret. */
  async createKey(fields: Record<string, unknown> = {}): Promise<{ secret: string; id: string }> {
    const res = await this.admin('POST', '/keys', { name: unique('e2e-key'), ...fields });
    if (res.status !== 201) throw new Error(`key creation failed: ${res.status} ${res.text}`);
    return { secret: res.json.key as string, id: res.json.id as string };
  }

  /** Create a console user and return a JWT for them. */
  async createConsoleUser(role: 'admin' | 'viewer'): Promise<{ email: string; password: string; token: string }> {
    const email = `${unique(role)}@e2e.test`;
    const password = 'e2e-user-password-123';
    const created = await this.admin('POST', '/users', { email, password, role });
    if (created.status !== 201) throw new Error(`user creation failed: ${created.status} ${created.text}`);
    const login = await this.request('POST', '/admin/api/auth/login', { body: { email, password } });
    return { email, password, token: login.json.access_token as string };
  }

  async chat(content: string, extra: Record<string, unknown> = {}, model = 'eval-chat'): Promise<ApiResponse> {
    return this.request('POST', '/v1/chat/completions', {
      token: this.credential,
      body: { model, messages: [{ role: 'user', content }], ...extra },
    });
  }

  /**
   * `<unique>` in step text becomes a fresh token (so the semantic cache, rate
   * limits, and name uniqueness never see an earlier run); `<same>` repeats it.
   */
  expand(text: string): string {
    if (text.includes('<unique>')) this.vars.unique = unique('run');
    return text.replaceAll('<unique>', this.vars.unique ?? '').replaceAll('<same>', this.vars.unique ?? '');
  }

  get currentPage(): Page {
    if (!this.page) throw new Error('This step needs a browser: tag the scenario @ui');
    return this.page;
  }
}

setWorldConstructor(GatewayWorld);
