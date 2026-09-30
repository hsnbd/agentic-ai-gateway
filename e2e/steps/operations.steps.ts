import assert from 'node:assert/strict';

import { Given, Then, When } from '@cucumber/cucumber';

import { GatewayWorld, unique } from '../support/world';

/** The last response body, for assertion messages. */
function body(world: GatewayWorld): string {
  return world.response?.text ?? 'no response';
}

// ---------------------------------------------------------------- legacy completions

When(
  'I request a legacy completion for model {string} with prompt {string}',
  async function (this: GatewayWorld, model: string, prompt: string) {
    await this.request('POST', '/v1/completions', {
      token: this.credential,
      body: { model, prompt: this.expand(prompt), no_cache: true },
    });
  },
);

When(
  'I stream a legacy completion for model {string} with prompt {string}',
  async function (this: GatewayWorld, model: string, prompt: string) {
    await this.request('POST', '/v1/completions', {
      token: this.credential,
      body: { model, prompt: this.expand(prompt), stream: true, no_cache: true },
    });
  },
);

Then('the completion text is {string}', function (this: GatewayWorld, text: string) {
  assert.equal(this.response?.json?.object, 'text_completion', body(this));
  assert.equal(this.response?.json?.choices?.[0]?.text, this.expand(text));
});

Then('the streamed completion reads {string}', function (this: GatewayWorld, text: string) {
  const frames = (this.response?.text ?? '')
    .split('\n\n')
    .map((frame) => frame.replace(/^data: /, '').trim())
    .filter(Boolean);
  assert.equal(frames.at(-1), '[DONE]', body(this));
  const chunks = frames.slice(0, -1).map((frame) => JSON.parse(frame));
  assert.ok(chunks.every((chunk) => chunk.object === 'text_completion'), body(this));
  const joined = chunks.map((chunk) => chunk.choices[0].text ?? '').join('');
  assert.equal(joined.trim(), this.expand(text));
});

// ---------------------------------------------------------------- tool results

When(
  'I send {string} the result {string} of a {string} tool call',
  async function (this: GatewayWorld, model: string, result: string, tool: string) {
    await this.request('POST', '/v1/chat/completions', {
      token: this.credential,
      body: {
        model,
        no_cache: true,
        messages: [
          { role: 'user', content: 'What is the weather in Paris?' },
          {
            role: 'assistant',
            content: null,
            tool_calls: [
              { id: 'call_1', type: 'function', function: { name: tool, arguments: '{"city":"Paris"}' } },
            ],
          },
          { role: 'tool', tool_call_id: 'call_1', content: result },
        ],
      },
    });
  },
);

// ---------------------------------------------------------------- Anthropic token counting

When(
  'I count the tokens of {string} for model {string}',
  async function (this: GatewayWorld, text: string, model: string) {
    await this.request('POST', '/v1/messages/count_tokens', {
      token: this.credential,
      body: { model, messages: [{ role: 'user', content: text }] },
    });
  },
);

When('I count the tokens of a malformed Messages request', async function (this: GatewayWorld) {
  await this.request('POST', '/v1/messages/count_tokens', {
    token: this.credential,
    body: { model: 'eval-chat', messages: 'not a list' },
  });
});

Then('the input token count is between {int} and {int}', function (this: GatewayWorld, low: number, high: number) {
  const count = this.response?.json?.input_tokens;
  assert.ok(typeof count === 'number' && count >= low && count <= high, body(this));
});

Then('the Anthropic error type is {string}', function (this: GatewayWorld, type: string) {
  assert.equal(this.response?.json?.type, 'error', body(this));
  assert.equal(this.response?.json?.error?.type, type, body(this));
});

// ---------------------------------------------------------------- console sessions

async function signIn(world: GatewayWorld, email: string, password: string) {
  return world.request('POST', '/admin/api/auth/login', { body: { email, password } });
}

Given('a new console admin is signed in', async function (this: GatewayWorld) {
  const user = await this.createConsoleUser('admin');
  const login = await signIn(this, user.email, user.password);
  assert.equal(login.status, 200, login.text);
  Object.assign(this.vars, {
    email: user.email,
    password: user.password,
    access: login.json.access_token,
    refresh: login.json.refresh_token,
    userId: login.json.user.id,
  });
});

When('they renew their session', async function (this: GatewayWorld) {
  const res = await this.request('POST', '/admin/api/auth/refresh', {
    body: { refresh_token: this.vars.refresh },
  });
  assert.equal(res.status, 200, res.text);
  this.vars.renewedAccess = res.json.access_token;
});

When('they try to renew with the same refresh token again', async function (this: GatewayWorld) {
  await this.request('POST', '/admin/api/auth/refresh', { body: { refresh_token: this.vars.refresh } });
});

Then('the renewed access token works', async function (this: GatewayWorld) {
  const res = await this.request('GET', '/admin/api/auth/me', { token: this.vars.renewedAccess });
  assert.equal(res.status, 200, res.text);
  assert.equal(res.json.email, this.vars.email);
});

When('they change their password to {string}', async function (this: GatewayWorld, password: string) {
  const res = await this.request('POST', '/admin/api/users/me/change-password', {
    token: this.vars.access,
    body: { current_password: this.vars.password, new_password: password },
  });
  assert.equal(res.status, 200, res.text);
  this.vars.newPassword = password;
});

When('they try to change their password with a wrong current password', async function (this: GatewayWorld) {
  await this.request('POST', '/admin/api/users/me/change-password', {
    token: this.vars.access,
    body: { current_password: 'definitely-wrong', new_password: 'another-long-password' },
  });
});

Then('their earlier access token is rejected', async function (this: GatewayWorld) {
  const res = await this.request('GET', '/admin/api/auth/me', { token: this.vars.access });
  assert.equal(res.status, 401, res.text);
});

Then('they can sign in with the new password but not the old one', async function (this: GatewayWorld) {
  assert.equal((await signIn(this, this.vars.email, this.vars.password)).status, 401);
  assert.equal((await signIn(this, this.vars.email, this.vars.newPassword)).status, 200);
});

When('they try to demote themselves to viewer', async function (this: GatewayWorld) {
  await this.request('PATCH', `/admin/api/users/${this.vars.userId}`, {
    token: this.vars.access,
    body: { role: 'viewer' },
  });
});

When('they try to delete their own account', async function (this: GatewayWorld) {
  await this.request('DELETE', `/admin/api/users/${this.vars.userId}`, { token: this.vars.access });
});

Then('the error detail mentions {string}', function (this: GatewayWorld, text: string) {
  const detail = String(this.response?.json?.detail ?? this.response?.json?.error?.message ?? '');
  assert.ok(detail.includes(text), `expected ${JSON.stringify(detail)} to mention ${text}`);
});

// ---------------------------------------------------------------- admin operations

When('an admin reloads the model configuration', async function (this: GatewayWorld) {
  await this.admin('POST', '/config/reload');
});

Then('the reloaded catalogue includes {string}', function (this: GatewayWorld, model: string) {
  assert.equal(this.response?.status, 200, body(this));
  assert.ok((this.response?.json?.models as string[]).includes(model), body(this));
  assert.ok((this.response?.json?.deployment_count as number) > 0);
});

When('an admin lists the guardrail policies', async function (this: GatewayWorld) {
  await this.admin('GET', '/guardrails/policies');
});

Then('the guardrail policy {string} is listed', function (this: GatewayWorld, name: string) {
  const names = (this.response?.json?.items ?? []).map((policy: { name: string }) => policy.name);
  assert.ok(names.includes(name), `policies: ${names.join(', ')}`);
});

When('an admin invalidates the whole semantic cache', async function (this: GatewayWorld) {
  const res = await this.admin('POST', '/cache/invalidate', { all_entries: true });
  assert.equal(res.status, 200, res.text);
  assert.ok(res.json.invalidated >= 1, `nothing was invalidated: ${res.text}`);
});

When('an admin asks for the dashboard over {string}', async function (this: GatewayWorld, window: string) {
  await this.admin('GET', `/dashboard/summary?window=${encodeURIComponent(window)}`);
});

When('an admin asks for request logs ordered by {string}', async function (this: GatewayWorld, field: string) {
  await this.admin('GET', `/logs?order_by=${encodeURIComponent(field)}`);
});

Given('a team with a budget of {float} USD', async function (this: GatewayWorld, budget: number) {
  const res = await this.admin('POST', '/teams', { name: unique('e2e-team'), max_budget_usd: budget });
  assert.equal(res.status, 201, res.text);
  this.vars.teamId = res.json.id;
});

Given('a virtual key in that team', async function (this: GatewayWorld) {
  const key = await this.createKey({ team_id: this.vars.teamId });
  this.credential = key.secret;
  this.vars.keyId = key.id;
});

Then("the team's usage reports {int} request(s)", async function (this: GatewayWorld, count: number) {
  const deadline = Date.now() + 10_000;
  let res = await this.admin('GET', `/teams/${this.vars.teamId}/usage`);
  while (res.json?.requests < count && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 250));
    res = await this.admin('GET', `/teams/${this.vars.teamId}/usage`);
  }
  assert.equal(res.json?.requests, count, res.text);
});

// ---------------------------------------------------------------- console pages

Then('the page shows {string}', async function (this: GatewayWorld, text: string) {
  await this.currentPage.getByRole('main').getByText(this.expand(text), { exact: true }).first().waitFor();
});

When('I group usage by {string}', async function (this: GatewayWorld, label: string) {
  const page = this.currentPage;
  await page.getByRole('combobox', { name: 'Group by' }).click();
  await page.getByRole('option', { name: label, exact: true }).click();
});
