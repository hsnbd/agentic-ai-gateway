import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { Then, When } from '@cucumber/cucumber';
import type { Locator } from 'playwright';

import { GatewayWorld } from '../support/world';

// ---------------------------------------------------------------- API

When(
  'I send a chat request for model {string} saying {string} with conditional routing and tags {string}',
  async function (this: GatewayWorld, model: string, text: string, tags: string) {
    await this.chat(
      this.expand(text),
      { routing_strategy: 'conditional', tags: tags.split(',').map((t) => t.trim()), no_cache: true },
      model,
    );
  },
);

When(
  'I send a chat request for model {string} saying {string} under guardrail policy {string}',
  async function (this: GatewayWorld, model: string, text: string, policy: string) {
    await this.chat(this.expand(text), { guardrail_policy: policy, no_cache: true }, model);
  },
);

Then('the admin cache stats report the latency saved', async function (this: GatewayWorld) {
  const res = await this.admin('GET', '/cache/stats');
  assert.equal(res.status, 200, res.text);
  assert.equal(typeof res.json.estimated_latency_saved_ms, 'number', res.text);
});

// ---------------------------------------------------------------- Teams UI

function teamRow(world: GatewayWorld, name: string): Locator {
  return world.currentPage.getByRole('row').filter({ has: world.currentPage.getByRole('gridcell', { name, exact: true }) });
}

When(
  'I create a team named {string} with a budget of {int} USD',
  async function (this: GatewayWorld, name: string, budget: number) {
    const page = this.currentPage;
    await page.getByRole('button', { name: 'Create team' }).click();
    const dialog = page.getByRole('dialog');
    await dialog.getByRole('textbox', { name: 'Team name' }).fill(this.expand(name));
    await dialog.getByRole('spinbutton', { name: 'Max budget (USD)' }).fill(String(budget));
    await dialog.getByRole('button', { name: 'Create team' }).click();
    await dialog.waitFor({ state: 'detached' });
  },
);

Then(
  'the team {string} is listed with budget {string}',
  async function (this: GatewayWorld, name: string, budget: string) {
    await teamRow(this, this.expand(name)).getByText(budget).waitFor();
  },
);

When('I rename the team {string} to {string}', async function (this: GatewayWorld, name: string, next: string) {
  const current = this.expand(name);
  const renamed = this.expand(next);
  const page = this.currentPage;
  await teamRow(this, current).getByRole('button', { name: `Edit ${current}` }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByRole('textbox', { name: 'Team name' }).fill(renamed);
  await dialog.getByRole('button', { name: 'Save team' }).click();
  await dialog.waitFor({ state: 'detached' });
});

When('I open the usage of team {string}', async function (this: GatewayWorld, name: string) {
  const team = this.expand(name);
  await teamRow(this, team).getByRole('button', { name: `Usage for ${team}` }).click();
});

Then('I see {string} in the dialog', async function (this: GatewayWorld, text: string) {
  await this.currentPage.getByRole('dialog').getByText(text).first().waitFor();
});

When('I close the dialog', async function (this: GatewayWorld) {
  const dialog = this.currentPage.getByRole('dialog');
  await dialog.getByRole('button', { name: 'Close' }).click();
  await dialog.waitFor({ state: 'detached' });
});

When('I delete the team {string}', async function (this: GatewayWorld, name: string) {
  const team = this.expand(name);
  const page = this.currentPage;
  await teamRow(this, team).getByRole('button', { name: `Delete ${team}` }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Delete team' }).click();
});

Then('the team {string} is not listed', async function (this: GatewayWorld, name: string) {
  await teamRow(this, this.expand(name)).waitFor({ state: 'detached' });
});

// ---------------------------------------------------------------- Logs export

When('I export the logs as CSV', async function (this: GatewayWorld) {
  const page = this.currentPage;
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Export CSV' }).click(),
  ]);
  assert.equal(download.suggestedFilename(), 'request-logs.csv');
  const path = await download.path();
  assert.ok(path, 'download was not saved');
  this.vars.csv = await readFile(path, 'utf8');
});

Then('the downloaded CSV has a header row and that request', function (this: GatewayWorld) {
  const [header, ...rows] = this.vars.csv.trim().split(/\r?\n/);
  assert.ok(header.startsWith('request_id,'), header);
  assert.ok(rows.some((row) => row.startsWith(this.vars.requestId)), `${this.vars.requestId} not in export`);
});
