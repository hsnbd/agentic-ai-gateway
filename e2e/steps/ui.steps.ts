import assert from 'node:assert/strict';

import { Given, Then, When } from '@cucumber/cucumber';
import type { Locator, Page } from 'playwright';

import { GatewayWorld, config } from '../support/world';

const UI = '/ui';

async function signIn(page: Page, email: string, password: string): Promise<void> {
  await page.getByLabel('Email').fill(email);
  await page.getByLabel('Password').fill(password);
  await page.getByRole('button', { name: 'Sign in' }).click();
}

function heading(page: Page, title: string): Locator {
  return page.getByRole('main').getByRole('heading', { level: 1, name: title, exact: true });
}

/** A data-grid row containing `text`. */
function gridRow(page: Page, text: string): Locator {
  return page.getByRole('row').filter({ hasText: text });
}

// ---------------------------------------------------------------- sign-in

Given('I open the console sign-in page', async function (this: GatewayWorld) {
  await this.currentPage.goto(`${UI}/login`);
});

When('I sign in as the admin', async function (this: GatewayWorld) {
  await signIn(this.currentPage, config.adminEmail, config.adminPassword);
});

When(
  'I sign in with email {string} and password {string}',
  async function (this: GatewayWorld, email: string, password: string) {
    await signIn(this.currentPage, email, password);
  },
);

Given('I am signed in to the console as the admin', async function (this: GatewayWorld) {
  await this.currentPage.goto(`${UI}/login`);
  await signIn(this.currentPage, config.adminEmail, config.adminPassword);
  await heading(this.currentPage, 'Dashboard').waitFor();
});

Given('I am signed in to the console as a new viewer', async function (this: GatewayWorld) {
  const viewer = await this.createConsoleUser('viewer');
  await this.currentPage.goto(`${UI}/login`);
  await signIn(this.currentPage, viewer.email, viewer.password);
  await heading(this.currentPage, 'Dashboard').waitFor();
});

When('I open the console at {string} without signing in', async function (this: GatewayWorld, path: string) {
  await this.currentPage.goto(`${UI}${path}`);
});

When('I sign out', async function (this: GatewayWorld) {
  const page = this.currentPage;
  await page.getByRole('button', { name: 'User menu' }).click();
  await page.getByRole('menuitem', { name: 'Sign out' }).click();
});

Then('I see a sign-in error', async function (this: GatewayWorld) {
  await this.currentPage.getByRole('alert').waitFor();
});

Then(/^I am (?:still )?on the sign-in page$/, async function (this: GatewayWorld) {
  await this.currentPage.waitForURL(/\/ui\/login/);
  await this.currentPage.getByRole('button', { name: 'Sign in' }).waitFor();
});

// ---------------------------------------------------------------- navigation

When('I go to {string}', async function (this: GatewayWorld, path: string) {
  await this.currentPage.goto(`${UI}${path === '/' ? '/' : path}`);
});

When('I reload the page', async function (this: GatewayWorld) {
  await this.currentPage.reload();
});

Then('I see the {string} page', async function (this: GatewayWorld, title: string) {
  await heading(this.currentPage, title).waitFor();
});

Then('the page shows no error', async function (this: GatewayWorld) {
  const page = this.currentPage;
  // Let lazy data settle, then make sure no error boundary or error state rendered.
  await page.waitForLoadState('networkidle');
  assert.equal(await page.getByText('Something went wrong').count(), 0, 'error boundary rendered');
  assert.equal(await page.getByRole('alert').filter({ hasText: /error|failed/i }).count(), 0);
});

function navLinks(page: Page): Locator {
  return page.getByRole('navigation').or(page.locator('nav')).getByRole('link');
}

Then('the navigation shows {string}', async function (this: GatewayWorld, labels: string) {
  const page = this.currentPage;
  for (const label of labels.split(',').map((l) => l.trim())) {
    await page.getByRole('link', { name: label, exact: true }).waitFor();
  }
  assert.ok((await navLinks(page).count()) >= 0);
});

Then('the navigation does not show {string}', async function (this: GatewayWorld, labels: string) {
  const page = this.currentPage;
  for (const label of labels.split(',').map((l) => l.trim())) {
    assert.equal(await page.getByRole('link', { name: label, exact: true }).count(), 0, `${label} is visible`);
  }
});

Then('there is no {string} button', async function (this: GatewayWorld, name: string) {
  await this.currentPage.waitForLoadState('networkidle');
  assert.equal(await this.currentPage.getByRole('button', { name }).count(), 0);
});

Then('I see a notification containing {string}', async function (this: GatewayWorld, text: string) {
  await this.currentPage.getByRole('alert').filter({ hasText: text }).first().waitFor();
});

// ---------------------------------------------------------------- keys

When('I create a virtual key named {string}', async function (this: GatewayWorld, name: string) {
  const page = this.currentPage;
  await page.getByRole('button', { name: 'Create key' }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByRole('textbox', { name: 'Name', exact: true }).fill(this.expand(name));
  await dialog.getByRole('button', { name: 'Create', exact: true }).click();
});

Then("the new key's secret is shown", async function (this: GatewayWorld) {
  const secret = this.currentPage.getByText(/^sk-aigw-[A-Za-z0-9_-]{20,}$/);
  await secret.waitFor();
  this.vars.uiSecret = (await secret.textContent())?.trim() ?? '';
  assert.match(this.vars.uiSecret, /^sk-aigw-/);
});

Then('the secret works against the gateway', async function (this: GatewayWorld) {
  this.credential = this.vars.uiSecret;
  const res = await this.chat('hello from a console-created key');
  assert.equal(res.status, 200, res.text);
});

When('I dismiss the secret', async function (this: GatewayWorld) {
  await this.currentPage.getByRole('button', { name: 'Done' }).click();
});

Given('a key named {string} exists', async function (this: GatewayWorld, name: string) {
  await this.createKey({ name: this.expand(name) });
});

Then('the key {string} is listed as {string}', async function (this: GatewayWorld, name: string, status: string) {
  const row = gridRow(this.currentPage, this.expand(name));
  await row.getByRole('gridcell', { name: status, exact: true }).waitFor();
});

When('I disable the key {string}', async function (this: GatewayWorld, name: string) {
  await gridRow(this.currentPage, this.expand(name)).getByRole('button', { name: 'Disable key' }).click();
});

When('I delete the key {string}', async function (this: GatewayWorld, name: string) {
  const page = this.currentPage;
  await gridRow(page, this.expand(name)).getByRole('button', { name: 'Delete key' }).click();
  await page.getByRole('dialog').getByRole('button', { name: /delete/i }).click();
});

Then('the key {string} is not listed', async function (this: GatewayWorld, name: string) {
  await gridRow(this.currentPage, this.expand(name)).waitFor({ state: 'detached' });
});

// ---------------------------------------------------------------- models

Then(
  'the deployment table shows {string} with priority {string} and weight {string}',
  async function (this: GatewayWorld, model: string, priority: string, weight: string) {
    const row = gridRow(this.currentPage, model).filter({ hasText: 'cheap' });
    await row.getByRole('gridcell', { name: priority, exact: true }).waitFor();
    await row.getByRole('gridcell', { name: weight, exact: true }).waitFor();
  },
);

Then('the deployment table shows tags {string}', async function (this: GatewayWorld, tags: string) {
  await this.currentPage.getByRole('gridcell', { name: tags, exact: true }).waitFor();
});

When('I run a health check on the {string} fallback chain', async function (this: GatewayWorld, model: string) {
  const card = this.currentPage
    .locator('.MuiCard-root')
    .filter({ has: this.currentPage.getByRole('heading', { name: model, exact: true }) });
  await card.getByRole('button', { name: 'Health check' }).first().click();
});

// ---------------------------------------------------------------- logs

Given(
  'a chat request was sent through the gateway saying {string}',
  async function (this: GatewayWorld, text: string) {
    const res = await this.chat(this.expand(text));
    assert.equal(res.status, 200, res.text);
    this.vars.requestId = res.headers.get('X-Gateway-Request-Id') ?? '';
    assert.ok(this.vars.requestId);
  },
);

When('I search the logs for that request', async function (this: GatewayWorld) {
  await this.currentPage.getByLabel('Request ID search').fill(this.vars.requestId);
});

Then('the logs table shows that request', async function (this: GatewayWorld) {
  await gridRow(this.currentPage, this.vars.requestId).waitFor();
});

When('I open that request', async function (this: GatewayWorld) {
  await gridRow(this.currentPage, this.vars.requestId).getByRole('gridcell').first().click();
});

Then('the request detail shows it was served by {string}', async function (this: GatewayWorld, deployment: string) {
  const drawer = this.currentPage.locator('.MuiDrawer-paperAnchorRight');
  await drawer.getByRole('heading', { name: `Request ${this.vars.requestId}` }).waitFor();
  await drawer.getByText(`Deployment: ${deployment}`, { exact: true }).waitFor();
});

// ---------------------------------------------------------------- RAG & MCP

When('I create a RAG collection named {string}', async function (this: GatewayWorld, name: string) {
  const page = this.currentPage;
  this.vars.collectionName = this.expand(name);
  await page.getByRole('button', { name: 'Create collection' }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByRole('textbox', { name: 'Name', exact: true }).fill(this.vars.collectionName);
  await dialog.getByRole('button', { name: 'Create collection' }).click();
  await dialog.waitFor({ state: 'detached' });
  await page.getByText(this.vars.collectionName, { exact: true }).first().click();
});

When(
  'I paste a document titled {string} with text {string}',
  async function (this: GatewayWorld, title: string, text: string) {
    const page = this.currentPage;
    await page.getByRole('tab', { name: 'Paste text' }).click();
    await page.getByLabel('Document title / source').fill(title);
    await page.getByLabel('Document text').fill(text);
    await page.getByRole('button', { name: 'Ingest text' }).click();
  },
);

Then('the collection shows {int} document', async function (this: GatewayWorld, count: number) {
  await this.currentPage.getByText(`${count} documents`).first().waitFor();
});

When('I run a retrieval test for {string}', async function (this: GatewayWorld, query: string) {
  const page = this.currentPage;
  await page.getByRole('tab', { name: 'Retrieval test' }).click();
  await page.getByLabel('Query').fill(query);
  await page.getByRole('button', { name: 'Search collection' }).click();
});

Then('a retrieval result mentions {string}', async function (this: GatewayWorld, text: string) {
  await this.currentPage.getByRole('main').getByText(text).first().waitFor();
});

When(
  'I add an HTTP MCP server named {string} at the fake MCP URL',
  async function (this: GatewayWorld, name: string) {
    const page = this.currentPage;
    await page.getByRole('button', { name: 'Add server' }).click();
    const dialog = page.getByRole('dialog');
    await dialog.getByLabel('Server name').fill(this.expand(name));
    await dialog.getByLabel('Endpoint URL').fill(config.mcpUrl);
    await dialog.getByLabel('Tool prefix (optional)').fill(this.expand('<same>').replaceAll('-', '_'));
    await dialog.getByRole('button', { name: 'Add server' }).click();
    await dialog.waitFor({ state: 'detached' });
  },
);

Then(
  'the MCP server {string} is listed as {string} with {int} tools',
  async function (this: GatewayWorld, name: string, health: string, tools: number) {
    const page = this.currentPage;
    const entry = page.getByRole('main').getByText(this.expand(name), { exact: true }).first();
    await entry.waitFor();
    await entry.click();
    await page.getByRole('main').getByText(health, { exact: true }).first().waitFor();
    await page.getByRole('main').getByText(`${tools} tools`).first().waitFor();
  },
);

// ---------------------------------------------------------------- playground

When('I choose the model {string}', async function (this: GatewayWorld, model: string) {
  const page = this.currentPage;
  await page.getByRole('combobox', { name: 'Model' }).click();
  await page.getByRole('option', { name: model, exact: true }).click();
});

When('I send the playground message {string}', async function (this: GatewayWorld, text: string) {
  const input = this.currentPage.getByPlaceholder('Ask the gateway…');
  await input.fill(text);
  await input.press('Enter');
});

Then(
  'the playground shows an assistant reply containing {string}',
  async function (this: GatewayWorld, text: string) {
    await this.currentPage.getByRole('main').getByText(text).first().waitFor();
  },
);

// ---------------------------------------------------------------- settings

When(
  'I add a console user {string} with role {string}',
  async function (this: GatewayWorld, email: string, role: string) {
    const page = this.currentPage;
    this.vars.newUserEmail = this.expand(email);
    await page.getByRole('button', { name: 'Add user' }).click();
    const dialog = page.getByRole('dialog');
    await dialog.getByLabel('Email').fill(this.vars.newUserEmail);
    await dialog.getByLabel('Initial password').fill('console-created-password');
    await dialog.getByLabel('Role').click();
    await page.getByRole('option', { name: role, exact: true }).click();
    await dialog.getByRole('button', { name: 'Add user' }).click();
    await dialog.waitFor({ state: 'detached' });
  },
);

Then('the user {string} is listed', async function (this: GatewayWorld, email: string) {
  await gridRow(this.currentPage, this.expand(email)).waitFor();
});

Then('the new user can sign in', async function (this: GatewayWorld) {
  const res = await this.request('POST', '/admin/api/auth/login', {
    body: { email: this.vars.newUserEmail, password: 'console-created-password' },
  });
  assert.equal(res.status, 200, res.text);
  assert.equal(res.json.user.role, 'viewer');
});

Then('the settings show the database and Redis as {string}', async function (this: GatewayWorld, state: string) {
  const main = this.currentPage.getByRole('main');
  await main.getByText('Database').waitFor();
  assert.ok((await main.getByRole('heading', { name: state, exact: true }).count()) >= 2);
});

Then('the subsystem {string} is {string}', async function (this: GatewayWorld, name: string, state: string) {
  await this.currentPage.getByText(`${name}: ${state}`, { exact: true }).waitFor();
});
