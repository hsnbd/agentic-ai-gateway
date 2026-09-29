import assert from 'node:assert/strict';

import { Given, Then, When } from '@cucumber/cucumber';

import { GatewayWorld } from '../support/world';

const TOKEN_KEY = 'aigateway.console.token';

async function storedToken(world: GatewayWorld): Promise<string> {
  const token = await world.currentPage.evaluate((key) => localStorage.getItem(key), TOKEN_KEY);
  assert.ok(token, 'no console token in the browser');
  return token;
}

When('my console access token stops being valid', async function (this: GatewayWorld) {
  // Revoke only the access token (no refresh token in the body): the browser
  // is left with exactly what an expired access token looks like.
  const res = await this.request('POST', '/admin/api/auth/logout', { token: await storedToken(this) });
  assert.equal(res.status, 200, res.text);
});

Given('I remember my console access token', async function (this: GatewayWorld) {
  this.vars.consoleToken = await storedToken(this);
});

Then('the remembered console token is rejected by the gateway', async function (this: GatewayWorld) {
  const deadline = Date.now() + 5000;
  // Logout is sent in the background as the console navigates away.
  for (;;) {
    const res = await this.request('GET', '/admin/api/auth/me', { token: this.vars.consoleToken });
    if (res.status === 401) return;
    if (Date.now() > deadline) assert.fail(`token still accepted: ${res.status}`);
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
});
