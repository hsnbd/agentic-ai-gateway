import assert from 'node:assert/strict';

import { Given, Then, When } from '@cucumber/cucumber';

import { GatewayWorld, unique } from '../support/world';

Given('a virtual key allowing {int} request(s) in flight', async function (this: GatewayWorld, n: number) {
  const key = await this.createKey({ max_parallel_requests: n });
  this.credential = key.secret;
  this.vars.keyId = key.id;
});

Given('a virtual key allowed only on {string}', async function (this: GatewayWorld, route: string) {
  const key = await this.createKey({ allowed_routes: [route] });
  this.credential = key.secret;
  this.vars.keyId = key.id;
});

When('I send {int} slow chat requests at once', async function (this: GatewayWorld, n: number) {
  // `__slow__` makes the fake upstream take 1.5s, so the requests overlap.
  const results = await Promise.all(
    Array.from({ length: n }, (_, i) => this.chat(`__slow__ parallel ${i} ${unique('p')}`)),
  );
  this.vars.parallelStatuses = results
    .map((r) => r.status)
    .sort()
    .join(',');
});

Then('one request succeeds and one is rejected with status {int}', function (this: GatewayWorld, status: number) {
  assert.equal(this.vars.parallelStatuses, `200,${status}`);
});

Then('the streamed response contains {string}', function (this: GatewayWorld, text: string) {
  assert.ok(this.response?.text.includes(text), this.response?.text ?? 'no response');
});

Then('the streamed response does not contain {string}', function (this: GatewayWorld, text: string) {
  assert.ok(!this.response?.text.includes(text), this.response?.text ?? 'no response');
});
