import assert from 'node:assert/strict';

import { Given, Then, When } from '@cucumber/cucumber';

import { GatewayWorld, unique } from '../support/world';

/** How long the e2e stack's fake MCP server keeps a session (--session-ttl). */
const MCP_SESSION_TTL_MS = 5_000;

// ---------------------------------------------------------------- MCP governance

Given('a virtual key allowed only its {string} MCP tool', async function (this: GatewayWorld, tool: string) {
  const key = await this.createKey({ allowed_tools: [`${this.vars.mcpPrefix}__${tool}`] });
  this.credential = key.secret;
});

Then('the only MCP tool listed is its {string} tool', async function (this: GatewayWorld, tool: string) {
  const res = await this.request('GET', `/v1/mcp/tools?server_id=${this.vars.mcpServerId}`, {
    token: this.credential,
  });
  assert.equal(res.status, 200, res.text);
  const names = (res.json as Array<{ function: { name: string } }>).map((t) => t.function.name);
  assert.deepEqual(names, [`${this.vars.mcpPrefix}__${tool}`]);
});

Then('the tool result does not contain {string}', function (this: GatewayWorld, text: string) {
  const content = JSON.stringify(this.response?.json?.message?.content);
  assert.ok(!content.includes(text), `tool result: ${content}`);
});

Then(
  'the tool-call audit log shows an {string} call to its {string} tool',
  async function (this: GatewayWorld, status: string, tool: string) {
    const name = `${this.vars.mcpPrefix}__${tool}`;
    const res = await this.admin('GET', `/tool-calls?tool=${encodeURIComponent(name)}`);
    assert.equal(res.status, 200, res.text);
    const statuses = (res.json.items as Array<{ status: string; source: string }>).map((row) => row.status);
    assert.ok(statuses.includes(status), `audit: ${res.text}`);
  },
);

When("the MCP server's session expires", async function () {
  await new Promise((resolve) => setTimeout(resolve, MCP_SESSION_TTL_MS + 500));
});

// ---------------------------------------------------------------- RAG tenancy

async function teamKey(world: GatewayWorld, team: string): Promise<string> {
  const slot = `team:${team}`;
  if (!world.vars[slot]) {
    const res = await world.admin('POST', '/teams', { name: unique(`e2e-${team}`) });
    assert.equal(res.status, 201, res.text);
    world.vars[slot] = res.json.id;
  }
  return (await world.createKey({ team_id: world.vars[slot] })).secret;
}

Given(
  'a RAG collection owned by team {string} with a document about {string}',
  async function (this: GatewayWorld, team: string, topic: string) {
    const token = await teamKey(this, team);
    const created = await this.request('POST', '/v1/rag/collections', {
      token,
      body: { name: unique('e2e-team-collection') },
    });
    assert.equal(created.status, 201, created.text);
    assert.ok(created.json.owner_team_id, created.text);
    this.vars.collectionId = created.json.id;
    const doc = await this.request('POST', `/v1/rag/collections/${created.json.id}/documents`, {
      token,
      body: { title: topic, content: `Facts about the ${topic}.`, source: 'topic.md' },
    });
    assert.equal(doc.status, 201, doc.text);
  },
);

async function searchAs(world: GatewayWorld, team: string, query: string): Promise<void> {
  await world.request('POST', '/v1/rag/search', {
    token: await teamKey(world, team),
    body: { collection_id: world.vars.collectionId, query },
  });
}

When('a key from team {string} searches that collection', async function (this: GatewayWorld, team: string) {
  await searchAs(this, team, 'anything');
});

When(
  'a key from team {string} searches that collection for {string}',
  async function (this: GatewayWorld, team: string, query: string) {
    await searchAs(this, team, query);
  },
);

When('I add a document to that collection', async function (this: GatewayWorld) {
  await this.request('POST', `/v1/rag/collections/${this.vars.collectionId}/documents`, {
    token: this.credential,
    body: { title: 'new', content: 'A new fact.', source: 'new.md' },
  });
});

When('I reindex that collection', async function (this: GatewayWorld) {
  await this.request('POST', `/v1/rag/collections/${this.vars.collectionId}/reindex`, {
    token: this.credential,
  });
});

Then('the collection index is in sync with {int} chunk(s)', async function (this: GatewayWorld, chunks: number) {
  const deadline = Date.now() + 10_000;
  let status: Record<string, unknown> = {};
  while (Date.now() < deadline) {
    const res = await this.request('GET', `/v1/rag/collections/${this.vars.collectionId}/index`, {
      token: this.credential,
    });
    status = res.json;
    if ((status.reindex as { status?: string } | null)?.status === 'completed' && status.in_sync) break;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  assert.equal(status.in_sync, true, JSON.stringify(status));
  assert.equal(status.indexed_chunks, chunks, JSON.stringify(status));
});
