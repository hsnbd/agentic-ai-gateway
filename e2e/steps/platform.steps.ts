import assert from 'node:assert/strict';

import { DataTable, Given, Then, When } from '@cucumber/cucumber';

import { GatewayWorld, config, unique } from '../support/world';

// ---------------------------------------------------------------- RAG

Given('a RAG collection with the documents:', async function (this: GatewayWorld, table: DataTable) {
  const created = await this.request('POST', '/v1/rag/collections', {
    token: this.credential,
    body: { name: unique('e2e-collection') },
  });
  assert.equal(created.status, 201, created.text);
  this.vars.collectionId = created.json.id;
  for (const row of table.hashes()) {
    const doc = await this.request('POST', `/v1/rag/collections/${this.vars.collectionId}/documents`, {
      token: this.credential,
      body: { title: row.title, content: row.content, source: `${row.title}.txt` },
    });
    assert.equal(doc.status, 201, doc.text);
    assert.equal(doc.json.status, 'ready', doc.text);
    this.vars[`doc:${row.title}`] = doc.json.id;
  }
});

When('I search the collection for {string}', async function (this: GatewayWorld, query: string) {
  const res = await this.request('POST', '/v1/rag/search', {
    token: this.credential,
    body: { collection_id: this.vars.collectionId, query, top_k: 3 },
  });
  assert.equal(res.status, 200, res.text);
});

Then('the top search result mentions {string}', function (this: GatewayWorld, text: string) {
  const results: Array<{ text: string }> = this.response?.json?.results ?? [];
  assert.ok(results.length > 0, 'no search results');
  assert.ok(results[0].text.includes(text), `top result: ${results[0].text}`);
});

Then('no search result mentions {string}', function (this: GatewayWorld, text: string) {
  const results: Array<{ text: string }> = this.response?.json?.results ?? [];
  assert.ok(results.every((r) => !r.text.includes(text)), JSON.stringify(results));
});

When('I query the collection with {string}', async function (this: GatewayWorld, question: string) {
  await this.request('POST', '/v1/rag/query', {
    token: this.credential,
    body: {
      collection_id: this.vars.collectionId,
      top_k: 1,
      request: { model: 'eval-chat', messages: [{ role: 'user', content: question }] },
    },
  });
});

Then('the RAG answer cites a source mentioning {string}', function (this: GatewayWorld, text: string) {
  const body = this.response?.json;
  assert.ok(body?.response?.choices?.[0]?.message?.content, this.response?.text ?? "no response");
  const sources: Array<{ text: string }> = body.sources ?? [];
  assert.ok(sources.some((s) => s.text.includes(text)), JSON.stringify(sources));
});

When('I delete the {string} document', async function (this: GatewayWorld, title: string) {
  const res = await this.request(
    'DELETE',
    `/v1/rag/collections/${this.vars.collectionId}/documents/${this.vars[`doc:${title}`]}`,
    { token: this.credential },
  );
  assert.equal(res.status, 200, res.text);
});

When('I try to create a RAG collection', async function (this: GatewayWorld) {
  await this.request('POST', '/v1/rag/collections', {
    token: this.credential,
    body: { name: unique('forbidden') },
  });
});

// ---------------------------------------------------------------- MCP

Given('the fake MCP server is registered', async function (this: GatewayWorld) {
  const prefix = unique('mcp').replaceAll('-', '_');
  const res = await this.request('POST', '/v1/mcp/servers', {
    token: config.masterKey,
    body: { name: prefix, transport: 'http', url: config.mcpUrl, tool_prefix: prefix },
  });
  assert.equal(res.status, 201, res.text);
  assert.equal(res.json.health_status, 'healthy', res.text);
  this.vars.mcpPrefix = prefix;
  this.vars.mcpServerId = res.json.id;
});

Then(
  'its tools {string}, {string} and {string} are listed',
  async function (this: GatewayWorld, a: string, b: string, c: string) {
    const res = await this.request('GET', `/v1/mcp/tools?server_id=${this.vars.mcpServerId}`, {
      token: this.credential,
    });
    assert.equal(res.status, 200, res.text);
    const names = (res.json as Array<{ function: { name: string } }>).map((t) => t.function.name);
    for (const tool of [a, b, c]) {
      assert.ok(names.includes(`${this.vars.mcpPrefix}__${tool}`), `tools: ${names.join(', ')}`);
    }
  },
);

async function callTool(world: GatewayWorld, tool: string, args: Record<string, unknown>): Promise<void> {
  const res = await world.request('POST', '/v1/mcp/tools/call', {
    token: world.credential,
    body: { name: `${world.vars.mcpPrefix}__${tool}`, arguments: args },
  });
  assert.equal(res.status, 200, res.text);
}

When('I call its {string} tool with a={int} and b={int}', async function (this: GatewayWorld, tool: string, a: number, b: number) {
  await callTool(this, tool, { a, b });
});

When('I call its {string} tool', async function (this: GatewayWorld, tool: string) {
  await callTool(this, tool, {});
});

When('I call its {string} tool with text {string}', async function (this: GatewayWorld, tool: string, text: string) {
  await callTool(this, tool, { text });
});

Then('the tool result contains {string}', function (this: GatewayWorld, text: string) {
  const content = JSON.stringify(this.response?.json?.message?.content);
  assert.ok(content.includes(text), `tool result: ${content}`);
});

When('I try to register an MCP server that runs {string}', async function (this: GatewayWorld, command: string) {
  await this.request('POST', '/v1/mcp/servers', {
    token: this.credential,
    body: { name: unique('evil'), transport: 'stdio', command, args: ['-c', 'id'] },
  });
});

// ---------------------------------------------------------------- guardrails

Then('an admin sees a {string} guardrail violation', async function (this: GatewayWorld, rule: string) {
  const res = await this.admin('GET', '/guardrails/violations?limit=50');
  const rules = (res.json.items as Array<{ rule: string }>).map((v) => v.rule);
  assert.ok(rules.includes(rule), `violations: ${rules.join(', ')}`);
});

// ---------------------------------------------------------------- observability

interface LogRow {
  request_id: string;
  model: string;
  status: string;
  stream: boolean;
  cost_usd: number;
}

async function recentLogs(world: GatewayWorld, query = ''): Promise<LogRow[]> {
  const res = await world.admin('GET', `/logs?limit=100${query}`);
  assert.equal(res.status, 200, res.text);
  return res.json.items as LogRow[];
}

/** Request logs are written after the response is sent; poll briefly. */
async function eventually<T>(probe: () => Promise<T | undefined>, what: string): Promise<T> {
  const deadline = Date.now() + 10_000;
  for (;;) {
    const value = await probe();
    if (value !== undefined) return value;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

Then(
  'the request log entry for this request is a {string} with a cost above {int}',
  async function (this: GatewayWorld, status: string, minimum: number) {
    const requestId = this.response?.headers.get('X-Gateway-Request-Id');
    assert.ok(requestId, 'no X-Gateway-Request-Id header');
    const row = await eventually(
      async () => (await recentLogs(this)).find((log) => log.request_id === requestId),
      `log for ${requestId}`,
    );
    assert.equal(row.status, status);
    assert.ok(row.cost_usd > minimum, `cost ${row.cost_usd}`);
  },
);

Then(
  'the request log has an {string} entry for model {string}',
  async function (this: GatewayWorld, status: string, model: string) {
    await eventually(
      async () => (await recentLogs(this, `&status=${status}&model=${model}`))[0],
      `a ${status} log for ${model}`,
    );
  },
);

Then('the request log has a streamed {string} entry', async function (this: GatewayWorld, status: string) {
  await eventually(
    async () => (await recentLogs(this, `&status=${status}`)).find((log) => log.stream),
    'a streamed log entry',
  );
});

Then('the metrics include {string}', async function (this: GatewayWorld, name: string) {
  const res = await this.request('GET', '/metrics');
  assert.equal(res.status, 200);
  assert.ok(res.text.includes(name), `metric ${name} missing`);
});

Then("the key's recorded spend is above 0", async function (this: GatewayWorld) {
  const spend = await eventually(async () => {
    const res = await this.admin('GET', `/keys/${this.vars.keyId}`);
    return res.json.spend_usd > 0 ? (res.json.spend_usd as number) : undefined;
  }, 'recorded spend');
  assert.ok(spend > 0);
});

When('I check readiness', async function (this: GatewayWorld) {
  await this.request('GET', '/readyz');
});

Then(
  'readiness reports {string}, {string} and {string} as healthy',
  function (this: GatewayWorld, a: string, b: string, c: string) {
    const checks = this.response?.json?.checks ?? {};
    for (const name of [a, b, c]) assert.equal(checks[name], true, JSON.stringify(checks));
  },
);
