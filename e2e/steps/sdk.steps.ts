import assert from 'node:assert/strict';

import Anthropic from '@anthropic-ai/sdk';
import { Then, When } from '@cucumber/cucumber';
import OpenAI from 'openai';

import { GatewayWorld, config } from '../support/world';

/** Results from the official SDKs, kept per scenario. */
interface SdkState {
  completion?: OpenAI.Chat.Completions.ChatCompletion;
  embeddings?: OpenAI.Embeddings.CreateEmbeddingResponse;
  models?: string[];
  anthropicText?: string;
}

function sdk(world: GatewayWorld): SdkState {
  const holder = world as unknown as { sdk?: SdkState };
  holder.sdk ??= {};
  return holder.sdk;
}

function openai(world: GatewayWorld): OpenAI {
  return new OpenAI({ baseURL: `${config.baseUrl}/v1`, apiKey: world.credential, maxRetries: 0 });
}

function anthropic(world: GatewayWorld): Anthropic {
  return new Anthropic({ baseURL: config.baseUrl, apiKey: world.credential, maxRetries: 0 });
}

// ---------------------------------------------------------------- OpenAI SDK

When('I ask {string} {string} with the OpenAI SDK', async function (this: GatewayWorld, model: string, text: string) {
  sdk(this).completion = await openai(this).chat.completions.create({
    model,
    messages: [{ role: 'user', content: text }],
  });
});

When(
  'I ask {string} {string} with the OpenAI SDK offering the {string} tool',
  async function (this: GatewayWorld, model: string, text: string, tool: string) {
    sdk(this).completion = await openai(this).chat.completions.create({
      model,
      messages: [{ role: 'user', content: text }],
      tools: [
        {
          type: 'function',
          function: {
            name: tool,
            description: 'Look up the weather for a city',
            parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
          },
        },
      ],
    });
  },
);

When('I stream {string} {string} with the OpenAI SDK', async function (this: GatewayWorld, model: string, text: string) {
  const stream = await openai(this).chat.completions.create({
    model,
    stream: true,
    messages: [{ role: 'user', content: text }],
  });
  this.streamed = '';
  for await (const chunk of stream) {
    this.streamed += chunk.choices[0]?.delta?.content ?? '';
  }
});

When(
  'I embed {string} with {string} using the OpenAI SDK',
  async function (this: GatewayWorld, text: string, model: string) {
    sdk(this).embeddings = await openai(this).embeddings.create({ model, input: text });
  },
);

When('I list models with the OpenAI SDK', async function (this: GatewayWorld) {
  const ids: string[] = [];
  for await (const model of openai(this).models.list()) ids.push(model.id);
  sdk(this).models = ids;
});

When(
  'I ask {string} {string} with the OpenAI SDK grounded in the collection',
  async function (this: GatewayWorld, model: string, text: string) {
    sdk(this).completion = await openai(this).chat.completions.create({
      model,
      messages: [{ role: 'user', content: text }],
      // @ts-expect-error gateway extension passed through the SDK's extra body
      aigw: { rag: { collection_id: this.vars.collectionId, top_k: 1 } },
    });
  },
);

When(
  'I ask {string} {string} with the OpenAI SDK using the registered MCP server',
  async function (this: GatewayWorld, model: string, text: string) {
    sdk(this).completion = await openai(this).chat.completions.create({
      model,
      messages: [{ role: 'user', content: text }],
      // @ts-expect-error gateway extension passed through the SDK's extra body
      aigw: { mcp: { servers: [this.vars.mcpServerId] } },
    });
  },
);

/** The gateway's `aigw` extras on an SDK completion. */
function extras(world: GatewayWorld): { sources?: Array<{ text: string }>; tool_calls_executed?: number } {
  return (sdk(world).completion as unknown as { aigw?: Record<string, never> })?.aigw ?? {};
}

Then('the SDK reply cites a source mentioning {string}', function (this: GatewayWorld, text: string) {
  const sources = extras(this).sources ?? [];
  assert.ok(sources.some((source) => source.text.includes(text)), JSON.stringify(sources));
});

Then('the SDK reports {int} tool call(s) executed by the gateway', function (this: GatewayWorld, count: number) {
  assert.equal(extras(this).tool_calls_executed, count);
});

Then('the SDK reply is {string}', function (this: GatewayWorld, text: string) {
  assert.equal(sdk(this).completion?.choices[0]?.message.content, text);
});

Then('the SDK reports token usage', function (this: GatewayWorld) {
  const usage = sdk(this).completion?.usage;
  assert.ok(usage && usage.total_tokens > 0, `usage: ${JSON.stringify(usage)}`);
});

Then('the SDK reply calls the {string} tool', function (this: GatewayWorld, tool: string) {
  const choice = sdk(this).completion?.choices[0];
  assert.equal(choice?.finish_reason, 'tool_calls');
  const call = choice?.message.tool_calls?.[0];
  assert.ok(call && call.type === 'function', 'expected a function tool call');
  assert.equal(call.function.name, tool);
  JSON.parse(call.function.arguments);
});

Then('I get {int} embedding of {int} dimensions', function (this: GatewayWorld, count: number, dims: number) {
  const data = sdk(this).embeddings?.data ?? [];
  assert.equal(data.length, count);
  assert.equal(data[0]?.embedding.length, dims);
});

Then('the model list includes {string}', function (this: GatewayWorld, model: string) {
  assert.ok(sdk(this).models?.includes(model), `models: ${sdk(this).models?.join(', ')}`);
});

// ---------------------------------------------------------------- Anthropic SDK

async function sendAnthropic(world: GatewayWorld, model: string, text: string, system?: string): Promise<void> {
  const message = await anthropic(world).messages.create({
    model,
    max_tokens: 64,
    ...(system ? { system } : {}),
    messages: [{ role: 'user', content: text }],
  });
  sdk(world).anthropicText = message.content
    .map((block) => (block.type === 'text' ? block.text : ''))
    .join('');
}

When('I send {string} to {string} with the Anthropic SDK', async function (this: GatewayWorld, text: string, model: string) {
  await sendAnthropic(this, model, text);
});

When(
  'I send {string} to {string} with the Anthropic SDK and system prompt {string}',
  async function (this: GatewayWorld, text: string, model: string, system: string) {
    await sendAnthropic(this, model, text, system);
  },
);

When('I stream {string} to {string} with the Anthropic SDK', async function (this: GatewayWorld, text: string, model: string) {
  const stream = anthropic(this).messages.stream({
    model,
    max_tokens: 64,
    messages: [{ role: 'user', content: text }],
  });
  this.streamed = '';
  stream.on('text', (delta) => {
    this.streamed += delta;
  });
  await stream.finalMessage();
});

Then('the Anthropic reply is {string}', function (this: GatewayWorld, text: string) {
  assert.equal(sdk(this).anthropicText, text);
});

Then('the Anthropic reply contains {string}', function (this: GatewayWorld, text: string) {
  assert.ok(sdk(this).anthropicText?.includes(text), `reply: ${sdk(this).anthropicText}`);
});
