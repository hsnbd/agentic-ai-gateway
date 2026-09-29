import { After, AfterAll, Before, BeforeAll, Status } from '@cucumber/cucumber';
import { chromium } from 'playwright';

import { GatewayWorld, config, shared } from './world';

BeforeAll({ timeout: 180_000 }, async function () {
  // The stack may still be starting (image build, migrations, index creation).
  const deadline = Date.now() + 170_000;
  let last = '';
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${config.baseUrl}/readyz`);
      if (res.ok) return;
      last = `${res.status} ${await res.text()}`;
    } catch (error) {
      last = String(error);
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error(
    `Gateway at ${config.baseUrl} never became ready (${last}). ` +
      'Start it with: npm run stack:up',
  );
});

Before({ tags: '@ui' }, async function (this: GatewayWorld) {
  shared.browser ??= await chromium.launch({ headless: config.headless });
  // Wide enough that data grids render every column (they virtualise the rest).
  this.context = await shared.browser.newContext({
    baseURL: config.baseUrl,
    viewport: { width: 1920, height: 1080 },
  });
  // A trace (DOM snapshots, network, console per action) is kept for failed
  // scenarios: open it with `npx playwright show-trace reports/traces/<file>`.
  await this.context.tracing.start({ screenshots: true, snapshots: true });
  this.page = await this.context.newPage();
  this.page.setDefaultTimeout(15_000);
});

After({ tags: '@ui' }, async function (this: GatewayWorld, { result, pickle }) {
  const failed = result?.status === Status.FAILED;
  if (failed && this.page) {
    const screenshot = await this.page.screenshot({ fullPage: true }).catch(() => undefined);
    if (screenshot) this.attach(screenshot, 'image/png');
    this.attach(`URL at failure: ${this.page.url()}`, 'text/plain');
  }
  if (this.context) {
    const name = pickle.name.replace(/[^A-Za-z0-9]+/g, '-').slice(0, 80);
    await this.context.tracing
      .stop(failed ? { path: `reports/traces/${name}-${Date.now()}.zip` } : undefined)
      .catch(() => undefined);
    await this.context.close();
  }
});

AfterAll(async function () {
  await shared.browser?.close();
});
