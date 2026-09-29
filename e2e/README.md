# End-to-end scenarios

Cucumber features (Gherkin) with TypeScript step definitions. API scenarios use
`fetch` and the official `openai` and `@anthropic-ai/sdk` clients; `@ui`
scenarios drive the console in Chromium through Playwright.

```bash
npm ci && npx playwright install chromium   # once
npm run stack:up                            # build + start the dockerised stack
npm test                                    # everything (or test:api / test:ui)
npm run stack:down
```

The stack (`docker-compose.e2e.yaml`) is the real gateway image with its
console, Postgres, Redis Stack, and two deterministic fakes from `../scripts`:
an OpenAI-compatible upstream and an MCP server. It uses
`config/models.eval.yaml`, whose `eval-chat` primary deployment is deliberately
dead, so every chat request exercises failover.

| Variable | Default | Purpose |
|---|---|---|
| `E2E_BASE_URL` | `http://localhost:18000` | Gateway under test |
| `E2E_MASTER_KEY` | `sk-e2e-master-key` | Master key configured in the stack |
| `E2E_ADMIN_EMAIL` / `E2E_ADMIN_PASSWORD` | `admin@e2e.test` / `e2e-admin-password` | Bootstrap console admin |
| `E2E_MCP_URL` | `http://fake-mcp:4200/mcp` | MCP server URL as seen from the gateway |
| `E2E_HEADED` | unset | `1` shows the browser |

Scenarios share one database, so they never depend on each other's data:
`<unique>` in step text becomes a fresh token and `<same>` repeats it.
Reports (HTML, JUnit, JSON, with screenshots of failed UI steps) are written to
`reports/`.

When a `@ui` scenario fails, a Playwright trace is saved to `reports/traces/`
(DOM snapshots, network, and console for every action). Open it with
`npx playwright show-trace reports/traces/<file>.zip`. A step that exceeds the
60-second step limit even though each browser action has a 15-second timeout
means the whole machine paused (e.g. a laptop going to sleep mid-run), not
that the app hung.
