// Cucumber profiles: `api` needs only the running stack, `ui` also drives a
// real Chromium through Playwright. Reports land in reports/.
const common = {
  requireModule: ['tsx/cjs'],
  require: ['support/**/*.ts', 'steps/**/*.ts'],
  format: [
    'progress-bar',
    'html:reports/cucumber.html',
    'junit:reports/cucumber.xml',
    'json:reports/cucumber.json',
  ],
  formatOptions: { snippetInterface: 'async-await' },
  publishQuiet: true,
};

module.exports = {
  default: { ...common, paths: ['features/**/*.feature'] },
  all: { ...common, paths: ['features/**/*.feature'] },
  api: { ...common, paths: ['features/api/**/*.feature'] },
  ui: { ...common, paths: ['features/ui/**/*.feature'] },
};
