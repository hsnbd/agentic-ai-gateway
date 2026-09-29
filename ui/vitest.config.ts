import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// Unit tests for the console's logic layer. Pages are exercised end to end by
// the Cucumber @ui scenarios in ../e2e, against the real gateway.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    restoreMocks: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html', 'json-summary'],
      include: [
        'src/api/client.ts',
        'src/auth/**',
        'src/components/ErrorBoundary.tsx',
        'src/components/Shared.tsx',
        'src/features/knowledge/LongText.tsx',
        'src/features/observability/format.ts',
        'src/features/playground/inspector.ts',
        'src/features/playground/sse.ts',
        'src/theme/**',
      ],
      thresholds: { lines: 100, branches: 100, functions: 100, statements: 100 },
    },
  },
});
