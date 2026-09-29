import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The gateway the dev server proxies to: `make dev` serves on 4000.
const gateway = process.env.AIGATEWAY_URL ?? 'http://localhost:4000';

export default defineConfig({
  base: '/ui/',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/admin': gateway,
      '/v1': gateway,
      '/healthz': gateway,
      '/metrics': gateway,
    },
  },
  build: { outDir: '../app/ui_static', emptyOutDir: true },
});
