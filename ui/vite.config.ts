import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/ui/',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/admin': 'http://localhost:8000',
      '/v1': 'http://localhost:8000',
      '/healthz': 'http://localhost:8000',
      '/metrics': 'http://localhost:8000',
    },
  },
  build: { outDir: '../app/ui_static', emptyOutDir: true },
});
