import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // During development, forward every request starting with /api to the Python
    // backend, so the browser talks to one address and needs no CORS setup.
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
  test: {
    // jsdom gives tests a simulated browser page to render components into.
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
    // Show console output from passing tests too, so warnings such as a skipped
    // archive comparison are always visible.
    silent: false,
  },
})
