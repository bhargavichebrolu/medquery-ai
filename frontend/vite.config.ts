import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    // jsdom gives tests a simulated browser page to render components into.
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
    // Show console output from passing tests too, so warnings such as a skipped
    // archive comparison are always visible.
    silent: false,
  },
})
