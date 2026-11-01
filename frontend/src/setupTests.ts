import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, vi } from 'vitest'

// No test talks to a real backend. By default fetch never answers (pages stay in
// their "loading" state); tests that need a response replace it with their own.
beforeEach(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn(() => new Promise<Response>(() => {})),
  )
})

// Remove anything a test rendered, any saved model choice and the fetch stand-in,
// so the next test starts from a fresh page.
afterEach(() => {
  cleanup()
  localStorage.clear()
  vi.unstubAllGlobals()
})
