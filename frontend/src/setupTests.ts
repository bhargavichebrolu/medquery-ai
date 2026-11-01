import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// Remove anything a test rendered, and any saved model choice, so the next
// test starts from a fresh page.
afterEach(() => {
  cleanup()
  localStorage.clear()
})
