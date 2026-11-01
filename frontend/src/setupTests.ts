import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// Remove anything a test rendered so the next test starts with an empty page.
afterEach(() => {
  cleanup()
})
