import { afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'
import 'vitest-localstorage-mock'

// Cleanup after each test
afterEach(() => {
  cleanup()
  localStorage.clear()
  vi.clearAllMocks()
})
