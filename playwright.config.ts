import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  timeout: 60000, // 60s timeout for neural model loading
  use: {
    baseURL: 'http://localhost:5173/gogogo/play/',
    headless: false, // Show browser!
    screenshot: 'on',
    video: 'on-first-retry',
  },
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173/gogogo/play/',
    reuseExistingServer: true,
    timeout: 30000,
  },
})
