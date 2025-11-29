import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/gogogo/', // GitHub Pages base path
  test: {
    globals: true,
    environment: 'jsdom',
    exclude: [
      '**/node_modules/**',
      '**/dist/**',
      '**/build/**',
      '**/archive/**', // Exclude archived code from tests
      '**/.{idea,git,cache,output,temp}/**'
    ]
  }
})
