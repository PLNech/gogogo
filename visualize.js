#!/usr/bin/env node

/**
 * CLI Visualization Runner
 * Compiles and runs the TypeScript visualization tool
 */

import { execSync } from 'child_process'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

console.log('\n🔨 Compiling TypeScript...')

try {
  // Compile TypeScript
  execSync('npx tsc src/cli/visualize.ts --outDir dist --module esnext --target es2022 --moduleResolution bundler --esModuleInterop', {
    stdio: 'inherit',
    cwd: __dirname
  })

  console.log('✓ Compilation complete\n')

  // Run the compiled JS
  import('./dist/cli/visualize.js')
    .then(module => {
      if (module.runDemo) {
        module.runDemo()
      }
    })
    .catch(err => {
      console.error('Error running visualization:', err)
      process.exit(1)
    })

} catch (error) {
  console.error('\n❌ Compilation failed')
  console.error(error.message)
  process.exit(1)
}
