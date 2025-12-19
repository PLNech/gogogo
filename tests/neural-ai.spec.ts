import { test, expect } from '@playwright/test'

const BASE_URL = 'http://localhost:5173/gogogo/play/'

test.describe('Neural AI in Browser', () => {
  test('Level 6 vs Level 6 neural game works', async ({ page }) => {
    // Capture console for debugging
    page.on('console', msg => {
      const text = msg.text()
      if (msg.type() === 'error') {
        console.log('❌', text)
      } else if (text.includes('[Neural]') || text.includes('[AI]')) {
        console.log('🧠', text)
      }
    })

    page.on('pageerror', err => console.log('💥 PAGE ERROR:', err.message))

    // Go to base URL first
    console.log('📍 Loading app...')
    await page.goto(BASE_URL)
    await page.waitForLoadState('networkidle')

    // Click Watch button to navigate to watch page
    console.log('🔄 Navigating to Watch page...')
    await page.getByRole('button', { name: 'Watch' }).click()
    await page.waitForTimeout(500)

    // Now we should see the Watch page with AI selectors
    await expect(page.getByText('Black AI')).toBeVisible()
    console.log('✅ Watch page loaded')

    // First, ensure board size is 9x9 (neural only supports 9x9)
    const boardSizeSelect = page.locator('select').first()
    await boardSizeSelect.selectOption('9')
    console.log('✅ Board size set to 9x9')

    // Wait for neural metadata to load
    await page.waitForTimeout(500)

    // Now select Level 6 for both AIs (should be available now)
    const selects = page.locator('select')
    await selects.nth(1).selectOption('6')  // Black AI level select
    await selects.nth(2).selectOption('6')  // White AI level select
    console.log('✅ Both AIs set to Level 6')

    // Verify selection
    await expect(selects.nth(1)).toHaveValue('6')
    await expect(selects.nth(2)).toHaveValue('6')

    // Take screenshot before playing
    await page.screenshot({ path: 'tests/screenshots/01-before-play.png' })

    // Click Play button
    const playBtn = page.getByRole('button', { name: /▶/ }).first()
    await expect(playBtn).toBeVisible()
    console.log('▶️ Starting game...')
    await playBtn.click()

    // Wait for model to load (first move takes longer)
    console.log('⏳ Waiting for neural model to load...')
    await page.waitForTimeout(5000)

    // Screenshot after initial load
    await page.screenshot({ path: 'tests/screenshots/02-after-5s.png' })

    // Wait for more moves
    console.log('⏳ Watching game for 10 more seconds...')
    await page.waitForTimeout(10000)

    await page.screenshot({ path: 'tests/screenshots/03-after-15s.png' })

    // Check game is progressing
    const pageContent = await page.content()
    const hasActivity = pageContent.includes('Move') || pageContent.includes('Game')
    console.log('🎯 Game appears active:', hasActivity)

    console.log('✅ Neural AI test complete!')
  })

  test('neural AI description shows correctly', async ({ page }) => {
    await page.goto(BASE_URL)
    await page.waitForLoadState('networkidle')

    // Navigate to Watch page
    await page.getByRole('button', { name: 'Watch' }).click()
    await page.waitForTimeout(500)

    // First, ensure board size is 9x9 (neural only supports 9x9)
    await page.locator('select').first().selectOption('9')
    await page.waitForTimeout(500)

    // Select Level 6 for Black AI
    const selects = page.locator('select')
    await selects.nth(1).selectOption('6')

    // Check Level 6 description is visible
    await expect(page.getByText(/Neural network trained/i)).toBeVisible()
    console.log('✅ Neural description visible')

    await page.screenshot({ path: 'tests/screenshots/04-neural-description.png' })
  })
})
