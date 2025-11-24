import { describe, it, expect, beforeEach } from 'vitest'
import { useCurrencyStore } from '../../state/currencyStore'

describe('Currency System', () => {
  beforeEach(() => {
    // Reset store before each test
    useCurrencyStore.setState({ stones: 0 })
  })

  it('starts with 0 stones', () => {
    const { stones } = useCurrencyStore.getState()
    expect(stones).toBe(0)
  })

  it('can earn stones', () => {
    useCurrencyStore.getState().earnStones(10)
    expect(useCurrencyStore.getState().stones).toBe(10)

    useCurrencyStore.getState().earnStones(5)
    expect(useCurrencyStore.getState().stones).toBe(15)
  })

  it('can spend stones', () => {
    useCurrencyStore.getState().earnStones(100)
    const success = useCurrencyStore.getState().spendStones(30)

    expect(success).toBe(true)
    expect(useCurrencyStore.getState().stones).toBe(70)
  })

  it('cannot spend more stones than available', () => {
    useCurrencyStore.getState().earnStones(10)
    const success = useCurrencyStore.getState().spendStones(20)

    expect(success).toBe(false)
    expect(useCurrencyStore.getState().stones).toBe(10)
  })

  it('can check if can afford', () => {
    useCurrencyStore.getState().earnStones(50)

    expect(useCurrencyStore.getState().canAfford(30)).toBe(true)
    expect(useCurrencyStore.getState().canAfford(50)).toBe(true)
    expect(useCurrencyStore.getState().canAfford(51)).toBe(false)
  })
})
