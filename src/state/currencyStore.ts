import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface CurrencyState {
  stones: number
  earnStones: (amount: number) => void
  spendStones: (amount: number) => boolean
  canAfford: (amount: number) => boolean
}

export const useCurrencyStore = create<CurrencyState>()(
  persist(
    (set, get) => ({
      stones: 0,

      earnStones: (amount: number) => {
        set((state) => ({ stones: state.stones + amount }))
      },

      spendStones: (amount: number) => {
        const { stones } = get()
        if (stones >= amount) {
          set({ stones: stones - amount })
          return true
        }
        return false
      },

      canAfford: (amount: number) => {
        return get().stones >= amount
      },
    }),
    {
      name: 'gogogo-currency',
    }
  )
)
