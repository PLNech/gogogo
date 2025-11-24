import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { UpgradeId } from '../domain/upgrades/types'
import { UPGRADES, getAvailableUpgrades } from '../domain/upgrades/upgrades'
import { useCurrencyStore } from './currencyStore'

interface UpgradeStoreState {
  purchased: Set<UpgradeId>

  // Actions
  purchaseUpgrade: (id: UpgradeId) => boolean
  hasPurchased: (id: UpgradeId) => boolean
  getAvailable: () => UpgradeId[]
}

export const useUpgradeStore = create<UpgradeStoreState>()(
  persist(
    (set, get) => ({
      purchased: new Set(),

      purchaseUpgrade: (id: UpgradeId) => {
        const upgrade = UPGRADES[id]
        if (!upgrade) return false

        // Check if already purchased
        if (get().purchased.has(id)) return false

        // Check prerequisite
        if (upgrade.prerequisite && !get().purchased.has(upgrade.prerequisite)) {
          return false
        }

        // Check currency
        const currencyStore = useCurrencyStore.getState()
        if (!currencyStore.canAfford(upgrade.cost)) {
          return false
        }

        // Purchase
        if (currencyStore.spendStones(upgrade.cost)) {
          set((state) => ({
            purchased: new Set([...state.purchased, id])
          }))
          return true
        }

        return false
      },

      hasPurchased: (id: UpgradeId) => {
        return get().purchased.has(id)
      },

      getAvailable: () => {
        return getAvailableUpgrades(get().purchased)
      },
    }),
    {
      name: 'gogogo-upgrades',
      // Need to serialize Set to Array for localStorage
      partialize: (state) => ({
        purchased: Array.from(state.purchased),
      }),
      merge: (persistedState, currentState) => {
        const persisted = persistedState as { purchased?: UpgradeId[] }
        return {
          ...currentState,
          purchased: new Set(persisted.purchased || []),
        }
      },
    }
  )
)
