import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { IdleSourceId, IdleProgress } from '../domain/idle/types'
import { calculateIdleRate } from '../domain/idle/sources'

interface IdleStoreState {
  sources: Set<IdleSourceId>
  lastCollectionTime: number
  totalIdleEarned: number

  purchaseSource: (id: IdleSourceId, cost: number, spendStones: (amount: number) => boolean) => boolean
  calculatePending: () => IdleProgress
  collectIdle: (earnStones: (amount: number) => void) => number
  resetIdle: () => void
}

export const useIdleStore = create<IdleStoreState>()(
  persist(
    (set, get) => ({
      sources: new Set<IdleSourceId>(),
      lastCollectionTime: Date.now(),
      totalIdleEarned: 0,

      purchaseSource: (id, cost, spendStones) => {
        if (!spendStones(cost)) return false

        set(state => ({
          sources: new Set([...state.sources, id])
        }))
        return true
      },

      calculatePending: () => {
        const state = get()
        const now = Date.now()
        const elapsedMs = now - state.lastCollectionTime
        const elapsedSeconds = elapsedMs / 1000

        const ratePerSecond = calculateIdleRate(state.sources)
        const pendingStones = Math.floor(ratePerSecond * elapsedSeconds)

        return {
          pendingStones,
          elapsedSeconds,
          ratePerSecond
        }
      },

      collectIdle: (earnStones) => {
        const { pendingStones } = get().calculatePending()

        if (pendingStones > 0) {
          earnStones(pendingStones)
          set(state => ({
            lastCollectionTime: Date.now(),
            totalIdleEarned: state.totalIdleEarned + pendingStones
          }))
        }

        return pendingStones
      },

      resetIdle: () => {
        set({
          sources: new Set(),
          lastCollectionTime: Date.now(),
          totalIdleEarned: 0
        })
      }
    }),
    {
      name: 'gogogo-idle',
      partialize: (state) => ({
        sources: Array.from(state.sources),
        lastCollectionTime: state.lastCollectionTime,
        totalIdleEarned: state.totalIdleEarned,
      }),
      merge: (persistedState: any, currentState) => ({
        ...currentState,
        ...persistedState,
        sources: new Set(persistedState?.sources || []),
      }),
    }
  )
)
