import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface StatsState {
  gamesPlayed: number
  wins: number
  losses: number
  draws: number

  recordGame: (result: 'win' | 'loss' | 'draw') => void
  resetStats: () => void
}

export const useStatsStore = create<StatsState>()(
  persist(
    (set) => ({
      gamesPlayed: 0,
      wins: 0,
      losses: 0,
      draws: 0,

      recordGame: (result) => {
        set(state => ({
          gamesPlayed: state.gamesPlayed + 1,
          wins: result === 'win' ? state.wins + 1 : state.wins,
          losses: result === 'loss' ? state.losses + 1 : state.losses,
          draws: result === 'draw' ? state.draws + 1 : state.draws,
        }))
      },

      resetStats: () => {
        set({
          gamesPlayed: 0,
          wins: 0,
          losses: 0,
          draws: 0,
        })
      }
    }),
    {
      name: 'gogogo-stats'
    }
  )
)
