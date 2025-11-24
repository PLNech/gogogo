import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type GamePhase =
  | 'M1_INTRO'           // 1x1 board, place first stone
  | 'M1_CAPTURE'         // 3x3 board, capture white stone
  | 'M2_PRACTICE_1'      // Empty board practice #1
  | 'M2_PRACTICE_2'      // Empty board practice #2
  | 'M2_PRACTICE_3'      // Empty board practice #3
  | 'M2_AI_OPPONENT'     // 5x5 with AI opponent
  | 'FREE_PLAY'          // Unlocked after M2

interface GameState {
  phase: GamePhase
  gamesPlayed: number
  practiceMoves: number

  // Actions
  completePhase: () => void
  incrementPractice: () => void
  reset: () => void
}

export const useGameStore = create<GameState>()(
  persist(
    (set, get) => ({
      phase: 'M1_INTRO',
      gamesPlayed: 0,
      practiceMoves: 0,

      completePhase: () => {
        const { phase } = get()

        set({ gamesPlayed: get().gamesPlayed + 1 })

        switch (phase) {
          case 'M1_INTRO':
            set({ phase: 'M1_CAPTURE' })
            break
          case 'M1_CAPTURE':
            set({ phase: 'M2_PRACTICE_1' })
            break
          case 'M2_PRACTICE_1':
            set({ phase: 'M2_PRACTICE_2' })
            break
          case 'M2_PRACTICE_2':
            set({ phase: 'M2_PRACTICE_3' })
            break
          case 'M2_PRACTICE_3':
            set({ phase: 'M2_AI_OPPONENT' })
            break
          case 'M2_AI_OPPONENT':
            set({ phase: 'FREE_PLAY' })
            break
        }
      },

      incrementPractice: () => {
        set({ practiceMoves: get().practiceMoves + 1 })
      },

      reset: () => {
        set({ phase: 'M1_INTRO', gamesPlayed: 0, practiceMoves: 0 })
      },
    }),
    {
      name: 'gogogo-game',
    }
  )
)
