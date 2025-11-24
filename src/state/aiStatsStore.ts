import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { AIConfig } from '../core/ai/types'

export interface MatchResult {
  id: string
  timestamp: number
  boardSize: number
  maxMoves: number
  winner: 'black' | 'white' | 'draw'
  blackScore: number
  whiteScore: number
  blackCaptures: number
  whiteCaptures: number
  moveCount: number
  blackConfig: AIConfig
  whiteConfig: AIConfig
}

export interface AIStats {
  configSignature: string // Stringified config for matching
  wins: number
  losses: number
  draws: number
  totalGames: number
  averageScore: number
  averageCaptures: number
  winRate: number
}

interface AIStatsState {
  matches: MatchResult[]
  addMatch: (result: MatchResult) => void
  getStats: (config: AIConfig) => AIStats
  getLeaderboard: () => AIStats[]
  clearHistory: () => void
}

function configSignature(config: AIConfig): string {
  return `L${config.level}_C${config.captureWeight}_T${config.territoryWeight}_I${config.influenceWeight}_R${config.randomness}`
}

export const useAIStatsStore = create<AIStatsState>()(
  persist(
    (set, get) => ({
      matches: [],

      addMatch: (result) => {
        set((state) => ({
          matches: [result, ...state.matches].slice(0, 100) // Keep last 100 matches
        }))
      },

      getStats: (config) => {
        const signature = configSignature(config)
        const matches = get().matches

        // Find all matches where this config was used
        const configMatches = matches.filter(
          m => configSignature(m.blackConfig) === signature || configSignature(m.whiteConfig) === signature
        )

        if (configMatches.length === 0) {
          return {
            configSignature: signature,
            wins: 0,
            losses: 0,
            draws: 0,
            totalGames: 0,
            averageScore: 0,
            averageCaptures: 0,
            winRate: 0,
          }
        }

        let wins = 0
        let losses = 0
        let draws = 0
        let totalScore = 0
        let totalCaptures = 0

        for (const match of configMatches) {
          const isBlack = configSignature(match.blackConfig) === signature
          const isWhite = configSignature(match.whiteConfig) === signature

          if (match.winner === 'draw') {
            draws++
          } else if ((isBlack && match.winner === 'black') || (isWhite && match.winner === 'white')) {
            wins++
          } else {
            losses++
          }

          if (isBlack) {
            totalScore += match.blackScore
            totalCaptures += match.blackCaptures
          } else {
            totalScore += match.whiteScore
            totalCaptures += match.whiteCaptures
          }
        }

        const totalGames = configMatches.length

        return {
          configSignature: signature,
          wins,
          losses,
          draws,
          totalGames,
          averageScore: totalScore / totalGames,
          averageCaptures: totalCaptures / totalGames,
          winRate: totalGames > 0 ? (wins / totalGames) * 100 : 0,
        }
      },

      getLeaderboard: () => {
        const matches = get().matches
        const configMap = new Map<string, AIConfig>()

        // Collect all unique configs
        for (const match of matches) {
          const blackSig = configSignature(match.blackConfig)
          const whiteSig = configSignature(match.whiteConfig)
          if (!configMap.has(blackSig)) configMap.set(blackSig, match.blackConfig)
          if (!configMap.has(whiteSig)) configMap.set(whiteSig, match.whiteConfig)
        }

        // Get stats for each config
        const stats = Array.from(configMap.values()).map(config => get().getStats(config))

        // Sort by win rate, then by total games
        return stats
          .filter(s => s.totalGames >= 3) // Only show configs with at least 3 games
          .sort((a, b) => {
            if (Math.abs(a.winRate - b.winRate) < 1) {
              return b.totalGames - a.totalGames
            }
            return b.winRate - a.winRate
          })
      },

      clearHistory: () => {
        set({ matches: [] })
      },
    }),
    {
      name: 'gogogo-ai-stats',
    }
  )
)
