export interface AIConfig {
  level: 1 | 2 | 3 | 4 | 5 | 6 // Difficulty level (6 = Neural Network)
  searchDepth: number // How many moves ahead to consider
  randomness: number // 0-1, how much randomness in move selection
  captureWeight: number // Multiplier for capture scoring
  territoryWeight: number // Multiplier for territory scoring
  influenceWeight: number // Multiplier for influence scoring
  useNeural?: boolean // Use neural network for move selection
}

export const AI_PRESETS: Record<number, AIConfig> = {
  1: {
    level: 1,
    searchDepth: 1,
    randomness: 0.6,
    captureWeight: 40,
    territoryWeight: 3,
    influenceWeight: 1,
  },
  2: {
    level: 2,
    searchDepth: 1,
    randomness: 0.4,
    captureWeight: 80,
    territoryWeight: 8,
    influenceWeight: 4,
  },
  3: {
    level: 3,
    searchDepth: 2,
    randomness: 0.25,
    captureWeight: 120,
    territoryWeight: 15,
    influenceWeight: 10,
  },
  4: {
    level: 4,
    searchDepth: 2,
    randomness: 0.15,
    captureWeight: 180,
    territoryWeight: 25,
    influenceWeight: 18,
  },
  5: {
    level: 5,
    searchDepth: 3,
    randomness: 0.05, // Less random
    captureWeight: 400, // MUCH more aggressive on captures
    territoryWeight: 35,
    influenceWeight: 30,
  },
  // Neural Network AI - trained on pro games
  6: {
    level: 6,
    searchDepth: 0, // Uses neural network policy directly
    randomness: 0.1, // Small temperature for variety
    captureWeight: 0, // Not used
    territoryWeight: 0, // Not used
    influenceWeight: 0, // Not used
    useNeural: true,
  },
}
