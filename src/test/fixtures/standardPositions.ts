import { Board } from '../../core/go/types'
import { setupPosition, parseAsciiBoard } from '../utils/boardSetup'

/**
 * Standard test fixtures for Go positions
 * Each fixture includes description and use case
 */

// ============================================================================
// Empty Boards
// ============================================================================

export const empty9x9 = (): Board => parseAsciiBoard(`
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
`)

export const empty5x5 = (): Board => parseAsciiBoard(`
  . . . . .
  . . . . .
  . . . . .
  . . . . .
  . . . . .
`)

export const empty3x3 = (): Board => parseAsciiBoard(`
  . . .
  . . .
  . . .
`)

// ============================================================================
// Opening Positions (Fuseki)
// ============================================================================

/**
 * Standard 4-4 (hoshi) opening on 9x9
 * Black occupies two corners
 */
export const opening_44_44 = (): Board => parseAsciiBoard(`
  . . . . . . . . .
  . . . . . . . . .
  . . X . . . X . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
`)

/**
 * Mixed opening: 4-4 and 3-3
 */
export const opening_44_33 = (): Board => parseAsciiBoard(`
  . . . . . . . . .
  . . . . . . . . .
  . . X . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . X . .
  . . . . . . . . .
  . . . . . . . . .
`)

// ============================================================================
// Life and Death (Tsumego)
// ============================================================================

/**
 * Simple capture: Black in atari
 * White to play and capture
 */
export const atari_simple = (): Board => setupPosition([
  ['.', '.', '.'],
  ['.', 'X', '.'],
  ['O', 'O', '.']
])

/**
 * Self-capture trap
 * If white plays at (0,0), it's self-capture
 */
export const selfCaptureCorner = (): Board => setupPosition([
  ['.', 'X', '.'],
  ['X', '.', '.'],
  ['.', '.', '.']
])

/**
 * Classic snapback position
 * Looks like capture but leads to counter-capture
 */
export const snapback = (): Board => parseAsciiBoard(`
  . . . . .
  . X X X .
  . X O O X
  . X O . O
  . . O O .
`)

/**
 * Shortage of liberties
 * Both groups racing for life
 */
export const captureRace = (): Board => parseAsciiBoard(`
  . . . . .
  . X X O .
  X . X O .
  X X O O .
  . . . . .
`)

// ============================================================================
// Tactical Positions
// ============================================================================

/**
 * Ladder position - works
 * Black can ladder-capture white
 */
export const ladderWorks = (): Board => parseAsciiBoard(`
  . . . . . . . . .
  . . . . . . . . .
  . . . O . . . . .
  . . X X O . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
`)

/**
 * Ladder position - broken
 * Ladder doesn't work due to ladder breaker
 */
export const ladderBroken = (): Board => parseAsciiBoard(`
  . . . . . . . . .
  . . . . . . . . .
  . . . O . . . . .
  . . X X O . . . .
  . . . . . . . . .
  . . . . . O . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
`)

// ============================================================================
// Territory and Endgame
// ============================================================================

/**
 * Simple territory
 * Black has secure corner territory
 */
export const blackCornerTerritory = (): Board => parseAsciiBoard(`
  . . . . .
  . X X X .
  . X . . .
  . X . . .
  . . . . .
`)

/**
 * Mixed territory
 * Both players have established areas
 */
export const mixedTerritory = (): Board => parseAsciiBoard(`
  . . . . . . . . .
  . X X X . O O O .
  . X . X . O . O .
  . X X X . O O O .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
`)

// ============================================================================
// Exports for easy importing
// ============================================================================

export const fixtures = {
  // Empty boards
  empty9x9,
  empty5x5,
  empty3x3,

  // Opening
  opening_44_44,
  opening_44_33,

  // Life and death
  atari_simple,
  selfCaptureCorner,
  snapback,
  captureRace,

  // Tactical
  ladderWorks,
  ladderBroken,

  // Territory
  blackCornerTerritory,
  mixedTerritory
}

export default fixtures
