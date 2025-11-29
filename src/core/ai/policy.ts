import type { Board, Position, Stone } from '../go/types'
import { getStone, placeStone, getGroup, countLiberties } from '../go/board'
import { getGroupStrength, isAlive } from '../go/groupAnalysis'

/**
 * Heuristic Policy Network for Go
 * Computes move priors (probability distribution) for MCTS guidance
 */

export interface MovePrior {
  position: Position
  prior: number
}

/**
 * Compute move priors for all legal moves
 * Returns Map of position key -> prior probability
 */
export function computeMovePriors(board: Board, player: Stone): Map<string, number> {
  const scores = new Map<string, number>()
  const emptyPositions: Position[] = []

  // Find all empty positions
  for (let row = 0; row < board.size; row++) {
    for (let col = 0; col < board.size; col++) {
      if (getStone(board, row, col) === null) {
        emptyPositions.push({ row, col })
      }
    }
  }

  // Evaluate each position
  for (const pos of emptyPositions) {
    const score = evaluateMove(board, pos, player)
    const key = `${pos.row},${pos.col}`
    scores.set(key, Math.max(score, 0.001)) // Minimum small positive value
  }

  // Normalize to probabilities (sum = 1.0)
  const total = Array.from(scores.values()).reduce((sum, val) => sum + val, 0)
  const priors = new Map<string, number>()

  for (const [key, score] of scores) {
    priors.set(key, score / total)
  }

  return priors
}

/**
 * Evaluate a single move's quality
 * Returns raw score (higher = better)
 */
export function evaluateMove(board: Board, pos: Position, player: Stone): number {
  let score = 1.0 // Base score

  // Check if move is legal (basic check - not self-capture)
  const testBoard = placeStone(board, pos.row, pos.col, player)
  if (testBoard === null) {
    return 0.001 // Illegal move, minimal score
  }

  const group = getGroup(testBoard, pos.row, pos.col)
  const liberties = countLiberties(testBoard, group)

  // Self-capture check
  if (liberties === 0) {
    return -10.0 // Heavily penalize self-capture
  }

  // Position bonus (corners > edges > center)
  score += getPositionBonus(board, pos) * 2.0

  // Capture evaluation
  const captureBonus = evaluateCapture(board, pos, player)
  score += captureBonus * 10.0 // Very high weight for captures

  // Atari escape (save own groups in atari)
  const escapeBonus = evaluateAtariEscape(board, pos, player)
  score += escapeBonus * 8.0

  // Connection to friendly groups
  const connectionBonus = evaluateConnection(board, pos, player)
  score += connectionBonus * 3.0

  // Atari attack (put opponent in atari)
  const atariBonus = evaluateAtariAttack(board, pos, player)
  score += atariBonus * 6.0

  // Eye formation potential
  const eyeBonus = evaluateEyeFormation(board, pos, player)
  score += eyeBonus * 4.0

  return score
}

// ============================================================================
// Evaluation Helpers
// ============================================================================

function getPositionBonus(board: Board, pos: Position): number {
  const { row, col } = pos
  const size = board.size

  // Corner bonus
  const isCorner = (row === 0 || row === size - 1) && (col === 0 || col === size - 1)
  if (isCorner) return 1.5

  // Edge bonus
  const isEdge = row === 0 || row === size - 1 || col === 0 || col === size - 1
  if (isEdge) return 1.0

  // Center penalty (generally weaker)
  return 0.5
}

function evaluateCapture(board: Board, pos: Position, player: Stone): number {
  const opponent: Stone = player === 'black' ? 'white' : 'black'
  let captureValue = 0

  // Check adjacent opponent groups BEFORE placing stone
  const adjacent = [
    { row: pos.row - 1, col: pos.col },
    { row: pos.row + 1, col: pos.col },
    { row: pos.row, col: pos.col - 1 },
    { row: pos.row, col: pos.col + 1 }
  ]

  for (const adjPos of adjacent) {
    if (adjPos.row < 0 || adjPos.row >= board.size || adjPos.col < 0 || adjPos.col >= board.size) {
      continue
    }

    const adjStone = getStone(board, adjPos.row, adjPos.col)
    if (adjStone === opponent) {
      const oppGroup = getGroup(board, adjPos.row, adjPos.col)
      const oppLiberties = countLiberties(board, oppGroup)

      // Check if this is their last liberty
      // If group has 1 liberty and this move fills it, it's a capture
      if (oppLiberties === 1) {
        // Verify this move actually fills that liberty
        const groupLiberties: Position[] = []
        for (const groupPos of oppGroup) {
          const groupAdj = [
            { row: groupPos.row - 1, col: groupPos.col },
            { row: groupPos.row + 1, col: groupPos.col },
            { row: groupPos.row, col: groupPos.col - 1 },
            { row: groupPos.row, col: groupPos.col + 1 }
          ]
          for (const libPos of groupAdj) {
            if (libPos.row < 0 || libPos.row >= board.size || libPos.col < 0 || libPos.col >= board.size) {
              continue
            }
            if (getStone(board, libPos.row, libPos.col) === null) {
              groupLiberties.push(libPos)
            }
          }
        }

        // Check if our move fills the only liberty
        const fillsLiberty = groupLiberties.some(lib => lib.row === pos.row && lib.col === pos.col)
        if (fillsLiberty) {
          captureValue += oppGroup.length // Value = size of captured group
        }
      }
    }
  }

  return captureValue
}

function evaluateAtariEscape(board: Board, pos: Position, player: Stone): number {
  // Check if any of our groups adjacent to this move are in atari
  const adjacent = [
    { row: pos.row - 1, col: pos.col },
    { row: pos.row + 1, col: pos.col },
    { row: pos.row, col: pos.col - 1 },
    { row: pos.row, col: pos.col + 1 }
  ]

  let escapeValue = 0

  for (const adjPos of adjacent) {
    if (adjPos.row < 0 || adjPos.row >= board.size || adjPos.col < 0 || adjPos.col >= board.size) {
      continue
    }

    const adjStone = getStone(board, adjPos.row, adjPos.col)
    if (adjStone === player) {
      const friendlyGroup = getGroup(board, adjPos.row, adjPos.col)
      const liberties = countLiberties(board, friendlyGroup)

      // Group is in atari, this move might save it
      if (liberties === 1) {
        const testBoard = placeStone(board, pos.row, pos.col, player)
        if (testBoard) {
          const newLiberties = countLiberties(testBoard, getGroup(testBoard, adjPos.row, adjPos.col))
          if (newLiberties > 1) {
            escapeValue += friendlyGroup.length * 2 // High value for saving group
          }
        }
      }
    }
  }

  return escapeValue
}

function evaluateConnection(board: Board, pos: Position, player: Stone): number {
  // Count adjacent friendly stones
  const adjacent = [
    { row: pos.row - 1, col: pos.col },
    { row: pos.row + 1, col: pos.col },
    { row: pos.row, col: pos.col - 1 },
    { row: pos.row, col: pos.col + 1 }
  ]

  let friendlyCount = 0
  for (const adjPos of adjacent) {
    if (adjPos.row < 0 || adjPos.row >= board.size || adjPos.col < 0 || adjPos.col >= board.size) {
      continue
    }

    if (getStone(board, adjPos.row, adjPos.col) === player) {
      friendlyCount++
    }
  }

  return friendlyCount * 0.5 // Connection bonus
}

function evaluateAtariAttack(board: Board, pos: Position, player: Stone): number {
  const opponent: Stone = player === 'black' ? 'white' : 'black'
  const testBoard = placeStone(board, pos.row, pos.col, player)
  if (!testBoard) return 0

  let atariValue = 0

  // Check adjacent opponent groups
  const adjacent = [
    { row: pos.row - 1, col: pos.col },
    { row: pos.row + 1, col: pos.col },
    { row: pos.row, col: pos.col - 1 },
    { row: pos.row, col: pos.col + 1 }
  ]

  for (const adjPos of adjacent) {
    if (adjPos.row < 0 || adjPos.row >= board.size || adjPos.col < 0 || adjPos.col >= board.size) {
      continue
    }

    const adjStone = getStone(testBoard, adjPos.row, adjPos.col)
    if (adjStone === opponent) {
      const oppGroup = getGroup(testBoard, adjPos.row, adjPos.col)
      const oppLiberties = countLiberties(testBoard, oppGroup)

      // Put opponent in atari
      if (oppLiberties === 1) {
        atariValue += oppGroup.length // Value based on group size
      }
    }
  }

  return atariValue
}

function evaluateEyeFormation(board: Board, pos: Position, player: Stone): number {
  // Check if this move would create or protect an eye
  // Simplified: check if surrounded by friendly stones

  const adjacent = [
    { row: pos.row - 1, col: pos.col },
    { row: pos.row + 1, col: pos.col },
    { row: pos.row, col: pos.col - 1 },
    { row: pos.row, col: pos.col + 1 }
  ]

  let friendlyAdjacent = 0
  let totalAdjacent = 0

  for (const adjPos of adjacent) {
    if (adjPos.row < 0 || adjPos.row >= board.size || adjPos.col < 0 || adjPos.col >= board.size) {
      continue
    }

    totalAdjacent++
    if (getStone(board, adjPos.row, adjPos.col) === player) {
      friendlyAdjacent++
    }
  }

  // If mostly surrounded by friendly stones, potential eye point
  if (friendlyAdjacent >= totalAdjacent * 0.75) {
    return 1.0
  }

  return 0
}
