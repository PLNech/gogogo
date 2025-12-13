import type { Board, Position, Stone } from '../go/types'
import { getStone, placeStone, getGroup, countLiberties } from '../go/board'
import { getGroupStrength, isAlive, countEyes } from '../go/groupAnalysis'

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

  // Count total stones to determine game phase
  let totalStones = 0
  for (let row = 0; row < board.size; row++) {
    for (let col = 0; col < board.size; col++) {
      if (getStone(board, row, col) !== null) {
        totalStones++
      }
    }
  }

  // Evaluate each position
  for (const pos of emptyPositions) {
    const score = evaluateMove(board, pos, player, totalStones)
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
export function evaluateMove(board: Board, pos: Position, player: Stone, totalStones: number = 0): number {
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

  // CRITICAL: Tactical lookahead - check if this move creates a dead group
  // Use 2-ply search to see if opponent can capture this group immediately
  if (liberties <= 2) {
    const willDie = canOpponentCaptureInTwoMoves(testBoard, group, player)
    if (willDie) {
      // This move creates a group that will be captured in 1-2 moves
      return -50.0 // MASSIVE penalty for playing dead stones
    }
  }

  // Determine game phase
  const boardArea = board.size * board.size
  const openingThreshold = Math.floor(boardArea * 0.15) // First 15% of game is opening
  const isOpening = totalStones < openingThreshold

  // Position bonus (corners > edges > center)
  score += getPositionBonus(board, pos) * 2.0

  // Capture evaluation (immediate capture)
  const captureBonus = evaluateCapture(board, pos, player)
  score += captureBonus * 12.0 // Very high weight for captures

  // OFFENSIVE TACTICAL LOOKAHEAD: Can I capture opponent groups in 1-2 moves?
  // TODO: evaluateAttackOpportunity not implemented - using simpler capture evaluation
  // const attackOpportunity = evaluateAttackOpportunity(testBoard, pos, player)
  // score += attackOpportunity

  // Atari escape (save own groups in atari)
  const escapeBonus = evaluateAtariEscape(board, pos, player)
  if (escapeBonus > 0) {
    // CRITICAL: Use tactical lookahead to verify the escape actually saves the group
    const testBoard2 = placeStone(board, pos.row, pos.col, player)
    if (testBoard2) {
      const resultGroup = getGroup(testBoard2, pos.row, pos.col)

      // Use 2-ply lookahead: can this group survive opponent's response?
      const willDie = canOpponentCaptureInTwoMoves(testBoard2, resultGroup, player)

      if (!willDie) {
        score += escapeBonus * 8.0 // Genuine escape - reward it
      } else {
        // "Escape" leads to dead group - heavily penalize
        score -= 30.0 // STRONG penalty for futile moves
      }
    }
  }

  // Connection to friendly groups (opening-aware on large boards)
  const connectionBonus = evaluateConnection(board, pos, player)
  // Only apply opening clustering penalty on large boards (13x13+)
  // On small boards, normal density checks are sufficient
  if (isOpening && board.size >= 13) {
    // Count adjacent friendly stones
    const adjacent = [
      { row: pos.row - 1, col: pos.col },
      { row: pos.row + 1, col: pos.col },
      { row: pos.row, col: pos.col - 1 },
      { row: pos.row, col: pos.col + 1 }
    ]
    let adjacentCount = 0
    for (const adj of adjacent) {
      if (adj.row >= 0 && adj.row < board.size && adj.col >= 0 && adj.col < board.size) {
        if (getStone(board, adj.row, adj.col) === player) {
          adjacentCount++
        }
      }
    }
    // Large board opening: STRONGLY penalize playing adjacent unless connecting 2+ groups
    if (adjacentCount > 0 && connectionBonus <= 1.0) {
      // Not a true connection (2+ groups), just extending a single group
      score += -8.0 // STRONG penalty for clustering in opening
    } else {
      score += connectionBonus * 5.0 // Reward true connections
    }
  } else {
    // Small boards or middle/endgame: normal connection evaluation
    score += connectionBonus * 5.0
  }

  // Atari attack (put opponent in atari)
  const atariBonus = evaluateAtariAttack(board, pos, player)
  score += atariBonus * 6.0

  // Eye formation potential
  const eyeBonus = evaluateEyeFormation(board, pos, player)
  score += eyeBonus * 4.0

  // CRITICAL: Density penalty - avoid playing in areas with too many friendly stones
  const densityPenalty = evaluateDensity(board, pos, player)
  score += densityPenalty * 5.0 // High weight for density penalty

  return score
}

// ============================================================================
// Evaluation Helpers
// ============================================================================

export function getPositionBonus(board: Board, pos: Position): number {
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

export function evaluateCapture(board: Board, pos: Position, player: Stone): number {
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

export function evaluateAtariEscape(board: Board, pos: Position, player: Stone): number {
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

export function evaluateConnection(board: Board, pos: Position, player: Stone): number {
  // CRITICAL: Distinguish between connecting separate groups vs building walls
  const adjacent = [
    { row: pos.row - 1, col: pos.col },
    { row: pos.row + 1, col: pos.col },
    { row: pos.row, col: pos.col - 1 },
    { row: pos.row, col: pos.col + 1 }
  ]

  // Find unique groups nearby
  const nearbyGroups = new Map<string, number>() // groupKey -> groupSize
  let adjacentFriendlyCount = 0

  for (const adjPos of adjacent) {
    if (adjPos.row < 0 || adjPos.row >= board.size || adjPos.col < 0 || adjPos.col >= board.size) {
      continue
    }

    if (getStone(board, adjPos.row, adjPos.col) === player) {
      adjacentFriendlyCount++

      // Track which group this belongs to and its size
      const group = getGroup(board, adjPos.row, adjPos.col)
      const groupKey = group.map(p => `${p.row},${p.col}`).sort().join('|')
      nearbyGroups.set(groupKey, group.length)
    }
  }

  // Connecting 2+ separate groups: ALWAYS GOOD
  if (nearbyGroups.size >= 2) {
    return nearbyGroups.size * 1.5 // Strong bonus for true connections
  }

  // Adjacent to single group
  if (nearbyGroups.size === 1) {
    const groupSize = Array.from(nearbyGroups.values())[0]!

    // Large group (>= 8 stones) + high adjacency (3-4): BAD (heavy wall)
    if (groupSize >= 8 && adjacentFriendlyCount >= 3) {
      return -2.0 // Penalty for making walls with large groups
    }

    // Small group (< 8 stones): OK to extend
    return 0.5 // Small bonus for extending groups
  }

  return 0 // No adjacent friendly stones
}

export function evaluateAtariAttack(board: Board, pos: Position, player: Stone): number {
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

export function evaluateEyeFormation(board: Board, pos: Position, player: Stone): number {
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

function evaluateDensity(board: Board, pos: Position, player: Stone): number {
  // Count friendly stones in 3-space radius
  let friendlyCount = 0
  let totalSpaces = 0

  for (let dr = -3; dr <= 3; dr++) {
    for (let dc = -3; dc <= 3; dc++) {
      if (dr === 0 && dc === 0) continue
      const r = pos.row + dr
      const c = pos.col + dc
      if (r < 0 || r >= board.size || c < 0 || c >= board.size) continue

      totalSpaces++
      const stone = getStone(board, r, c)
      if (stone === player) {
        friendlyCount++
      }
    }
  }

  if (totalSpaces === 0) return 0

  const density = friendlyCount / totalSpaces

  // Only penalize if there are MANY stones (>= 12) AND high density
  // This avoids penalizing small corner groups (3-5 stones)
  if (friendlyCount >= 12 && density > 0.5) {
    return -(density - 0.5) * 20.0 // Strong penalty for large dense formations
  }

  // Moderate penalty for medium-large formations (8-11 stones, >40% density)
  if (friendlyCount >= 8 && density > 0.4) {
    return -(density - 0.4) * 10.0
  }

  return 0 // Small groups or low density is fine
}

/**
 * Tactical lookahead: Can opponent capture this group in 1-2 moves?
 * Uses exhaustive search on liberty points (cheap because only searching liberties)
 */
function canOpponentCaptureInTwoMoves(board: Board, group: Position[], player: Stone): boolean {
  const opponent: Stone = player === 'black' ? 'white' : 'black'
  const liberties = countLiberties(board, group)

  // Safe if >= 3 liberties
  if (liberties >= 3) return false

  // Already captured
  if (liberties === 0) return true

  // Find liberty positions
  const libertyPositions = new Set<string>()
  for (const groupPos of group) {
    const adjacent = [
      { row: groupPos.row - 1, col: groupPos.col },
      { row: groupPos.row + 1, col: groupPos.col },
      { row: groupPos.row, col: groupPos.col - 1 },
      { row: groupPos.row, col: groupPos.col + 1 }
    ]
    for (const adjPos of adjacent) {
      if (adjPos.row < 0 || adjPos.row >= board.size || adjPos.col < 0 || adjPos.col >= board.size) continue
      if (getStone(board, adjPos.row, adjPos.col) === null) {
        libertyPositions.add(`${adjPos.row},${adjPos.col}`)
      }
    }
  }

  // 1-ply: Can opponent capture immediately?
  for (const libKey of libertyPositions) {
    const [row, col] = libKey.split(',').map(Number)
    const testBoard = placeStone(board, row!, col!, opponent)
    if (!testBoard) continue

    // Check if group is now captured
    const stillExists = group.some(pos => getStone(testBoard, pos.row, pos.col) === player)
    if (!stillExists) return true // Group captured in 1 move
  }

  // 2-ply: Can opponent capture after player's defensive move?
  if (liberties === 1) {
    // Player tries to defend by playing on the single liberty
    const libKey = Array.from(libertyPositions)[0]!
    const [row, col] = libKey.split(',').map(Number)
    const defendBoard = placeStone(board, row!, col!, player)
    if (!defendBoard) return true // Can't defend = will die

    // After defense, can opponent still capture?
    const newGroup = getGroup(defendBoard, row!, col!)
    const newLiberties = countLiberties(defendBoard, newGroup)

    if (newLiberties <= 1) {
      // Still in immediate danger
      return true
    }
  }

  return false
}
