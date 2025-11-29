import type { Board, Position, Stone } from './types'
import { getStone, getGroup, countLiberties, getAdjacentPositions } from './board'

/**
 * Advanced group analysis for Go AI
 * Eye detection, life/death status, group strength assessment
 */

export type LifeStatus = 'alive' | 'dead' | 'unsettled'

/**
 * Check if a position is a true eye for a given color
 * True eye: empty point surrounded by same color, with controlled diagonals
 */
export function hasEye(board: Board, pos: Position, color: Stone): boolean {
  const stone = getStone(board, pos.row, pos.col)

  // Eye must be empty
  if (stone !== null) return false

  // Check all adjacent positions are same color or edge
  const adjacent = getAdjacentPositions(board, pos.row, pos.col)
  for (const adjPos of adjacent) {
    const adjStone = getStone(board, adjPos.row, adjPos.col)
    if (adjStone !== color) return false
  }

  // Check diagonal control (simplified: at least 3 of 4 diagonals controlled)
  const diagonals = getDiagonalPositions(board, pos.row, pos.col)
  let controlledDiagonals = 0

  for (const diagPos of diagonals) {
    const diagStone = getStone(board, diagPos.row, diagPos.col)
    if (diagStone === color || diagStone === null) {
      controlledDiagonals++
    }
  }

  // Need at least 3/4 diagonals (or all if in corner/edge)
  const minRequired = diagonals.length === 4 ? 3 : diagonals.length
  return controlledDiagonals >= minRequired
}

/**
 * Count eyes in a group (simplified heuristic)
 * Returns number of likely eyes
 */
export function countEyes(board: Board, pos: Position): number {
  const stone = getStone(board, pos.row, pos.col)
  if (stone === null) return 0

  const group = getGroup(board, pos.row, pos.col)
  const liberties = countLiberties(board, group)

  // Single stone has no eyes
  if (group.length === 1) return 0

  // Heuristic: check adjacent empty spaces for potential eyes
  const emptyAdjacent = new Set<string>()

  for (const groupPos of group) {
    const adjacent = getAdjacentPositions(board, groupPos.row, groupPos.col)
    for (const adjPos of adjacent) {
      if (getStone(board, adjPos.row, adjPos.col) === null) {
        emptyAdjacent.add(`${adjPos.row},${adjPos.col}`)
      }
    }
  }

  // Count potential eyes
  let eyeCount = 0
  for (const emptyKey of emptyAdjacent) {
    const [row, col] = emptyKey.split(',').map(Number)
    if (hasEye(board, { row: row!, col: col! }, stone)) {
      eyeCount++
    }
  }

  return eyeCount
}

/**
 * Determine if a group is alive, dead, or unsettled
 */
export function isAlive(board: Board, pos: Position): LifeStatus {
  const stone = getStone(board, pos.row, pos.col)
  if (stone === null) return 'unsettled'

  const group = getGroup(board, pos.row, pos.col)
  const liberties = countLiberties(board, group)
  const eyes = countEyes(board, pos)

  // Two eyes = unconditionally alive
  if (eyes >= 2) return 'alive'

  // In atari (1 liberty) = unsettled (could be captured)
  if (liberties === 1) return 'unsettled'

  // Very few liberties and no eyes = likely dead/unsettled
  if (liberties <= 2 && eyes === 0) return 'unsettled'

  // Otherwise unsettled (need reading to determine)
  return 'unsettled'
}

/**
 * Rate group strength on 0-1 scale
 * Considers liberties, eyes, territory, position
 */
export function getGroupStrength(board: Board, pos: Position): number {
  const stone = getStone(board, pos.row, pos.col)
  if (stone === null) return 0

  const group = getGroup(board, pos.row, pos.col)
  const liberties = countLiberties(board, group)
  const eyes = countEyes(board, pos)

  // Base score from liberties (normalized)
  let strength = Math.min(liberties / 8, 1.0) * 0.4

  // Bonus for eyes
  strength += eyes * 0.2

  // Bonus for size
  const sizeBonus = Math.min(group.length / 10, 1.0) * 0.1
  strength += sizeBonus

  // Position bonus (corner/edge stones are stronger)
  const positionBonus = getPositionBonus(board, group)
  strength += positionBonus * 0.3

  return Math.min(strength, 1.0)
}

/**
 * Find critical points for a group
 * Returns positions that are vital for group's survival
 */
export function getCriticalPoints(board: Board, pos: Position): Position[] {
  const stone = getStone(board, pos.row, pos.col)
  if (stone === null) return []

  const group = getGroup(board, pos.row, pos.col)
  const critical: Position[] = []

  // Find all liberties
  const liberties = new Set<string>()
  for (const groupPos of group) {
    const adjacent = getAdjacentPositions(board, groupPos.row, groupPos.col)
    for (const adjPos of adjacent) {
      if (getStone(board, adjPos.row, adjPos.col) === null) {
        liberties.add(`${adjPos.row},${adjPos.col}`)
      }
    }
  }

  // All liberties are critical if group has few liberties
  if (liberties.size <= 2) {
    for (const libKey of liberties) {
      const [row, col] = libKey.split(',').map(Number)
      critical.push({ row: row!, col: col! })
    }
  }

  // Eye points are critical
  for (const libKey of liberties) {
    const [row, col] = libKey.split(',').map(Number)
    if (hasEye(board, { row: row!, col: col! }, stone)) {
      critical.push({ row: row!, col: col! })
    }
  }

  return critical
}

// ============================================================================
// Helper Functions
// ============================================================================

function getDiagonalPositions(board: Board, row: number, col: number): Position[] {
  const diagonals: Position[] = []
  const offsets = [
    [-1, -1], [-1, 1], [1, -1], [1, 1]
  ]

  for (const [dr, dc] of offsets) {
    const newRow = row + dr!
    const newCol = col + dc!
    if (newRow >= 0 && newRow < board.size && newCol >= 0 && newCol < board.size) {
      diagonals.push({ row: newRow, col: newCol })
    }
  }

  return diagonals
}

function getPositionBonus(board: Board, group: Position[]): number {
  let cornerEdgeCount = 0

  for (const pos of group) {
    const onEdge = pos.row === 0 || pos.row === board.size - 1 ||
                   pos.col === 0 || pos.col === board.size - 1
    const onCorner = (pos.row === 0 || pos.row === board.size - 1) &&
                     (pos.col === 0 || pos.col === board.size - 1)

    if (onCorner) cornerEdgeCount += 2
    else if (onEdge) cornerEdgeCount += 1
  }

  return Math.min(cornerEdgeCount / group.length, 1.0)
}
