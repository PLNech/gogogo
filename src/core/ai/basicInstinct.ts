/**
 * Basic Instinct patterns from Senseis Library
 * "Natural first moves that require no reading but form fundamental good play"
 *
 * Source: https://senseis.xmp.net/?BasicInstinct
 */

import type { Board, Position } from '../go/types'
import { getStone, getGroup, countLiberties } from '../go/board'

/**
 * Pattern 1: Atari → Extend
 * If a stone is in atari (1 liberty), extend it
 */
export function detectAtariExtend(board: Board, player: 'black' | 'white'): Position[] {
  const extensions: Position[] = []

  // Find all friendly groups with 1 liberty
  const checked = new Set<string>()

  for (let row = 0; row < board.size; row++) {
    for (let col = 0; col < board.size; col++) {
      const key = `${row},${col}`
      if (checked.has(key)) continue

      const stone = getStone(board, row, col)
      if (stone !== player) continue

      const group = getGroup(board, row, col)
      if (countLiberties(board, group) === 1) {
        // Mark all positions in group as checked
        group.forEach(p => checked.add(`${p.row},${p.col}`))

        // Find the liberty (extend position)
        for (const pos of group) {
          const adjacents = [
            { row: pos.row - 1, col: pos.col },
            { row: pos.row + 1, col: pos.col },
            { row: pos.row, col: pos.col - 1 },
            { row: pos.row, col: pos.col + 1 },
          ]

          for (const adj of adjacents) {
            if (adj.row < 0 || adj.row >= board.size || adj.col < 0 || adj.col >= board.size) continue
            if (getStone(board, adj.row, adj.col) === null) {
              extensions.push(adj)
            }
          }
        }
      }
    }
  }

  return extensions
}

/**
 * Pattern 2: Tsuke → Hane
 * When opponent attaches to your stone, hane (bend around)
 */
export function detectTsukeHane(board: Board, pos: Position, player: 'black' | 'white'): { pattern: string; bonus: number } {
  let bonus = 0
  const opponent: 'black' | 'white' = player === 'black' ? 'white' : 'black'

  // Check if opponent is adjacent (tsuke)
  const adjacents = [
    { row: pos.row - 1, col: pos.col, dir: 'up' },
    { row: pos.row + 1, col: pos.col, dir: 'down' },
    { row: pos.row, col: pos.col - 1, dir: 'left' },
    { row: pos.row, col: pos.col + 1, dir: 'right' },
  ]

  for (const adj of adjacents) {
    if (adj.row < 0 || adj.row >= board.size || adj.col < 0 || adj.col >= board.size) continue

    if (getStone(board, adj.row, adj.col) === opponent) {
      // Opponent is adjacent - check if current pos is a hane
      // Hane means wrapping around the opponent stone
      const isHane = checkIfHane(board, pos, { row: adj.row, col: adj.col }, player)
      if (isHane) {
        bonus += 25
        return { pattern: 'tsuke-hane', bonus }
      }
    }
  }

  return { pattern: '', bonus: 0 }
}

function checkIfHane(board: Board, pos: Position, opponentPos: Position, player: 'black' | 'white'): boolean {
  // Check if there's a friendly stone diagonal to both pos and opponentPos
  const diagonals = [
    { row: pos.row - 1, col: pos.col - 1 },
    { row: pos.row - 1, col: pos.col + 1 },
    { row: pos.row + 1, col: pos.col - 1 },
    { row: pos.row + 1, col: pos.col + 1 },
  ]

  for (const diag of diagonals) {
    if (diag.row < 0 || diag.row >= board.size || diag.col < 0 || diag.col >= board.size) continue
    if (getStone(board, diag.row, diag.col) === player) {
      // Check if this diagonal is also adjacent to opponent
      const distToOpp = Math.abs(diag.row - opponentPos.row) + Math.abs(diag.col - opponentPos.col)
      if (distToOpp === 1) {
        return true
      }
    }
  }

  return false
}

/**
 * Pattern 3: Head of Two Stones → Hane
 * Play hane at the head of opponent's two connected stones
 */
export function detectHeadOfTwo(board: Board, pos: Position, player: 'black' | 'white'): { pattern: string; bonus: number } {
  const opponent: 'black' | 'white' = player === 'black' ? 'white' : 'black'

  // Check all four directions for two connected opponent stones
  const directions = [
    [{ row: -1, col: 0 }, { row: -2, col: 0 }], // Up
    [{ row: 1, col: 0 }, { row: 2, col: 0 }],   // Down
    [{ row: 0, col: -1 }, { row: 0, col: -2 }], // Left
    [{ row: 0, col: 1 }, { row: 0, col: 2 }],   // Right
  ]

  for (const [first, second] of directions) {
    const r1 = pos.row + first.row
    const c1 = pos.col + first.col
    const r2 = pos.row + second.row
    const c2 = pos.col + second.col

    if (r1 < 0 || r1 >= board.size || c1 < 0 || c1 >= board.size) continue
    if (r2 < 0 || r2 >= board.size || c2 < 0 || c2 >= board.size) continue

    if (getStone(board, r1, c1) === opponent && getStone(board, r2, c2) === opponent) {
      // Found two connected opponent stones - this is the head
      return { pattern: 'head-of-two', bonus: 20 }
    }
  }

  return { pattern: '', bonus: 0 }
}

/**
 * Pattern 6: Peep → Connect
 * When opponent peeps (threatens separation), connect
 */
export function detectPeepConnect(board: Board, pos: Position, player: 'black' | 'white'): { pattern: string; bonus: number } {
  // Check if this move connects two friendly groups threatened by opponent peep
  const adjacents = [
    { row: pos.row - 1, col: pos.col },
    { row: pos.row + 1, col: pos.col },
    { row: pos.row, col: pos.col - 1 },
    { row: pos.row, col: pos.col + 1 },
  ]

  let friendlyCount = 0
  for (const adj of adjacents) {
    if (adj.row < 0 || adj.row >= board.size || adj.col < 0 || adj.col >= board.size) continue
    if (getStone(board, adj.row, adj.col) === player) {
      friendlyCount++
    }
  }

  // If connecting to 2+ friendly stones, likely responding to peep
  if (friendlyCount >= 2) {
    return { pattern: 'peep-connect', bonus: 18 }
  }

  return { pattern: '', bonus: 0 }
}

/**
 * Evaluate all basic instinct patterns for a position
 */
export function evaluateBasicInstinct(board: Board, pos: Position, player: 'black' | 'white', level: number): number {
  let bonus = 0

  // Level 1: No basic instinct
  if (level < 2) return 0

  // Check for atari extension (high priority for all levels)
  const atariExtensions = detectAtariExtend(board, player)
  if (atariExtensions.some(ext => ext.row === pos.row && ext.col === pos.col)) {
    // Level 2: Sometimes (50%)
    // Level 3+: Scrupulously (100%)
    bonus += level >= 3 ? 40 : 20
  }

  // Level 2+: Check tsuke-hane
  if (level >= 2) {
    const tsukeHane = detectTsukeHane(board, pos, player)
    if (tsukeHane.bonus > 0) {
      bonus += level >= 3 ? tsukeHane.bonus : tsukeHane.bonus * 0.6
    }
  }

  // Level 3+: Check head of two
  if (level >= 3) {
    const headOfTwo = detectHeadOfTwo(board, pos, player)
    if (headOfTwo.bonus > 0) {
      bonus += headOfTwo.bonus
    }
  }

  // Level 2+: Check peep-connect
  if (level >= 2) {
    const peepConnect = detectPeepConnect(board, pos, player)
    if (peepConnect.bonus > 0) {
      bonus += level >= 3 ? peepConnect.bonus : peepConnect.bonus * 0.7
    }
  }

  return bonus
}
