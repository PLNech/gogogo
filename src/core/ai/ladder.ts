/**
 * Ladder (Shicho/Stairs) detection and evaluation
 * A ladder is a zig-zag capturing sequence in Go
 */

import type { Board, Position } from '../go/types'
import { getStone, placeStone, getGroup, countLiberties } from '../go/board'

export interface LadderResult {
  isLadder: boolean
  works: boolean // Does the ladder succeed in capture?
  depth: number // How many moves to capture
  escapePath?: Position[] // Path the escaping stone takes
}

/**
 * Check if a position would start or continue a ladder
 * Ladders occur when a stone is in atari and tries to escape in a zig-zag pattern
 */
export function detectLadder(
  board: Board,
  pos: Position,
  player: 'black' | 'white',
  maxDepth: number = 8
): LadderResult {
  const opponent: 'black' | 'white' = player === 'black' ? 'white' : 'black'

  // Try the move
  const newBoard = placeStone(board, pos.row, pos.col, player)
  if (!newBoard) {
    return { isLadder: false, works: false, depth: 0 }
  }

  // Check if this puts opponent in atari
  const adjacents = [
    { row: pos.row - 1, col: pos.col },
    { row: pos.row + 1, col: pos.col },
    { row: pos.row, col: pos.col - 1 },
    { row: pos.row, col: pos.col + 1 },
  ]

  for (const adj of adjacents) {
    if (adj.row < 0 || adj.row >= board.size || adj.col < 0 || adj.col >= board.size) continue

    const stone = getStone(newBoard, adj.row, adj.col)
    if (stone === opponent) {
      const group = getGroup(newBoard, adj.row, adj.col)
      const liberties = countLiberties(newBoard, group)

      // In atari - check if it's a ladder
      if (liberties === 1) {
        const result = pursueLadder(newBoard, group, player, opponent, 0, maxDepth)
        if (result.isLadder) {
          return result
        }
      }
    }
  }

  return { isLadder: false, works: false, depth: 0 }
}

/**
 * Recursively check if a ladder works by reading ahead
 */
function pursueLadder(
  board: Board,
  targetGroup: Position[],
  attacker: 'black' | 'white',
  defender: 'black' | 'white',
  depth: number,
  maxDepth: number
): LadderResult {
  if (depth > maxDepth) {
    return { isLadder: false, works: false, depth }
  }

  const liberties = countLiberties(board, targetGroup)

  // Already captured
  if (liberties === 0) {
    return { isLadder: true, works: true, depth }
  }

  // Has multiple liberties - not a simple ladder
  if (liberties > 2) {
    return { isLadder: false, works: false, depth }
  }

  // Find the liberty (escape move for defender)
  const libertiesSet = new Set<string>()
  for (const pos of targetGroup) {
    const adjacents = [
      { row: pos.row - 1, col: pos.col },
      { row: pos.row + 1, col: pos.col },
      { row: pos.row, col: pos.col - 1 },
      { row: pos.row, col: pos.col + 1 },
    ]

    for (const adj of adjacents) {
      if (adj.row < 0 || adj.row >= board.size || adj.col < 0 || adj.col >= board.size) continue
      if (getStone(board, adj.row, adj.col) === null) {
        libertiesSet.add(`${adj.row},${adj.col}`)
      }
    }
  }

  // Defender escapes to liberty
  const libertyPositions = Array.from(libertiesSet).map(key => {
    const [row, col] = key.split(',').map(Number)
    return { row: row!, col: col! }
  })

  if (libertyPositions.length === 0) {
    return { isLadder: true, works: true, depth }
  }

  // Try defender's escape moves
  for (const escapePos of libertyPositions) {
    const escapeBoard = placeStone(board, escapePos.row, escapePos.col, defender)
    if (!escapeBoard) continue

    const newGroup = getGroup(escapeBoard, escapePos.row, escapePos.col)
    const newLiberties = countLiberties(escapeBoard, newGroup)

    // Check if attacker can put back in atari
    const atariMoves = findAtariMoves(escapeBoard, newGroup, attacker)

    if (atariMoves.length === 0) {
      // Defender escaped the ladder
      return { isLadder: true, works: false, depth }
    }

    // Try attacker's responses
    let anyWorks = false
    for (const atariMove of atariMoves) {
      const atariBoard = placeStone(escapeBoard, atariMove.row, atariMove.col, attacker)
      if (!atariBoard) continue

      const groupAfterAtari = getGroup(atariBoard, escapePos.row, escapePos.col)
      const result = pursueLadder(atariBoard, groupAfterAtari, attacker, defender, depth + 1, maxDepth)

      if (result.works) {
        anyWorks = true
        break
      }
    }

    if (!anyWorks) {
      // At least one escape works for defender
      return { isLadder: true, works: false, depth }
    }
  }

  // All escapes fail - ladder works
  return { isLadder: true, works: true, depth }
}

/**
 * Find moves that put a group in atari
 */
function findAtariMoves(board: Board, group: Position[], player: 'black' | 'white'): Position[] {
  const atariMoves: Position[] = []
  const checked = new Set<string>()

  for (const pos of group) {
    const adjacents = [
      { row: pos.row - 1, col: pos.col },
      { row: pos.row + 1, col: pos.col },
      { row: pos.row, col: pos.col - 1 },
      { row: pos.row, col: pos.col + 1 },
    ]

    for (const adj of adjacents) {
      const key = `${adj.row},${adj.col}`
      if (checked.has(key)) continue
      checked.add(key)

      if (adj.row < 0 || adj.row >= board.size || adj.col < 0 || adj.col >= board.size) continue
      if (getStone(board, adj.row, adj.col) !== null) continue

      // Try the move and check if it puts group in atari
      const testBoard = placeStone(board, adj.row, adj.col, player)
      if (!testBoard) continue

      const testGroup = getGroup(testBoard, group[0]!.row, group[0]!.col)
      if (testGroup.length > 0) {
        const liberties = countLiberties(testBoard, testGroup)
        if (liberties === 1) {
          atariMoves.push(adj)
        }
      }
    }
  }

  return atariMoves
}

/**
 * Check if a position is a ladder breaker
 * A ladder breaker is a stone that helps the escaping side by blocking the ladder path
 */
export function isLadderBreaker(board: Board, pos: Position, player: 'black' | 'white'): boolean {
  // Simplified check: stone on 3rd or 4th line that could interfere with ladder paths
  const distToEdge = Math.min(
    pos.row,
    pos.col,
    board.size - 1 - pos.row,
    board.size - 1 - pos.col
  )

  // Ladder breakers are typically 3-5 spaces from edge
  if (distToEdge < 2 || distToEdge > 5) return false

  // Check if there are nearby opponent stones that might be in a ladder
  const opponent: 'black' | 'white' = player === 'black' ? 'white' : 'black'
  let hasNearbyOpponent = false

  for (let dr = -4; dr <= 4; dr++) {
    for (let dc = -4; dc <= 4; dc++) {
      if (dr === 0 && dc === 0) continue
      const r = pos.row + dr
      const c = pos.col + dc
      if (r < 0 || r >= board.size || c < 0 || c >= board.size) continue

      if (getStone(board, r, c) === opponent) {
        hasNearbyOpponent = true
        break
      }
    }
    if (hasNearbyOpponent) break
  }

  return hasNearbyOpponent
}
