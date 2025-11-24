/**
 * Shape evaluation in Go
 * Good shapes: Knight's move (keima), large knight (ogeima), bamboo joint
 * Bad shapes: Empty triangle (gukei), dumplings, heavy groups
 */

import type { Board, Position } from '../go/types'
import { getStone } from '../go/board'

export interface ShapeEvaluation {
  isGoodShape: boolean
  isBadShape: boolean
  shapeType: string
  penalty: number
  bonus: number
}

/**
 * Detect empty triangle (empty triangle is considered the worst shape)
 * Three stones in a bent line with an empty point in the middle
 */
export function detectEmptyTriangle(board: Board, pos: Position, player: 'black' | 'white'): boolean {
  // Check all possible empty triangle configurations
  const trianglePatterns = [
    // Horizontal base
    [{ dr: 0, dc: 1 }, { dr: 1, dc: 0 }], // Right + Down
    [{ dr: 0, dc: 1 }, { dr: -1, dc: 0 }], // Right + Up
    [{ dr: 0, dc: -1 }, { dr: 1, dc: 0 }], // Left + Down
    [{ dr: 0, dc: -1 }, { dr: -1, dc: 0 }], // Left + Up
    // Diagonal configurations
    [{ dr: 1, dc: 1 }, { dr: 0, dc: 1 }], // Diagonal + Right
    [{ dr: 1, dc: 1 }, { dr: 1, dc: 0 }], // Diagonal + Down
    [{ dr: 1, dc: -1 }, { dr: 0, dc: -1 }], // Diagonal + Left
    [{ dr: 1, dc: -1 }, { dr: 1, dc: 0 }], // Diagonal + Down
  ]

  for (const pattern of trianglePatterns) {
    const stone1 = {
      row: pos.row + pattern[0]!.dr,
      col: pos.col + pattern[0]!.dc,
    }
    const stone2 = {
      row: pos.row + pattern[1]!.dr,
      col: pos.col + pattern[1]!.dc,
    }
    const emptyPoint = {
      row: pos.row + pattern[0]!.dr + pattern[1]!.dr,
      col: pos.col + pattern[0]!.dc + pattern[1]!.dc,
    }

    // Check if both adjacent stones are friendly and the diagonal is empty
    if (
      isInBounds(board, stone1) &&
      isInBounds(board, stone2) &&
      isInBounds(board, emptyPoint) &&
      getStone(board, stone1.row, stone1.col) === player &&
      getStone(board, stone2.row, stone2.col) === player &&
      getStone(board, emptyPoint.row, emptyPoint.col) === null
    ) {
      return true
    }
  }

  return false
}

/**
 * Detect dumpling shape (団子) - three or more stones tightly grouped
 */
export function detectDumplingShape(board: Board, pos: Position, player: 'black' | 'white'): boolean {
  // Count solidly connected friendly stones (orthogonally adjacent)
  let connectedCount = 0
  const adjacents = [
    { row: pos.row - 1, col: pos.col },
    { row: pos.row + 1, col: pos.col },
    { row: pos.row, col: pos.col - 1 },
    { row: pos.row, col: pos.col + 1 },
  ]

  for (const adj of adjacents) {
    if (!isInBounds(board, adj)) continue
    if (getStone(board, adj.row, adj.col) === player) {
      connectedCount++
    }
  }

  // Dumpling: 3 or 4 solidly connected stones
  return connectedCount >= 3
}

/**
 * Detect knight's move (keima) - good light shape
 * 2 spaces in one direction, 1 space perpendicular
 */
export function detectKnightMove(board: Board, pos: Position, player: 'black' | 'white'): boolean {
  const knightMoves = [
    { row: -2, col: -1 }, { row: -2, col: 1 },
    { row: -1, col: -2 }, { row: -1, col: 2 },
    { row: 1, col: -2 }, { row: 1, col: 2 },
    { row: 2, col: -1 }, { row: 2, col: 1 },
  ]

  for (const km of knightMoves) {
    const target = { row: pos.row + km.row, col: pos.col + km.col }
    if (!isInBounds(board, target)) continue
    if (getStone(board, target.row, target.col) === player) {
      return true
    }
  }

  return false
}

/**
 * Detect large knight's move (ogeima) - light, flexible shape
 * 2 spaces in both directions (diagonal)
 */
export function detectLargeKnightMove(board: Board, pos: Position, player: 'black' | 'white'): boolean {
  const largeKnightMoves = [
    { row: -2, col: -2 },
    { row: -2, col: 2 },
    { row: 2, col: -2 },
    { row: 2, col: 2 },
  ]

  for (const lkm of largeKnightMoves) {
    const target = { row: pos.row + lkm.row, col: pos.col + lkm.col }
    if (!isInBounds(board, target)) continue
    if (getStone(board, target.row, target.col) === player) {
      return true
    }
  }

  return false
}

/**
 * Detect one-point jump (ikken tobi) - solid, connected shape
 */
export function detectOneSpaceJump(board: Board, pos: Position, player: 'black' | 'white'): boolean {
  const jumpMoves = [
    { row: -2, col: 0 },
    { row: 2, col: 0 },
    { row: 0, col: -2 },
    { row: 0, col: 2 },
  ]

  for (const jump of jumpMoves) {
    const target = { row: pos.row + jump.row, col: pos.col + jump.col }
    if (!isInBounds(board, target)) continue
    if (getStone(board, target.row, target.col) === player) {
      return true
    }
  }

  return false
}

/**
 * Evaluate all shapes for a position
 */
export function evaluateShapes(board: Board, pos: Position, player: 'black' | 'white', moveCount: number): ShapeEvaluation {
  let penalty = 0
  let bonus = 0
  const shapes: string[] = []

  // Bad shapes (愚形)
  if (detectEmptyTriangle(board, pos, player)) {
    penalty += 50 // Empty triangle is very bad!
    shapes.push('empty-triangle')
  }

  if (detectDumplingShape(board, pos, player)) {
    penalty += 35 // Dumpling is inefficient
    shapes.push('dumpling')
  }

  // Good shapes - especially important in opening
  const openingMultiplier = moveCount < 15 ? 1.5 : 1.0

  if (detectKnightMove(board, pos, player)) {
    bonus += 20 * openingMultiplier // Knight's move is excellent in opening
    shapes.push('knight-move')
  }

  if (detectLargeKnightMove(board, pos, player)) {
    bonus += 15 * openingMultiplier // Large knight is flexible and light
    shapes.push('large-knight')
  }

  if (detectOneSpaceJump(board, pos, player)) {
    bonus += 10 * openingMultiplier // One-space jump is solid
    shapes.push('one-space-jump')
  }

  return {
    isGoodShape: bonus > 0,
    isBadShape: penalty > 0,
    shapeType: shapes.join(','),
    penalty,
    bonus,
  }
}

function isInBounds(board: Board, pos: Position): boolean {
  return pos.row >= 0 && pos.row < board.size && pos.col >= 0 && pos.col < board.size
}
