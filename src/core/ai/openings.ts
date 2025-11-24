import type { Board, Position } from '../go/types'
import { getStone } from '../go/board'

/**
 * Opening strategies and joseki patterns for AI
 */

// Standard opening points (hoshi/star points and 3-3, 3-4, 4-4)
export function getOpeningMoves(board: Board, moveCount: number, level: number): Position[] {
  const size = board.size
  const moves: Position[] = []

  // Only suggest openings for larger boards
  if (size < 7) return moves

  // Star points (hoshi) - 4-4 points
  const offset4 = 3 // 4-4 point on standard board
  const offset3 = 2 // 3-3 point
  const offset34 = 2 // 3-4 point (one coordinate)

  // For 9x9 and up
  if (size >= 9) {
    // Hoshi points (4-4) - most popular opening
    const hoshiPoints = [
      { row: offset4, col: offset4 },
      { row: offset4, col: size - 1 - offset4 },
      { row: size - 1 - offset4, col: offset4 },
      { row: size - 1 - offset4, col: size - 1 - offset4 },
    ]

    // 3-4 points (komoku) - common opening
    const komokuPoints = [
      { row: offset3, col: offset4 },
      { row: offset4, col: offset3 },
      { row: offset3, col: size - 1 - offset4 },
      { row: offset4, col: size - 1 - offset3 },
      { row: size - 1 - offset3, col: offset4 },
      { row: size - 1 - offset4, col: offset3 },
      { row: size - 1 - offset3, col: size - 1 - offset4 },
      { row: size - 1 - offset4, col: size - 1 - offset3 },
    ]

    // 3-3 points (san-san) - secure territory
    const sansanPoints = [
      { row: offset3, col: offset3 },
      { row: offset3, col: size - 1 - offset3 },
      { row: size - 1 - offset3, col: offset3 },
      { row: size - 1 - offset3, col: size - 1 - offset3 },
    ]

    // Tengen (center) - influence-based opening
    if (size % 2 === 1) {
      const center = Math.floor(size / 2)
      moves.push({ row: center, col: center })
    }

    // Level-based opening preferences
    if (level >= 3) {
      // Higher levels prefer balanced openings
      moves.push(...hoshiPoints)
      if (level >= 4) {
        moves.push(...komokuPoints)
      }
    } else if (level >= 2) {
      // Mid levels use star points
      moves.push(...hoshiPoints)
    } else {
      // Low levels play more randomly but still near corners
      moves.push(...hoshiPoints)
      moves.push(...sansanPoints)
    }
  }

  // Filter out occupied points
  return moves.filter(pos => getStone(board, pos.row, pos.col) === null)
}

/**
 * Simple joseki pattern recognition
 * Returns bonus score if move completes or extends a joseki pattern
 */
export function evaluateJoseki(board: Board, pos: Position, player: 'black' | 'white', level: number): number {
  if (level < 3) return 0 // Only level 3+ knows joseki

  let josekiBonus = 0
  const size = board.size

  // Check if we're in a corner area (scales with board size)
  // 9x9: 3 spaces from edge, 13x13: 4 spaces, 19x19: 6 spaces
  const cornerRadius = size <= 9 ? 3 : size <= 13 ? 4 : 6
  const isInCorner = (pos.row <= cornerRadius || pos.row >= size - cornerRadius - 1) &&
                     (pos.col <= cornerRadius || pos.col >= size - cornerRadius - 1)
  if (!isInCorner) return 0

  // Check if very close to corner (3-3, 3-4, 4-4 area)
  const distToTopLeft = Math.min(pos.row, pos.col)
  const distToTopRight = Math.min(pos.row, size - 1 - pos.col)
  const distToBottomLeft = Math.min(size - 1 - pos.row, pos.col)
  const distToBottomRight = Math.min(size - 1 - pos.row, size - 1 - pos.col)
  const minDistToCorner = Math.min(distToTopLeft, distToTopRight, distToBottomLeft, distToBottomRight)

  // Prefer classic joseki positions (3-3, 3-4, 4-4 area)
  if (minDistToCorner >= 2 && minDistToCorner <= 3) {
    josekiBonus += level >= 5 ? 40 : 25 // Master level knows these are key joseki points
  }

  // Simple heuristic: playing near opponent's corner stone is likely joseki
  const opponent: 'black' | 'white' = player === 'black' ? 'white' : 'black'

  // Count opponent and friendly stones in corner area
  let opponentNearby = 0
  let friendlyNearby = 0

  for (let dr = -3; dr <= 3; dr++) {
    for (let dc = -3; dc <= 3; dc++) {
      if (dr === 0 && dc === 0) continue
      const r = pos.row + dr
      const c = pos.col + dc
      if (r < 0 || r >= size || c < 0 || c >= size) continue

      const stone = getStone(board, r, c)
      if (stone === opponent) {
        const distance = Math.abs(dr) + Math.abs(dc)
        // Opponent stone nearby - this might be joseki response
        if (distance <= 2) {
          opponentNearby++
          josekiBonus += level >= 5 ? 30 : 20 // Master level more aggressive with joseki
        } else if (distance <= 3) {
          opponentNearby++
          josekiBonus += level >= 5 ? 15 : 10
        }
      } else if (stone === player) {
        friendlyNearby++
      }
    }
  }

  // If there's opponent activity in corner but we have few stones, prioritize joseki
  if (opponentNearby > 0 && friendlyNearby <= 1) {
    josekiBonus += level >= 5 ? 50 : 30 // Master level really wants to answer in corners
  }

  return josekiBonus
}

/**
 * Chinese opening fuseki pattern
 */
export function evaluateChineseOpening(board: Board, pos: Position, moveCount: number): number {
  if (moveCount > 10) return 0

  const size = board.size
  if (size < 13) return 0

  // Chinese opening emphasizes side framework with star point + extension
  // This is a simplified heuristic
  const isOnFourthLine = pos.row === 3 || pos.row === size - 4
  const isOnSide = pos.col >= 4 && pos.col <= size - 5

  if (isOnFourthLine && isOnSide) {
    return 10
  }

  return 0
}

/**
 * Detect if position is near a standard opening point
 */
export function isNearOpeningPoint(board: Board, pos: Position): boolean {
  const openingPoints = getOpeningMoves(board, 0, 5)
  for (const op of openingPoints) {
    const distance = Math.abs(pos.row - op.row) + Math.abs(pos.col - op.col)
    if (distance <= 3) return true
  }
  return false
}
