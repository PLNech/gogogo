import type { Board, Position, Stone } from './types'
import { getStone, getGroup, countLiberties } from './board'

/**
 * SOTA Board Feature Extraction (AlphaGo/KataGo style)
 * Multi-plane binary representation for neural networks
 */

export interface BoardFeatures {
  libertyPlanes: number[][][]  // [plane_idx][row][col] - 8 planes for 1,2,3,4,5,6,7,8+ liberties
  captureHistory: number[][]    // Last capture positions
  koStatus: number[][]          // Ko forbidden positions
  moveNumber: number            // Current move count
  playerToMove: 'black' | 'white'
}

/**
 * Extract all features from board state
 */
export function extractFeatures(
  board: Board,
  history: Board[],
  playerToMove: 'black' | 'white' = 'black',
  moveNumber: number = 0
): BoardFeatures {
  const libertyPlanes: number[][][] = []

  // Extract 8 liberty planes (1, 2, 3, 4, 5, 6, 7, 8+)
  for (let i = 1; i <= 8; i++) {
    libertyPlanes.push(getLibertyPlane(board, i))
  }

  return {
    libertyPlanes,
    captureHistory: getCaptureHistoryPlane(history, 0, board.size),
    koStatus: getKoStatusPlane(board, null),
    moveNumber,
    playerToMove
  }
}

/**
 * Get binary plane for stones with specific liberty count
 * Returns 1 where a stone has exactly N liberties (or 8+ for plane 8)
 */
export function getLibertyPlane(board: Board, libertyCount: number): number[][] {
  const plane: number[][] = Array(board.size).fill(0).map(() => Array(board.size).fill(0))

  for (let row = 0; row < board.size; row++) {
    for (let col = 0; col < board.size; col++) {
      const stone = getStone(board, row, col)
      if (stone === null) continue

      const group = getGroup(board, row, col)
      const liberties = countLiberties(board, group)

      // Plane 8 represents 8+ liberties
      if (libertyCount === 8) {
        plane[row]![col] = liberties >= 8 ? 1 : 0
      } else {
        plane[row]![col] = liberties === libertyCount ? 1 : 0
      }
    }
  }

  return plane
}

/**
 * Get capture history plane
 * Marks positions where stones were captured in recent moves
 */
export function getCaptureHistoryPlane(history: Board[], depth: number, boardSize?: number): number[][] {
  // For now, return empty plane
  // TODO: Track capture history when we implement game state tracking
  const size = history[depth]?.size ?? boardSize ?? 9
  return Array(size).fill(0).map(() => Array(size).fill(0))
}

/**
 * Get ko status plane
 * Marks position(s) where ko rule forbids play
 */
export function getKoStatusPlane(board: Board, koPosition: Position | null): number[][] {
  const plane: number[][] = Array(board.size).fill(0).map(() => Array(board.size).fill(0))

  if (koPosition) {
    plane[koPosition.row]![koPosition.col] = 1
  }

  return plane
}
