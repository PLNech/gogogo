import { Board, Position } from '../../core/go/types'
import { createBoard, placeStone } from '../../core/go/board'

/**
 * Test utility for setting up Go board positions
 *
 * Usage:
 *   const board = setupPosition([
 *     ['X', '.', '.'],
 *     ['.', 'O', '.'],
 *     ['.', '.', '.']
 *   ])
 *
 * Where:
 *   'X' or 'B' = black stone
 *   'O' or 'W' = white stone
 *   '.' or ' ' = empty
 */

type StoneChar = 'X' | 'B' | 'O' | 'W' | '.' | ' '

export function setupPosition(pattern: StoneChar[][]): Board {
  const size = pattern.length
  let board = createBoard(size)

  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const char = pattern[row]?.[col]
      if (!char || char === '.' || char === ' ') continue

      const stone = (char === 'X' || char === 'B') ? 'black' : 'white'
      const newBoard = placeStone(board, row, col, stone)
      if (newBoard) {
        board = newBoard
      }
    }
  }

  return board
}

/**
 * Parse ASCII board representation
 *
 * Usage:
 *   const board = parseAsciiBoard(`
 *     . . .
 *     . X .
 *     . . .
 *   `)
 */
export function parseAsciiBoard(ascii: string): Board {
  const lines = ascii
    .trim()
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0)

  const pattern: StoneChar[][] = lines.map(line => {
    return line.split(/\s+/).map(char => {
      if (char === 'X' || char === 'B') return 'X'
      if (char === 'O' || char === 'W') return 'O'
      return '.'
    }) as StoneChar[]
  })

  return setupPosition(pattern)
}

/**
 * Create a board with stones at specified positions
 *
 * Usage:
 *   const board = createBoardWithStones(9, [
 *     { pos: { row: 2, col: 2 }, color: 'black' },
 *     { pos: { row: 6, col: 6 }, color: 'white' }
 *   ])
 */
export function createBoardWithStones(
  size: number,
  stones: Array<{ pos: Position; color: 'black' | 'white' }>
): Board {
  let board = createBoard(size)

  for (const { pos, color } of stones) {
    const newBoard = placeStone(board, pos.row, pos.col, color)
    if (newBoard) {
      board = newBoard
    }
  }

  return board
}

/**
 * Verify board matches expected pattern
 * Useful for test assertions
 */
export function boardMatchesPattern(board: Board, pattern: StoneChar[][]): boolean {
  if (board.size !== pattern.length) return false

  for (let row = 0; row < board.size; row++) {
    for (let col = 0; col < board.size; col++) {
      const expected = pattern[row]?.[col]
      const actual = board.stones[row]?.[col]

      if (expected === '.' || expected === ' ') {
        if (actual !== null) return false
      } else if (expected === 'X' || expected === 'B') {
        if (actual !== 'black') return false
      } else if (expected === 'O' || expected === 'W') {
        if (actual !== 'white') return false
      }
    }
  }

  return true
}
