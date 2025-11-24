import { describe, it, expect } from 'vitest'
import { getAIMove } from './simpleAI'
import { createBoard, placeStone } from '../go/board'

describe('Simple AI', () => {
  it('finds valid move on empty board', () => {
    const board = createBoard(3)
    const move = getAIMove(board, 'white')

    expect(move).toBeDefined()
    expect(move?.row).toBeGreaterThanOrEqual(0)
    expect(move?.row).toBeLessThan(3)
    expect(move?.col).toBeGreaterThanOrEqual(0)
    expect(move?.col).toBeLessThan(3)
  })

  it('avoids occupied positions', () => {
    let board = createBoard(3)
    // Fill all but one position
    board = placeStone(board, 0, 0, 'black')!
    board = placeStone(board, 0, 1, 'white')!
    board = placeStone(board, 0, 2, 'black')!
    board = placeStone(board, 1, 0, 'white')!
    board = placeStone(board, 1, 1, 'black')!
    board = placeStone(board, 1, 2, 'white')!
    board = placeStone(board, 2, 0, 'black')!
    board = placeStone(board, 2, 1, 'white')!
    // Only (2,2) is empty

    const move = getAIMove(board, 'black')
    expect(move).toEqual({ row: 2, col: 2 })
  })

  it('returns null when board is full', () => {
    let board = createBoard(2)
    board = placeStone(board, 0, 0, 'black')!
    board = placeStone(board, 0, 1, 'white')!
    board = placeStone(board, 1, 0, 'black')!
    board = placeStone(board, 1, 1, 'white')!

    const move = getAIMove(board, 'black')
    expect(move).toBeNull()
  })

  it('prefers capturing moves', () => {
    // Better test: black has only one liberty left
    let board = createBoard(3)
    board = placeStone(board, 1, 1, 'black')!
    board = placeStone(board, 0, 1, 'white')!
    board = placeStone(board, 1, 0, 'white')!
    board = placeStone(board, 1, 2, 'white')!
    // Black at (1,1) has one liberty at (2,1)

    // Run it multiple times and verify capture is in the options
    const moves = Array.from({ length: 10 }, () => getAIMove(board, 'white'))
    const captureMove = moves.filter(m => m?.row === 2 && m?.col === 1)
    // Should pick capture move at least once (it's heavily weighted)
    expect(captureMove.length).toBeGreaterThan(0)
  })

  it('avoids self-capture moves', () => {
    // Set up position where one move would be self-capture
    let board = createBoard(3)
    board = placeStone(board, 0, 1, 'black')!
    board = placeStone(board, 1, 0, 'black')!
    // If white plays at (0,0), it would be self-capture

    const move = getAIMove(board, 'white')
    // Should not be (0,0)
    if (move) {
      expect(move.row === 0 && move.col === 0).toBe(false)
    }
  })
})
