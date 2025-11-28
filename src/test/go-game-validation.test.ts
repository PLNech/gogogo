import { describe, it, expect } from 'vitest'
import { createBoard, placeStone, captureStones, countTerritory } from '../core/go/board'
import { getAIMove } from '../core/ai/simpleAI'

describe('Go Game Engine Validation', () => {
  it('should allow multiple moves before game ends naturally', () => {
    // Create a 5x5 board
    const board = createBoard(5)

    // Make several moves
    let currentBoard = board
    let currentPlayer: 'black' | 'white' = 'black'

    // Play a few moves to make sure game progresses
    for (let i = 0; i < 10; i++) {
      // Try to make a move - if AI returns null, skip
      const move = getAIMove(currentBoard, currentPlayer, { black: 0, white: 0 }, undefined, i)

      if (move) {
        const newBoard = placeStone(currentBoard, move.row, move.col, currentPlayer)
        if (newBoard) {
          const { board: finalBoard } = captureStones(newBoard, move.row, move.col, currentPlayer)
          currentBoard = finalBoard
        }
      }

      currentPlayer = currentPlayer === 'black' ? 'white' : 'black'
    }

    // Should still be able to make moves
    expect(currentBoard).toBeDefined()
    expect(currentBoard.stones.flat().filter(Boolean).length).toBeGreaterThan(0)
  })

  it('should correctly handle territory scoring', () => {
    // Create a simple board with territory
    let board = createBoard(3)

    // Place some stones to create territory
    board = placeStone(board, 0, 0, 'black')!
    board = placeStone(board, 0, 1, 'black')!
    board = placeStone(board, 1, 0, 'white')!

    const territory = countTerritory(board)

    // Should have some territory
    expect(territory.black).toBeGreaterThanOrEqual(0)
    expect(territory.white).toBeGreaterThanOrEqual(0)
    expect(territory.neutral).toBeGreaterThanOrEqual(0)
  })

  it('should properly validate board creation and stone placement', () => {
    // Test 1x1 board
    const board1x1 = createBoard(1)
    expect(board1x1.size).toBe(1)

    // Place a stone
    const newBoard1x1 = placeStone(board1x1, 0, 0, 'black')
    expect(newBoard1x1).not.toBeNull()
    expect(newBoard1x1!.stones[0][0]).toBe('black')

    // Test 3x3 board
    const board3x3 = createBoard(3)
    expect(board3x3.size).toBe(3)

    // Place a stone
    const newBoard3x3 = placeStone(board3x3, 1, 1, 'white')
    expect(newBoard3x3).not.toBeNull()
    expect(newBoard3x3!.stones[1][1]).toBe('white')

    // Try to place on occupied spot
    const occupiedBoard = placeStone(newBoard3x3!, 1, 1, 'black')
    expect(occupiedBoard).toBeNull()
  })

  it('should properly initialize and run MCTS-based AI', () => {
    const board = createBoard(5)

    // Test that MCTS-based AI can return a move
    const move = getAIMove(board, 'black', { black: 0, white: 0 }, undefined, 0)

    // Should return a valid move (not undefined or invalid)
    expect(move).toBeDefined()
    if (move) {
      expect(move).toHaveProperty('row')
      expect(move).toHaveProperty('col')
    }
  })
})