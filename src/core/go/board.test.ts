import { describe, it, expect } from 'vitest'
import { createBoard, placeStone, getStone, captureStones, countTerritory } from './board'

describe('Board', () => {
  describe('createBoard', () => {
    it('creates 1x1 board', () => {
      const board = createBoard(1)
      expect(board.size).toBe(1)
      expect(getStone(board, 0, 0)).toBeNull()
    })

    it('creates 3x3 board', () => {
      const board = createBoard(3)
      expect(board.size).toBe(3)
      expect(getStone(board, 1, 1)).toBeNull()
    })

    it('creates 5x5 board', () => {
      const board = createBoard(5)
      expect(board.size).toBe(5)
    })
  })

  describe('placeStone', () => {
    it('places black stone on 1x1', () => {
      const board = createBoard(1)
      const newBoard = placeStone(board, 0, 0, 'black')
      expect(getStone(newBoard, 0, 0)).toBe('black')
    })

    it('places white stone on 3x3', () => {
      const board = createBoard(3)
      const newBoard = placeStone(board, 1, 1, 'white')
      expect(getStone(newBoard, 1, 1)).toBe('white')
    })

    it('returns null when position occupied', () => {
      const board = createBoard(3)
      const b1 = placeStone(board, 1, 1, 'black')
      const b2 = placeStone(b1, 1, 1, 'white')
      expect(b2).toBeNull()
    })

    it('returns null for out of bounds', () => {
      const board = createBoard(3)
      const result = placeStone(board, 5, 5, 'black')
      expect(result).toBeNull()
    })
  })

  describe('captureStones', () => {
    it('captures surrounded stone in corner', () => {
      // Black at (0,0), White surrounds at (0,1) and (1,0)
      let board = createBoard(3)
      board = placeStone(board, 0, 0, 'black')!
      board = placeStone(board, 0, 1, 'white')!
      board = placeStone(board, 1, 0, 'white')!

      const { board: newBoard, captured } = captureStones(board, 1, 0, 'white')
      expect(captured).toBe(1)
      expect(getStone(newBoard, 0, 0)).toBeNull()
    })

    it('captures group of stones', () => {
      // Two black stones captured by white
      let board = createBoard(3)
      board = placeStone(board, 0, 0, 'black')!
      board = placeStone(board, 0, 1, 'black')!
      board = placeStone(board, 0, 2, 'white')!
      board = placeStone(board, 1, 0, 'white')!
      board = placeStone(board, 1, 1, 'white')!

      const { board: newBoard, captured } = captureStones(board, 1, 1, 'white')
      expect(captured).toBe(2)
      expect(getStone(newBoard, 0, 0)).toBeNull()
      expect(getStone(newBoard, 0, 1)).toBeNull()
    })

    it('does not capture stones with liberties', () => {
      let board = createBoard(3)
      board = placeStone(board, 1, 1, 'black')!
      board = placeStone(board, 0, 1, 'white')!

      const { captured } = captureStones(board, 0, 1, 'white')
      expect(captured).toBe(0)
    })
  })

  describe('countTerritory', () => {
    it('counts territory on empty 3x3 board', () => {
      const board = createBoard(3)
      const territory = countTerritory(board)
      expect(territory.black).toBe(0)
      expect(territory.white).toBe(0)
      expect(territory.neutral).toBe(9)
    })

    it('counts simple territory', () => {
      // Black stone in center should control some area
      let board = createBoard(3)
      board = placeStone(board, 1, 1, 'black')!

      const territory = countTerritory(board)
      expect(territory.black).toBeGreaterThan(0)
    })
  })
})
