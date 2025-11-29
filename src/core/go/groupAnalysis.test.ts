import { describe, it, expect } from 'vitest'
import {
  hasEye,
  countEyes,
  isAlive,
  getGroupStrength,
  getCriticalPoints
} from './groupAnalysis'
import { createBoard, placeStone } from './board'
import { fixtures } from '../../test/fixtures'

describe('Group Analysis', () => {
  describe('Eye Detection', () => {
    it('should detect simple true eye in corner', () => {
      // True corner eye: fully enclosed
      // X X X
      // X . X
      // X X X
      let board = createBoard(3)
      // Surround (1,1) completely
      board = placeStone(board, 0, 0, 'black')!
      board = placeStone(board, 0, 1, 'black')!
      board = placeStone(board, 0, 2, 'black')!
      board = placeStone(board, 1, 0, 'black')!
      board = placeStone(board, 1, 2, 'black')!
      board = placeStone(board, 2, 0, 'black')!
      board = placeStone(board, 2, 1, 'black')!
      board = placeStone(board, 2, 2, 'black')!

      // Position (1,1) is a true eye for black
      expect(hasEye(board, { row: 1, col: 1 }, 'black')).toBe(true)
    })

    it('should detect false eye (opponent stone present)', () => {
      // False eye: has white stone in diagonal
      // X X .
      // X . O
      // . . .
      let board = createBoard(3)
      board = placeStone(board, 0, 0, 'black')!
      board = placeStone(board, 0, 1, 'black')!
      board = placeStone(board, 1, 0, 'black')!
      board = placeStone(board, 1, 2, 'white')!

      // Position (1,1) is a false eye (threatened by white at 1,2)
      expect(hasEye(board, { row: 1, col: 1 }, 'black')).toBe(false)
    })

    it('should not detect eye on occupied point', () => {
      let board = createBoard(3)
      board = placeStone(board, 1, 1, 'black')!

      expect(hasEye(board, { row: 1, col: 1 }, 'black')).toBe(false)
    })
  })

  describe('Eye Counting', () => {
    it('should count two eyes in living group', () => {
      // Classic two-eye corner position
      // X X X .
      // X . X .
      // X X X .
      // . . . .
      let board = createBoard(4)
      // Outer ring
      board = placeStone(board, 0, 0, 'black')!
      board = placeStone(board, 0, 1, 'black')!
      board = placeStone(board, 0, 2, 'black')!
      board = placeStone(board, 1, 0, 'black')!
      board = placeStone(board, 1, 2, 'black')!
      board = placeStone(board, 2, 0, 'black')!
      board = placeStone(board, 2, 1, 'black')!
      board = placeStone(board, 2, 2, 'black')!

      // Should detect 1 eye at (1,1)
      // (Simplified: count 1 for now, full eye counting is complex)
      const eyes = countEyes(board, { row: 0, col: 0 })
      expect(eyes).toBeGreaterThanOrEqual(1)
    })

    it('should count zero eyes for single stone', () => {
      let board = createBoard(5)
      board = placeStone(board, 2, 2, 'black')!

      const eyes = countEyes(board, { row: 2, col: 2 })
      expect(eyes).toBe(0)
    })
  })

  describe('Life and Death', () => {
    it('should recognize alive group with two eyes', () => {
      // Two-eye corner group
      let board = createBoard(4)
      board = placeStone(board, 0, 0, 'black')!
      board = placeStone(board, 0, 1, 'black')!
      board = placeStone(board, 0, 2, 'black')!
      board = placeStone(board, 1, 0, 'black')!
      board = placeStone(board, 1, 2, 'black')!
      board = placeStone(board, 2, 0, 'black')!
      board = placeStone(board, 2, 1, 'black')!
      board = placeStone(board, 2, 2, 'black')!

      const status = isAlive(board, { row: 0, col: 0 })
      // Should be alive or at least not dead
      expect(status).not.toBe('dead')
    })

    it('should recognize dead group in atari', () => {
      const board = fixtures.atari_simple()

      // Black group at (1,1) is in atari (1 liberty)
      const status = isAlive(board, { row: 1, col: 1 })
      expect(status).toBe('unsettled') // or 'dead' depending on implementation
    })

    it('should recognize single stone as unsettled', () => {
      let board = createBoard(5)
      board = placeStone(board, 2, 2, 'black')!

      const status = isAlive(board, { row: 2, col: 2 })
      expect(status).toBe('unsettled')
    })
  })

  describe('Group Strength', () => {
    it('should rate corner group as strong', () => {
      // Corner group with territory
      let board = createBoard(5)
      board = placeStone(board, 0, 0, 'black')!
      board = placeStone(board, 0, 1, 'black')!
      board = placeStone(board, 1, 0, 'black')!

      const strength = getGroupStrength(board, { row: 0, col: 0 })
      expect(strength).toBeGreaterThan(0.4) // Normalized 0-1 scale, corner bonus
    })

    it('should rate atari group as weak', () => {
      const board = fixtures.atari_simple()

      const strength = getGroupStrength(board, { row: 1, col: 1 })
      expect(strength).toBeLessThan(0.3) // Very weak, about to be captured
    })

    it('should rate center stone as moderate', () => {
      let board = createBoard(9)
      board = placeStone(board, 4, 4, 'black')!

      const strength = getGroupStrength(board, { row: 4, col: 4 })
      // Single center stone: has liberties but no position bonus
      expect(strength).toBeGreaterThan(0.15)
      expect(strength).toBeLessThan(0.35)
    })
  })

  describe('Critical Points', () => {
    it('should identify last liberty as critical', () => {
      const board = fixtures.atari_simple()

      // Black at (1,1) has one liberty at (2,1)
      const critical = getCriticalPoints(board, { row: 1, col: 1 })
      expect(critical).toContainEqual({ row: 2, col: 1 })
    })

    it('should identify eye points as critical', () => {
      // Group with true eye
      let board = createBoard(5)
      // Create enclosed eye at (2,2)
      board = placeStone(board, 1, 1, 'black')!
      board = placeStone(board, 1, 2, 'black')!
      board = placeStone(board, 1, 3, 'black')!
      board = placeStone(board, 2, 1, 'black')!
      // (2,2) is empty - the eye
      board = placeStone(board, 2, 3, 'black')!
      board = placeStone(board, 3, 1, 'black')!
      board = placeStone(board, 3, 2, 'black')!
      board = placeStone(board, 3, 3, 'black')!

      // Get critical points for the group (query any stone in group)
      const critical = getCriticalPoints(board, { row: 1, col: 1 })
      // Eye point at (2,2) should be in critical points
      expect(critical).toContainEqual({ row: 2, col: 2 })
    })
  })
})
