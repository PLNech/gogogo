import { describe, it, expect } from 'vitest'
import { computeMovePriors, evaluateMove } from './policy'
import { createBoard, placeStone } from '../go/board'
import { fixtures } from '../../test/fixtures'

describe('Policy Network (Heuristic)', () => {
  describe('Move Priors', () => {
    it('should compute priors for all legal moves', () => {
      const board = createBoard(5)
      const priors = computeMovePriors(board, 'black')

      // Should have entries for all empty positions
      expect(priors.size).toBe(25) // 5x5 = 25 positions

      // All priors should be positive and sum to ~1.0
      let sum = 0
      for (const prior of priors.values()) {
        expect(prior).toBeGreaterThan(0)
        sum += prior
      }
      expect(sum).toBeCloseTo(1.0, 2) // Allow small floating point error
    })

    it('should prioritize corner over edge over center', () => {
      const board = createBoard(5)
      const priors = computeMovePriors(board, 'black')

      const corner = priors.get('0,0')! // Corner
      const edge = priors.get('0,2')!   // Edge (top center)
      const center = priors.get('2,2')! // Center

      // Corner should have higher prior than edge
      expect(corner).toBeGreaterThan(edge)
      // Edge should have higher prior than center (on empty board)
      expect(edge).toBeGreaterThan(center)
    })

    it('should highly prioritize capture moves', () => {
      const board = fixtures.atari_simple()
      const priors = computeMovePriors(board, 'white')

      // Capturing move at (2,1) should have very high prior
      const capturePrior = priors.get('2,1')!
      const otherPrior = priors.get('2,2')! // Another empty position

      // Capture should be much higher than random empty move
      expect(capturePrior).toBeGreaterThan(otherPrior * 2) // At least 2x higher
    })

    it('should prioritize atari escape', () => {
      const board = fixtures.atari_simple()
      const priors = computeMovePriors(board, 'black')

      // Escape move at (2,1) should have high prior
      const escapePrior = priors.get('2,1')!
      const otherPrior = priors.get('0,2')!

      expect(escapePrior).toBeGreaterThan(otherPrior * 2)
    })

    it('should penalize self-capture', () => {
      const board = fixtures.selfCaptureCorner()
      const priors = computeMovePriors(board, 'white')

      // Self-capture at (0,0) should have very low prior
      const selfCapturePrior = priors.get('0,0')!
      const normalPrior = priors.get('2,2')!

      expect(selfCapturePrior).toBeLessThan(normalPrior * 0.1) // Much lower
    })
  })

  describe('Move Evaluation', () => {
    it('should evaluate move quality', () => {
      const board = createBoard(5)
      const cornerScore = evaluateMove(board, { row: 0, col: 0 }, 'black')
      const centerScore = evaluateMove(board, { row: 2, col: 2 }, 'black')

      // Corner should score higher than center on empty board
      expect(cornerScore).toBeGreaterThan(centerScore)
    })

    it('should give high score to captures', () => {
      const board = fixtures.atari_simple()
      const captureScore = evaluateMove(board, { row: 2, col: 1 }, 'white')
      const normalScore = evaluateMove(board, { row: 2, col: 2 }, 'white')

      expect(captureScore).toBeGreaterThan(normalScore * 2) // At least 2x higher
    })

    it('should give high score to atari escapes', () => {
      const board = fixtures.atari_simple()
      const escapeScore = evaluateMove(board, { row: 2, col: 1 }, 'black')
      const normalScore = evaluateMove(board, { row: 0, col: 2 }, 'black')

      expect(escapeScore).toBeGreaterThan(normalScore * 3)
    })

    it('should give negative score to self-capture', () => {
      const board = fixtures.selfCaptureCorner()
      const selfCaptureScore = evaluateMove(board, { row: 0, col: 0 }, 'white')

      expect(selfCaptureScore).toBeLessThan(0) // Negative score
    })

    it('should consider group strength in evaluation', () => {
      // Strong corner group
      let board = createBoard(5)
      board = placeStone(board, 0, 0, 'black')!
      board = placeStone(board, 0, 1, 'black')!
      board = placeStone(board, 1, 0, 'black')!

      // Move that connects to strong group
      const connectScore = evaluateMove(board, { row: 1, col: 1 }, 'black')
      // Move far from group
      const farScore = evaluateMove(board, { row: 4, col: 4 }, 'black')

      expect(connectScore).toBeGreaterThan(farScore)
    })
  })
})
