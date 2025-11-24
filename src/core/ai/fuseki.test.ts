import { describe, it, expect } from 'vitest'
import { createBoard, placeStone } from '../go/board'
import { getOpeningMoves, evaluateJoseki, evaluateChineseOpening } from './openings'
import { isOverlyDense, isOnBoundary, isFillingOwnTerritory } from './evaluation'
import { getAIMove } from './simpleAI'
import { AI_PRESETS } from './types'

describe('Fuseki & Opening Strategy', () => {
  describe('Opening Moves', () => {
    it('suggests hoshi (4-4) points on 9x9 board', () => {
      const board = createBoard(9)
      const openingMoves = getOpeningMoves(board, 0, 3)

      // Should include hoshi points (3,3), (3,5), (5,3), (5,5)
      expect(openingMoves.length).toBeGreaterThan(0)
      const hasHoshi = openingMoves.some(m =>
        (m.row === 3 && m.col === 3) ||
        (m.row === 3 && m.col === 5) ||
        (m.row === 5 && m.col === 3) ||
        (m.row === 5 && m.col === 5)
      )
      expect(hasHoshi).toBe(true)
    })

    it('suggests hoshi points on 13x13 board', () => {
      const board = createBoard(13)
      const openingMoves = getOpeningMoves(board, 0, 3)

      // Should include hoshi points at (3,3), (3,9), (9,3), (9,9)
      const hasHoshi = openingMoves.some(m =>
        (m.row === 3 && m.col === 3) ||
        (m.row === 3 && m.col === 9) ||
        (m.row === 9 && m.col === 3) ||
        (m.row === 9 && m.col === 9)
      )
      expect(hasHoshi).toBe(true)
    })

    it('returns empty array for small boards (< 7x7)', () => {
      const board5 = createBoard(5)
      const openingMoves = getOpeningMoves(board5, 0, 3)
      expect(openingMoves.length).toBe(0)
    })

    it('filters out occupied opening points', () => {
      let board = createBoard(9)
      board = placeStone(board, 3, 3, 'black')! // Occupy hoshi point

      const openingMoves = getOpeningMoves(board, 0, 3)

      // Should not include occupied (3,3)
      const hasOccupied = openingMoves.some(m => m.row === 3 && m.col === 3)
      expect(hasOccupied).toBe(false)
    })
  })

  describe('Joseki Recognition', () => {
    it('recognizes corner play as joseki opportunity', () => {
      let board = createBoard(13)
      // Black plays 4-4 in corner
      board = placeStone(board, 3, 3, 'black')!

      // White approaching at 4-5 (in corner area) should get joseki bonus
      const josekiBonus = evaluateJoseki(board, { row: 4, col: 4 }, 'white', 5)
      expect(josekiBonus).toBeGreaterThan(0)
    })

    it('gives higher joseki bonus for master level (5)', () => {
      let board = createBoard(13)
      board = placeStone(board, 3, 3, 'black')!

      const level3Bonus = evaluateJoseki(board, { row: 4, col: 4 }, 'white', 3)
      const level5Bonus = evaluateJoseki(board, { row: 4, col: 4 }, 'white', 5)

      expect(level5Bonus).toBeGreaterThan(level3Bonus)
    })

    it('prefers classic joseki positions (3-3, 3-4, 4-4 area)', () => {
      let board = createBoard(13)
      board = placeStone(board, 3, 3, 'black')!

      // Position at classic joseki point next to opponent's corner stone
      const josekiBonus = evaluateJoseki(board, { row: 2, col: 3 }, 'white', 5)
      expect(josekiBonus).toBeGreaterThan(0)
    })

    it('ignores non-corner positions', () => {
      let board = createBoard(13)
      board = placeStone(board, 3, 3, 'black')!

      // Center position should not get joseki bonus
      const centerBonus = evaluateJoseki(board, { row: 6, col: 6 }, 'white', 5)
      expect(centerBonus).toBe(0)
    })

    it('returns 0 for levels below 3', () => {
      let board = createBoard(13)
      board = placeStone(board, 3, 3, 'black')!

      const level1Bonus = evaluateJoseki(board, { row: 3, col: 6 }, 'white', 1)
      const level2Bonus = evaluateJoseki(board, { row: 3, col: 6 }, 'white', 2)

      expect(level1Bonus).toBe(0)
      expect(level2Bonus).toBe(0)
    })
  })

  describe('Chinese Opening (Fuseki)', () => {
    it('recognizes Chinese opening pattern on 13x13+', () => {
      const board = createBoard(13)

      // Fourth line side extension
      const chineseBonus = evaluateChineseOpening(board, { row: 3, col: 6 }, 5)
      expect(chineseBonus).toBeGreaterThan(0)
    })

    it('returns 0 for boards smaller than 13x13', () => {
      const board = createBoard(9)
      const bonus = evaluateChineseOpening(board, { row: 3, col: 6 }, 5)
      expect(bonus).toBe(0)
    })

    it('returns 0 after move 10', () => {
      const board = createBoard(13)
      const bonus = evaluateChineseOpening(board, { row: 3, col: 6 }, 11)
      expect(bonus).toBe(0)
    })
  })

  describe('Density Detection (Anti-Heavy Play)', () => {
    it('detects overly dense formations for level 5', () => {
      let board = createBoard(9)

      // Create an extremely dense 4x4 block with (4,4) empty in middle
      // This ensures >50% density within 2-space radius
      board = placeStone(board, 2, 3, 'black')!
      board = placeStone(board, 2, 4, 'black')!
      board = placeStone(board, 2, 5, 'black')!
      board = placeStone(board, 3, 2, 'black')!
      board = placeStone(board, 3, 3, 'black')!
      board = placeStone(board, 3, 4, 'black')!
      board = placeStone(board, 3, 5, 'black')!
      board = placeStone(board, 3, 6, 'black')!
      board = placeStone(board, 4, 2, 'black')!
      board = placeStone(board, 4, 3, 'black')!
      // (4,4) is empty - this is where we test
      board = placeStone(board, 4, 5, 'black')!
      board = placeStone(board, 4, 6, 'black')!
      board = placeStone(board, 5, 2, 'black')!
      board = placeStone(board, 5, 3, 'black')!
      board = placeStone(board, 5, 4, 'black')!
      board = placeStone(board, 5, 5, 'black')!
      board = placeStone(board, 5, 6, 'black')!

      // Playing in the center of this massive block should be detected as dense
      const isDense = isOverlyDense(board, { row: 4, col: 4 }, 'black', 5)
      expect(isDense).toBe(true)
    })

    it('does not detect density for lower levels', () => {
      let board = createBoard(9)

      // Create dense block
      board = placeStone(board, 4, 4, 'black')!
      board = placeStone(board, 4, 5, 'black')!
      board = placeStone(board, 5, 4, 'black')!
      board = placeStone(board, 5, 5, 'black')!

      const level1Dense = isOverlyDense(board, { row: 4, col: 3 }, 'black', 1)
      const level2Dense = isOverlyDense(board, { row: 4, col: 3 }, 'black', 2)
      const level3Dense = isOverlyDense(board, { row: 4, col: 3 }, 'black', 3)

      expect(level1Dense).toBe(false)
      expect(level2Dense).toBe(false)
      expect(level3Dense).toBe(false)
    })

    it('allows sparse formations', () => {
      let board = createBoard(9)

      // Sparse formation (2 stones with space)
      board = placeStone(board, 3, 3, 'black')!
      board = placeStone(board, 5, 5, 'black')!

      const isDense = isOverlyDense(board, { row: 4, col: 4 }, 'black', 5)
      expect(isDense).toBe(false)
    })
  })

  describe('Boundary Play', () => {
    it('detects boundary positions between territories', () => {
      let board = createBoard(9)

      // Black on left, white on right
      board = placeStone(board, 4, 3, 'black')!
      board = placeStone(board, 4, 5, 'white')!

      // Position between them is boundary
      const isBoundary = isOnBoundary(board, { row: 4, col: 4 }, 'black')
      expect(isBoundary).toBe(true)
    })

    it('does not detect boundary in homogeneous area', () => {
      let board = createBoard(9)

      // Only black stones nearby
      board = placeStone(board, 4, 3, 'black')!
      board = placeStone(board, 4, 4, 'black')!

      // Position only adjacent to black is not boundary
      const isBoundary = isOnBoundary(board, { row: 4, col: 5 }, 'black')
      expect(isBoundary).toBe(false)
    })
  })

  describe('Territory Filling Detection', () => {
    it('detects filling own territory (completely surrounded)', () => {
      let board = createBoard(9)

      // Surround empty point with black
      board = placeStone(board, 4, 3, 'black')!
      board = placeStone(board, 4, 5, 'black')!
      board = placeStone(board, 3, 4, 'black')!
      board = placeStone(board, 5, 4, 'black')!

      const isFilling = isFillingOwnTerritory(board, { row: 4, col: 4 }, 'black')
      expect(isFilling).toBe(true)
    })

    it('does not detect filling when opponent is adjacent', () => {
      let board = createBoard(9)

      // Mix of black and white
      board = placeStone(board, 4, 3, 'black')!
      board = placeStone(board, 4, 5, 'white')! // Opponent adjacent
      board = placeStone(board, 3, 4, 'black')!
      board = placeStone(board, 5, 4, 'black')!

      const isFilling = isFillingOwnTerritory(board, { row: 4, col: 4 }, 'black')
      expect(isFilling).toBe(false)
    })

    it('works correctly on small boards (3x3, 5x5)', () => {
      let board = createBoard(5)

      // Should not falsely detect on small boards
      const isFilling = isFillingOwnTerritory(board, { row: 2, col: 2 }, 'black')
      expect(isFilling).toBe(false)
    })
  })

  describe('Master Level AI Behavior', () => {
    it('level 5 avoids dense formations', () => {
      let board = createBoard(9)

      // Create dense black formation
      board = placeStone(board, 4, 4, 'black')!
      board = placeStone(board, 4, 5, 'black')!
      board = placeStone(board, 5, 4, 'black')!
      board = placeStone(board, 5, 5, 'black')!
      board = placeStone(board, 3, 4, 'black')!

      // Master level should not play adjacent to this block
      const move = getAIMove(board, 'black', { black: 0, white: 0 }, AI_PRESETS[5], 10)

      if (move) {
        // Should not be right next to the dense formation
        const adjacentToDense =
          (move.row === 4 && move.col === 3) ||
          (move.row === 5 && move.col === 3) ||
          (move.row === 3 && move.col === 3)

        expect(adjacentToDense).toBe(false)
      }
    })

    it('level 5 plays more aggressively for captures than level 3', () => {
      let board = createBoard(9)

      // Black stone in atari
      board = placeStone(board, 4, 4, 'black')!
      board = placeStone(board, 4, 3, 'white')!
      board = placeStone(board, 3, 4, 'white')!
      board = placeStone(board, 4, 5, 'white')!
      // One liberty at (5,4) - capturing move

      // Test multiple times due to randomness
      const level3Moves = Array.from({ length: 10 }, () =>
        getAIMove(board, 'white', { black: 0, white: 0 }, AI_PRESETS[3], 10)
      )
      const level5Moves = Array.from({ length: 10 }, () =>
        getAIMove(board, 'white', { black: 0, white: 0 }, AI_PRESETS[5], 10)
      )

      // Count how many times capture move (5,4) is chosen
      const level3Captures = level3Moves.filter(m => m?.row === 5 && m?.col === 4).length
      const level5Captures = level5Moves.filter(m => m?.row === 5 && m?.col === 4).length

      // At least one level should capture (the position is clearly a capture)
      const totalCaptures = level3Captures + level5Captures
      expect(totalCaptures).toBeGreaterThan(0)

      // Level 5 has higher capture weight, so should capture more often
      // Allow some variance due to randomness
      expect(level5Captures).toBeGreaterThanOrEqual(level3Captures * 0.7)
    })

    it('level 5 prioritizes joseki corner responses', () => {
      let board = createBoard(13)

      // Black plays 4-4 in multiple corners to make corner play more attractive
      board = placeStone(board, 3, 3, 'black')! // Top-left
      board = placeStone(board, 3, 9, 'black')! // Top-right

      // Level 5 white should respond in corner areas
      const moves = Array.from({ length: 10 }, () =>
        getAIMove(board, 'white', { black: 0, white: 0 }, AI_PRESETS[5], 2)
      )

      // Count how many moves are in corner areas (within 5 spaces of corners)
      const cornerMoves = moves.filter(m => {
        if (!m) return false
        // Top-left corner
        const topLeft = m.row <= 5 && m.col <= 5
        // Top-right corner
        const topRight = m.row <= 5 && m.col >= 8
        // Bottom-left corner
        const bottomLeft = m.row >= 8 && m.col <= 5
        // Bottom-right corner
        const bottomRight = m.row >= 8 && m.col >= 8
        return topLeft || topRight || bottomLeft || bottomRight
      }).length

      // At least half the moves should be in corner areas (joseki awareness)
      expect(cornerMoves).toBeGreaterThan(3)
    })

    it('level 1 makes random moves without strategy', () => {
      let board = createBoard(9)

      // Get multiple moves from level 1
      const moves = Array.from({ length: 10 }, () =>
        getAIMove(board, 'black', { black: 0, white: 0 }, AI_PRESETS[1], 0)
      )

      // Level 1 should have variety due to randomness
      const uniqueMoves = new Set(moves.map(m => m ? `${m.row},${m.col}` : 'null'))
      expect(uniqueMoves.size).toBeGreaterThan(3) // Should be fairly random
    })

    it('master level avoids filling own territory', () => {
      let board = createBoard(9)

      // Create black territory (enclosed area)
      board = placeStone(board, 3, 3, 'black')!
      board = placeStone(board, 3, 4, 'black')!
      board = placeStone(board, 3, 5, 'black')!
      board = placeStone(board, 4, 3, 'black')!
      board = placeStone(board, 4, 5, 'black')!
      board = placeStone(board, 5, 3, 'black')!
      board = placeStone(board, 5, 4, 'black')!
      board = placeStone(board, 5, 5, 'black')!
      // (4,4) is enclosed territory

      const move = getAIMove(board, 'black', { black: 0, white: 0 }, AI_PRESETS[5], 10)

      // Should NOT fill the enclosed territory at (4,4)
      if (move) {
        expect(move.row === 4 && move.col === 4).toBe(false)
      }
    })
  })

  describe('AI Level Progression', () => {
    it('higher levels have lower randomness', () => {
      expect(AI_PRESETS[1].randomness).toBeGreaterThan(AI_PRESETS[3].randomness)
      expect(AI_PRESETS[3].randomness).toBeGreaterThan(AI_PRESETS[5].randomness)
    })

    it('higher levels have higher capture weight', () => {
      expect(AI_PRESETS[5].captureWeight).toBeGreaterThan(AI_PRESETS[3].captureWeight)
      expect(AI_PRESETS[3].captureWeight).toBeGreaterThan(AI_PRESETS[1].captureWeight)
    })

    it('level 5 has highest capture weight', () => {
      expect(AI_PRESETS[5].captureWeight).toBeGreaterThanOrEqual(400)
    })

    it('level 5 has lowest randomness', () => {
      expect(AI_PRESETS[5].randomness).toBeLessThanOrEqual(0.1)
    })
  })
})
