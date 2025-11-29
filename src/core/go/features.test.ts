import { describe, it, expect } from 'vitest'
import {
  extractFeatures,
  getLibertyPlane,
  getCaptureHistoryPlane,
  getKoStatusPlane
} from './features'
import { createBoard, placeStone } from './board'
import { fixtures } from '../../test/fixtures'

describe('Feature Extraction', () => {
  describe('Liberty Planes', () => {
    it('should extract 1-liberty plane', () => {
      // Create position where black has 1 liberty
      const board = fixtures.atari_simple()
      const plane = getLibertyPlane(board, 1)

      // Black stone at (1,1) has 1 liberty at (2,1)
      expect(plane[1]![1]).toBe(1)
      // White stones at (2,0) and (2,1) have 3 and 4 liberties
      expect(plane[2]![0]).toBe(0) // 3 liberties, not 1
      expect(plane[2]![1]).toBe(0) // 4 liberties, not 1
    })

    it('should extract 2-liberty plane', () => {
      let board = createBoard(3)
      board = placeStone(board, 1, 1, 'black')!
      board = placeStone(board, 0, 1, 'white')!
      // Black at (1,1) now has 2 liberties: (1,0) and (1,2), (2,1)
      const plane = getLibertyPlane(board, 2)

      expect(plane[1]![1]).toBe(0) // Black has 3 liberties, not 2
    })

    it('should extract 8+ liberty plane', () => {
      let board = createBoard(9)
      board = placeStone(board, 4, 4, 'black')! // Center: 4 liberties
      const plane = getLibertyPlane(board, 8)

      // Center stone has 4 liberties, not 8+
      expect(plane[4]![4]).toBe(0)
    })

    it('should handle empty positions', () => {
      const board = createBoard(3)
      const plane = getLibertyPlane(board, 1)

      // All positions should be 0 (no stones)
      for (let row = 0; row < 3; row++) {
        for (let col = 0; col < 3; col++) {
          expect(plane[row]![col]).toBe(0)
        }
      }
    })
  })

  describe('Capture History', () => {
    it('should create empty history plane for new game', () => {
      const board = createBoard(9)
      const plane = getCaptureHistoryPlane([], 0)

      // All positions should be 0
      expect(plane.every(row => row.every(val => val === 0))).toBe(true)
    })

    it('should mark positions where captures occurred', () => {
      // This will be implemented when we add board history tracking
      const board = createBoard(3)
      const plane = getCaptureHistoryPlane([], 0, 3)

      expect(plane).toBeDefined()
      expect(plane.length).toBe(3)
      expect(plane[0]!.length).toBe(3)
    })
  })

  describe('Ko Status', () => {
    it('should create ko status plane', () => {
      const board = createBoard(9)
      const koPosition = null // No ko
      const plane = getKoStatusPlane(board, koPosition)

      // All positions should be 0 (no ko)
      expect(plane.every(row => row.every(val => val === 0))).toBe(true)
    })

    it('should mark ko position when present', () => {
      const board = createBoard(9)
      const koPosition = { row: 3, col: 3 }
      const plane = getKoStatusPlane(board, koPosition)

      // Ko position should be marked
      expect(plane[3]![3]).toBe(1)

      // Other positions should be 0
      expect(plane[0]![0]).toBe(0)
      expect(plane[4]![4]).toBe(0)
    })
  })

  describe('Full Feature Extraction', () => {
    it('should extract all features into BoardFeatures', () => {
      const board = fixtures.atari_simple()
      const history: typeof board[] = []

      const features = extractFeatures(board, history)

      expect(features).toBeDefined()
      expect(features.libertyPlanes).toBeDefined()
      expect(features.libertyPlanes.length).toBe(8) // 8 planes: 1,2,3,4,5,6,7,8+
      expect(features.captureHistory).toBeDefined()
      expect(features.koStatus).toBeDefined()
      expect(features.moveNumber).toBe(0)
      expect(features.playerToMove).toBe('black')
    })

    it('should handle complex board position', () => {
      const board = fixtures.opening_44_44()
      const history: typeof board[] = []

      const features = extractFeatures(board, history, 'white', 2)

      expect(features.moveNumber).toBe(2)
      expect(features.playerToMove).toBe('white')
      expect(features.libertyPlanes[0]).toBeDefined()
    })
  })
})
