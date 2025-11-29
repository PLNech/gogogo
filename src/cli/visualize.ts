#!/usr/bin/env node

/**
 * CLI Visualization Tool for GoGoGo AI
 *
 * Run with: npm run visualize
 *
 * Shows:
 * - Board state in ASCII
 * - Move evaluations and priors
 * - Group analysis
 * - AI decision-making process
 */

import { createBoard, placeStone, getStone, countLiberties, getGroup } from '../core/go/board'
import { computeMovePriors, evaluateMove } from '../core/ai/policy'
import { extractFeatures } from '../core/go/features'
import { getGroupStrength, isAlive, getCriticalPoints } from '../core/go/groupAnalysis'
import type { Board, Position, Stone } from '../core/go/types'

// ============================================================================
// Board Visualization
// ============================================================================

export function visualizeBoard(board: Board): string {
  const lines: string[] = []

  // Header
  lines.push('\n  ' + Array.from({ length: board.size }, (_, i) => i).join(' '))
  lines.push('  ' + '─'.repeat(board.size * 2 - 1))

  for (let row = 0; row < board.size; row++) {
    let line = `${row}│`
    for (let col = 0; col < board.size; col++) {
      const stone = getStone(board, row, col)
      if (stone === 'black') {
        line += '●'
      } else if (stone === 'white') {
        line += '○'
      } else {
        // Show intersection markers
        const isCorner = (row === 0 || row === board.size - 1) && (col === 0 || col === board.size - 1)
        const isEdge = row === 0 || row === board.size - 1 || col === 0 || col === board.size - 1
        line += isCorner ? '+' : (isEdge ? '·' : '·')
      }
      if (col < board.size - 1) line += ' '
    }
    lines.push(line)
  }

  return lines.join('\n')
}

// ============================================================================
// Policy Heatmap
// ============================================================================

export function visualizePolicyPriors(board: Board, player: Stone): string {
  const priors = computeMovePriors(board, player)
  const lines: string[] = []

  lines.push(`\n${player === 'black' ? '●' : '○'} Move Priors (probability):`)
  lines.push('  ' + Array.from({ length: board.size }, (_, i) => i).join('    '))

  for (let row = 0; row < board.size; row++) {
    let line = `${row}│`
    for (let col = 0; col < board.size; col++) {
      const key = `${row},${col}`
      const prior = priors.get(key) || 0
      const percent = (prior * 100).toFixed(1)
      line += percent.padStart(4, ' ')
      if (col < board.size - 1) line += ' '
    }
    lines.push(line)
  }

  return lines.join('\n')
}

// ============================================================================
// Top Moves
// ============================================================================

export function visualizeTopMoves(board: Board, player: Stone, count: number = 5): string {
  const priors = computeMovePriors(board, player)
  const moves: Array<{ pos: Position; prior: number; score: number }> = []

  for (const [key, prior] of priors) {
    const [row, col] = key.split(',').map(Number)
    const pos = { row: row!, col: col! }
    const score = evaluateMove(board, pos, player)
    moves.push({ pos, prior, score })
  }

  moves.sort((a, b) => b.prior - a.prior)

  const lines: string[] = []
  lines.push(`\nTop ${count} moves for ${player === 'black' ? '●' : '○'}:`)
  lines.push('─'.repeat(50))

  for (let i = 0; i < Math.min(count, moves.length); i++) {
    const move = moves[i]!
    const percent = (move.prior * 100).toFixed(2)
    lines.push(`${i + 1}. (${move.pos.row},${move.pos.col}) - ${percent}% (score: ${move.score.toFixed(2)})`)
  }

  return lines.join('\n')
}

// ============================================================================
// Group Analysis
// ============================================================================

export function visualizeGroupAnalysis(board: Board, pos: Position): string {
  const stone = getStone(board, pos.row, pos.col)
  if (stone === null) return '\nNo stone at this position'

  const group = getGroup(board, pos.row, pos.col)
  const liberties = countLiberties(board, group)
  const strength = getGroupStrength(board, pos)
  const lifeStatus = isAlive(board, pos)
  const critical = getCriticalPoints(board, pos)

  const lines: string[] = []
  lines.push(`\nGroup Analysis for ${stone === 'black' ? '●' : '○'} at (${pos.row},${pos.col}):`)
  lines.push('─'.repeat(50))
  lines.push(`Size: ${group.length} stones`)
  lines.push(`Liberties: ${liberties}`)
  lines.push(`Strength: ${(strength * 100).toFixed(0)}%`)
  lines.push(`Status: ${lifeStatus}`)
  lines.push(`Critical points: ${critical.map(p => `(${p.row},${p.col})`).join(', ')}`)

  return lines.join('\n')
}

// ============================================================================
// Feature Planes Visualization
// ============================================================================

export function visualizeFeaturePlanes(board: Board): string {
  const features = extractFeatures(board, [])
  const lines: string[] = []

  lines.push('\nLiberty Planes (1, 2, 3, 4+ liberties):')

  for (let planeIdx = 0; planeIdx < 4; planeIdx++) {
    lines.push(`\n${planeIdx + 1}-liberty plane:`)
    const plane = features.libertyPlanes[planeIdx]!

    for (let row = 0; row < board.size; row++) {
      let line = '  '
      for (let col = 0; col < board.size; col++) {
        line += plane[row]![col] === 1 ? '█' : '·'
        line += ' '
      }
      lines.push(line)
    }
  }

  return lines.join('\n')
}

// ============================================================================
// Interactive Demo
// ============================================================================

export function runDemo() {
  console.log('\n╔════════════════════════════════════════╗')
  console.log('║   GoGoGo AI Visualization Demo        ║')
  console.log('╚════════════════════════════════════════╝')

  // Create a simple position
  let board = createBoard(5)
  board = placeStone(board, 1, 1, 'black')!
  board = placeStone(board, 2, 2, 'black')!
  board = placeStone(board, 3, 1, 'white')!
  board = placeStone(board, 3, 2, 'white')!

  console.log(visualizeBoard(board))
  console.log(visualizeTopMoves(board, 'black', 5))
  console.log(visualizePolicyPriors(board, 'black'))
  console.log(visualizeGroupAnalysis(board, { row: 1, col: 1 }))
  console.log(visualizeFeaturePlanes(board))

  console.log('\n✓ Demo complete! All systems operational.\n')
}

// Run if called directly
if (require.main === module) {
  runDemo()
}

export { visualizeBoard, visualizePolicyPriors, visualizeTopMoves, visualizeGroupAnalysis, visualizeFeaturePlanes, runDemo }
