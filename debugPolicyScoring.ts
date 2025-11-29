import { createBoard, placeStone, getStone } from './src/core/go/board'
import type { Board } from './src/core/go/types'
import { evaluateMove } from './src/core/ai/policy'

// Simulate move 5 from the game
let board = createBoard(19)
board = placeStone(board, 0, 0, 'black')!
board = placeStone(board, 0, 1, 'white')!
board = placeStone(board, 1, 0, 'black')!
board = placeStone(board, 2, 0, 'white')!

const totalStones = 4

console.log('Board state:')
for (let r = 0; r < 5; r++) {
  let row = ''
  for (let c = 0; c < 5; c++) {
    const stone = getStone(board, r, c)
    row += stone === 'black' ? '● ' : stone === 'white' ? '○ ' : '· '
  }
  console.log(row)
}

console.log('\n=== Move 5: Black to play ===')
console.log(`Total stones: ${totalStones}`)
console.log(`Opening threshold (15%): ${Math.floor(19 * 19 * 0.15)}`)
console.log(`Is opening: ${totalStones < Math.floor(19 * 19 * 0.15)}`)

const candidates = [
  { pos: { row: 1, col: 1 }, label: '(1,1) - Adjacent to group' },
  { pos: { row: 0, col: 2 }, label: '(0,2) - New corner area' },
  { pos: { row: 3, col: 0 }, label: '(3,0) - Extending wall' },
  { pos: { row: 0, col: 18 }, label: '(0,18) - Opposite corner' }
]

// Import individual evaluation functions for detailed breakdown
import {
  evaluateCapture,
  evaluateAtariEscape,
  evaluateConnection,
  evaluateAtariAttack,
  evaluateEyeFormation
} from './src/core/ai/policy.js'

function getPositionBonus(board: Board, pos: { row: number, col: number }): number {
  const { row, col } = pos
  const size = board.size
  const isCorner = (row === 0 || row === size - 1) && (col === 0 || col === size - 1)
  if (isCorner) return 1.5
  const isEdge = row === 0 || row === size - 1 || col === 0 || col === size - 1
  if (isEdge) return 1.0
  return 0.5
}

for (const cand of candidates) {
  const score = evaluateMove(board, cand.pos, 'black', totalStones)
  console.log(`\n${cand.label}: TOTAL = ${score.toFixed(1)}`)

  // Break down components
  const posBonus = getPositionBonus(board, cand.pos) * 2.0
  const capBonus = evaluateCapture(board, cand.pos, 'black') * 12.0
  const escBonus = evaluateAtariEscape(board, cand.pos, 'black') * 8.0
  const connBonus = evaluateConnection(board, cand.pos, 'black')
  const atariBonus = evaluateAtariAttack(board, cand.pos, 'black') * 6.0
  const eyeBonus = evaluateEyeFormation(board, cand.pos, 'black') * 4.0

  console.log(`  Position: +${posBonus.toFixed(1)}`)
  console.log(`  Capture: +${capBonus.toFixed(1)}`)
  console.log(`  Escape: +${escBonus.toFixed(1)}`)
  console.log(`  Connection: ${connBonus.toFixed(1)}`)
  console.log(`  Atari attack: +${atariBonus.toFixed(1)}`)
  console.log(`  Eye: +${eyeBonus.toFixed(1)}`)
}
