import { createBoard, placeStone, getStone } from './src/core/go/board'
import type { Board, Position } from './src/core/go/types'
import { AI_PRESETS } from './src/core/ai/types'

// Import the scoreMove function directly
import { evaluateConnection } from './src/core/ai/evaluation'

// Copied from simpleAI.ts to expose scoring details
function getEmptyPositions(board: Board): Position[] {
  const positions: Position[] = []
  for (let row = 0; row < board.size; row++) {
    for (let col = 0; col < board.size; col++) {
      if (getStone(board, row, col) === null) {
        positions.push({ row, col })
      }
    }
  }
  return positions
}

// Run a few turns and show detailed scoring
async function debugFirstTurns(maxTurns: number, boardSize: number) {
  let board = createBoard(boardSize)
  let currentPlayer: 'black' | 'white' = 'black'
  let moveCount = 0

  console.log(JSON.stringify({
    type: 'game_start',
    boardSize,
    maxTurns
  }))

  for (let turn = 0; turn < maxTurns; turn++) {
    // Show board
    const boardState: string[][] = []
    for (let row = 0; row < board.size; row++) {
      const rowState: string[] = []
      for (let col = 0; col < board.size; col++) {
        const stone = getStone(board, row, col)
        rowState.push(stone || '.')
      }
      boardState.push(rowState)
    }

    console.log(JSON.stringify({
      type: 'turn_start',
      turn,
      player: currentPlayer,
      board: boardState
    }))

    // Get ALL possible moves with scores
    const emptyPositions = getEmptyPositions(board)
    const allMoves = emptyPositions.map(pos => {
      // Check connection value for this position
      const connectionValue = evaluateConnection(board, pos, currentPlayer)
      return {
        pos,
        connectionValue
      }
    }).sort((a, b) => b.connectionValue - a.connectionValue)

    // Show top 10 moves
    console.log(JSON.stringify({
      type: 'top_moves',
      turn,
      player: currentPlayer,
      topMoves: allMoves.slice(0, 10)
    }))

    // Make move (use simpleAI)
    const { getAIDecision } = await import('./src/core/ai/simpleAI.js')
    const decision = getAIDecision(board, currentPlayer, { black: 0, white: 0 }, AI_PRESETS[2], moveCount)

    console.log(JSON.stringify({
      type: 'ai_decision',
      turn,
      player: currentPlayer,
      decision: {
        action: decision.action,
        position: decision.position,
        confidence: decision.confidence,
        score: decision.score
      }
    }))

    if (decision.action === 'pass' || !decision.position) {
      console.log(JSON.stringify({ type: 'game_end', reason: 'ai_passed' }))
      break
    }

    // Make the move
    const newBoard = placeStone(board, decision.position.row, decision.position.col, currentPlayer)
    if (!newBoard) {
      console.log(JSON.stringify({ type: 'game_end', reason: 'illegal_move' }))
      break
    }

    board = newBoard
    currentPlayer = currentPlayer === 'black' ? 'white' : 'black'
    moveCount++
  }

  console.log(JSON.stringify({ type: 'game_complete', totalMoves: moveCount }))
}

// Parse args
const maxTurns = parseInt(process.argv[2] || '10')
const boardSize = parseInt(process.argv[3] || '9')

debugFirstTurns(maxTurns, boardSize).catch(console.error)
