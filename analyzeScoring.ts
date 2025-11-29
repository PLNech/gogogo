import { createBoard, placeStone, getStone } from './src/core/go/board'
import type { Board, Position } from './src/core/go/types'
import { AI_PRESETS } from './src/core/ai/types'
import type { AIConfig } from './src/core/ai/types'
import {
  evaluateInfluence,
  evaluateTerritorialSecurity,
  evaluateConnection,
  isFillingOwnTerritory,
  isOverlyDense,
  isOnBoundary,
  evaluateCornerWallControl,
  evaluateInvasionReduction,
  evaluateReduction,
  evaluateGroupHealth,
  findOpponentGroups,
  evaluateAttackValue
} from './src/core/ai/evaluation'
import { evaluateShapes } from './src/core/ai/shapes'
import { evaluateBasicInstinct } from './src/core/ai/basicInstinct'
import { evaluateJoseki, evaluateChineseOpening } from './src/core/ai/openings'
import { captureStones, getGroup, countLiberties } from './src/core/go/board'

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

function wouldBeSelfCapture(board: Board, pos: Position, player: 'black' | 'white'): boolean {
  const group = getGroup(board, pos.row, pos.col)
  const liberties = countLiberties(board, group)
  return liberties === 0
}

// Full scoreMove implementation with detailed breakdown
function scoreMove(board: Board, pos: Position, player: 'black' | 'white', config: AIConfig, moveCount: number): { total: number; breakdown: any } {
  const breakdown: any = {}
  let score = 0

  // Try the move
  const newBoard = placeStone(board, pos.row, pos.col, player)
  if (!newBoard) {
    return { total: -1000, breakdown: { illegal: -1000 } }
  }

  // Check filling own territory
  if (isFillingOwnTerritory(board, pos, player)) {
    const penalty = config.level >= 3 ? -2000 : config.level >= 2 ? -800 : -200
    breakdown.fillingTerritory = penalty
    score += penalty
  }

  // Check density
  if (config.level >= 4 && isOverlyDense(board, pos, player, config.level)) {
    const densityPenalty = config.level >= 5 ? -3000 : -600
    breakdown.density = densityPenalty
    score += densityPenalty
  }

  // Boundary bonus
  if (config.level >= 2 && isOnBoundary(board, pos, player)) {
    const boundaryBonus = config.level >= 5 ? 60 : config.level >= 3 ? 35 : 20
    breakdown.boundary = boundaryBonus
    score += boundaryBonus
  }

  // Capture evaluation
  let capturedStones = 0
  if (wouldBeSelfCapture(newBoard, pos, player)) {
    const { captured } = captureStones(newBoard, pos.row, pos.col, player)
    if (captured === 0) {
      return { total: -1000, breakdown: { selfCapture: -1000 } }
    }
    capturedStones = captured
    const captureValue = captured * config.captureWeight
    breakdown.capture = captureValue
    score += captureValue
  } else {
    const { board: boardAfterCapture, captured } = captureStones(newBoard, pos.row, pos.col, player)
    capturedStones = captured
    if (captured > 0) {
      const captureValue = captured * config.captureWeight
      breakdown.capture = captureValue
      score += captureValue
    }

    const group = getGroup(boardAfterCapture, pos.row, pos.col)
    const groupHealth = evaluateGroupHealth(boardAfterCapture, group)
    if (groupHealth !== 0) {
      breakdown.groupHealth = groupHealth
      score += groupHealth
    }
  }

  // Attack opportunities
  if (config.level >= 1) {
    const opponentGroups = findOpponentGroups(board, player)
    let attackTotal = 0
    for (const groupInfo of opponentGroups) {
      const isAttackPoint = groupInfo.attackPoints.some(
        ap => ap.row === pos.row && ap.col === pos.col
      )
      if (isAttackPoint) {
        const attackValue = evaluateAttackValue(board, groupInfo, player, config.level)
        attackTotal += attackValue
      }
    }
    if (attackTotal > 0) {
      breakdown.attack = attackTotal
      score += attackTotal
    }
  }

  // Shapes
  if (config.level >= 2) {
    const shapeEval = evaluateShapes(board, pos, player, moveCount)
    if (shapeEval.isBadShape) {
      const penaltyMultiplier = config.level >= 5 ? 2.0 : config.level >= 3 ? 1.5 : 1.0
      const shapePenalty = -shapeEval.penalty * penaltyMultiplier
      breakdown.shape = shapePenalty
      score += shapePenalty
    }
    if (shapeEval.isGoodShape) {
      const bonusMultiplier = config.level >= 5 ? 2.0 : config.level >= 3 ? 1.5 : 1.0
      const shapeBonus = shapeEval.bonus * bonusMultiplier
      breakdown.shape = (breakdown.shape || 0) + shapeBonus
      score += shapeBonus
    }
  }

  // Basic instinct
  if (config.level >= 2) {
    const instinctBonus = evaluateBasicInstinct(board, pos, player, config.level)
    if (instinctBonus !== 0) {
      breakdown.instinct = instinctBonus
      score += instinctBonus
    }
  }

  // Connection
  if (config.level >= 2) {
    const connectionBonus = evaluateConnection(board, pos, player)
    const connectionWeight = config.level >= 3 ? 1.5 : 0.7
    const weightedConnection = connectionBonus * connectionWeight
    if (weightedConnection !== 0) {
      breakdown.connection = weightedConnection
      score += weightedConnection
    }
  }

  // Influence
  if (config.level >= 2) {
    const influence = evaluateInfluence(board, pos, player)
    const influenceValue = influence * config.influenceWeight
    if (influenceValue !== 0) {
      breakdown.influence = influenceValue
      score += influenceValue
    }
  }

  // Territory
  if (config.level >= 3) {
    const security = evaluateTerritorialSecurity(board, pos, player)
    const territoryValue = security * (config.territoryWeight / 10)
    if (territoryValue !== 0) {
      breakdown.territory = territoryValue
      score += territoryValue
    }
  }

  // Joseki
  const josekiMoveLimit = config.level >= 5 ? 30 : 20
  if (config.level >= 3 && moveCount < josekiMoveLimit) {
    const josekiBonus = evaluateJoseki(board, pos, player, config.level)
    if (josekiBonus !== 0) {
      breakdown.joseki = josekiBonus
      score += josekiBonus
    }
  }

  // Fuseki
  if (config.level >= 4 && moveCount < 10) {
    const fusekiBonus = evaluateChineseOpening(board, pos, moveCount)
    const weightedFuseki = config.level >= 5 ? fusekiBonus * 2 : fusekiBonus
    if (weightedFuseki !== 0) {
      breakdown.fuseki = weightedFuseki
      score += weightedFuseki
    }
  }

  // Corner/wall distance bonuses
  const distToEdge = Math.min(
    pos.row,
    pos.col,
    board.size - 1 - pos.row,
    board.size - 1 - pos.col
  )

  let distanceBonus = 0
  if (distToEdge === 0) distanceBonus = 5
  else if (distToEdge === 1) distanceBonus = 15
  else if (distToEdge === 2) distanceBonus = 25
  else if (distToEdge === 3) distanceBonus = 20
  else if (distToEdge === 4) distanceBonus = 10

  if (distanceBonus > 0) {
    breakdown.distanceToEdge = distanceBonus
    score += distanceBonus
  }

  // Opening center penalty
  if (moveCount < 10) {
    const center = (board.size - 1) / 2
    const distToCenter = Math.abs(pos.row - center) + Math.abs(pos.col - center)
    if (distToCenter < board.size / 4) {
      const centerPenalty = -15
      breakdown.centerInOpening = centerPenalty
      score += centerPenalty
    }
  }

  // Corner/wall control
  if (config.level >= 2) {
    const cornerWallValue = evaluateCornerWallControl(board, pos, player, config.level)
    if (cornerWallValue !== 0) {
      breakdown.cornerWall = cornerWallValue
      score += cornerWallValue
    }
  }

  // Invasion/reduction
  if (config.level >= 4) {
    const invasionValue = evaluateInvasionReduction(board, pos, player, config.level)
    if (invasionValue !== 0) {
      breakdown.invasion = invasionValue
      score += invasionValue
    }

    const reductionValue = evaluateReduction(board, pos, player, config.level)
    if (reductionValue !== 0) {
      breakdown.reduction = reductionValue
      score += reductionValue
    }
  }

  // Randomness
  const randomValue = Math.random() * config.randomness * 10
  breakdown.random = randomValue
  score += randomValue

  return { total: score, breakdown }
}

// Analyze first N turns
async function analyzeTurns(maxTurns: number, boardSize: number, aiLevel: number) {
  let board = createBoard(boardSize)
  let currentPlayer: 'black' | 'white' = 'black'
  let moveCount = 0
  const config = AI_PRESETS[aiLevel]

  console.log(JSON.stringify({
    type: 'analysis_start',
    boardSize,
    maxTurns,
    aiLevel,
    aiConfig: config
  }))

  for (let turn = 0; turn < maxTurns; turn++) {
    const emptyPositions = getEmptyPositions(board)

    // Score all moves
    const allScores = emptyPositions.map(pos => {
      const { total, breakdown } = scoreMove(board, pos, currentPlayer, config, moveCount)
      return {
        pos,
        total,
        breakdown
      }
    }).sort((a, b) => b.total - a.total)

    // Show board state
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
      type: 'turn_analysis',
      turn,
      player: currentPlayer,
      board: boardState,
      topMoves: allScores.slice(0, 5),
      allMoveCount: allScores.length
    }))

    if (allScores.length === 0 || allScores[0]!.total <= -1000) {
      console.log(JSON.stringify({ type: 'game_end', reason: 'no_legal_moves' }))
      break
    }

    // Make the move
    const bestMove = allScores[0]!
    const newBoard = placeStone(board, bestMove.pos.row, bestMove.pos.col, currentPlayer)
    if (!newBoard) {
      console.log(JSON.stringify({ type: 'game_end', reason: 'illegal_move' }))
      break
    }

    board = newBoard
    currentPlayer = currentPlayer === 'black' ? 'white' : 'black'
    moveCount++
  }

  console.log(JSON.stringify({ type: 'analysis_complete', totalMoves: moveCount }))
}

// Parse args
const maxTurns = parseInt(process.argv[2] || '10')
const boardSize = parseInt(process.argv[3] || '9')
const aiLevel = parseInt(process.argv[4] || '1') // Default to level 1, not 0

analyzeTurns(maxTurns, boardSize, aiLevel).catch(console.error)
