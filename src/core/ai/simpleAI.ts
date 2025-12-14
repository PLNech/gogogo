import type { Board, Position, Stone } from '../go/types'
import { getStone, placeStone, captureStones, countTerritory, getGroup, countLiberties } from '../go/board'
import type { AIConfig } from './types'
import { AI_PRESETS } from './types'
import { GoMCTS } from './mcts'
import {
  estimateScore,
  evaluateInfluence,
  evaluateTerritorialSecurity,
  evaluateGroupHealth,
  findOpponentGroups,
  evaluateAttackValue,
  evaluateConnection,
  isFillingOwnTerritory,
  isOverlyDense,
  isOnBoundary,
  evaluateCornerWallControl,
  evaluateInvasionReduction,
  evaluateReduction
} from './evaluation'
import { evaluateShapes } from './shapes'
import { evaluateBasicInstinct } from './basicInstinct'
import { evaluateJoseki, evaluateChineseOpening } from './openings'
import { detectLadder, isLadderBreaker } from './ladder'

export interface AIDecision {
  action: 'move' | 'pass'
  position?: Position
  confidence: number
  score: number
  mctsData?: {
    bestMove?: Position
    winRate?: number
    visits?: number
  }
}

export function getAIMove(board: Board, player: 'black' | 'white', captures: { black: number; white: number } = { black: 0, white: 0 }, config?: Partial<AIConfig>, moveCount: number = 0): Position | null {
  const decision = getAIDecision(board, player, captures, config, moveCount)
  return decision.action === 'move' ? decision.position ?? null : null
}

export function getAIDecision(board: Board, player: 'black' | 'white', captures: { black: number; white: number } = { black: 0, white: 0 }, config?: Partial<AIConfig>, moveCount: number = 0): AIDecision {
  const fullConfig = { ...AI_PRESETS[2]!, ...config } as AIConfig

  // Estimate current score
  const currentScore = estimateScore(board, captures, player)

  // If no empty positions, must pass
  const emptyPositions = getEmptyPositions(board)
  if (emptyPositions.length === 0) {
    return { action: 'pass', confidence: 1.0, score: currentScore }
  }

  // Temporarily use heuristic evaluation for all moves
  // (In Phase 3, we'll integrate this with MCTS properly)
  const scoredMoves = emptyPositions
    .map(pos => ({
      position: pos,
      score: scoreMove(board, pos, player, fullConfig, captures, moveCount)
    }))
    .filter(m => m.score > -1000) // Filter out illegal moves
    .sort((a, b) => b.score - a.score)

  if (scoredMoves.length === 0) {
    // No legal moves, pass
    return { action: 'pass', confidence: 1.0, score: currentScore }
  }

  const bestMove = scoredMoves[0]!

  // Return heuristic-based decision
  return {
    action: 'move',
    position: bestMove.position,
    confidence: Math.min(1.0, bestMove.score / 100), // Normalize confidence
    score: currentScore
  }
}

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

function scoreMove(board: Board, pos: Position, player: 'black' | 'white', config: AIConfig, captures: { black: number; white: number }, moveCount: number): number {
  let score = 0

  // Try the move
  const newBoard = placeStone(board, pos.row, pos.col, player)
  if (!newBoard) return -1000 // Invalid move

  // CRITICAL: Check if filling own territory (ALL levels should avoid this)
  if (isFillingOwnTerritory(board, pos, player)) {
    // Level 1: Large penalty but randomness might overcome (-200)
    // Level 2: Very large penalty, rarely happens (-800)
    // Level 3+: Essentially impossible (-2000)
    const penalty = config.level >= 3 ? -2000 : config.level >= 2 ? -800 : -200
    score += penalty
  }

  // CRITICAL: Check for overly dense formations (Level 4-5)
  // Master players spread out, don't create heavy blocks
  if (config.level >= 4 && isOverlyDense(board, pos, player, config.level)) {
    // Level 4: Large penalty (-600)
    // Level 5: Massive penalty (-3000, must override all bonuses including invasion/corner)
    const densityPenalty = config.level >= 5 ? -3000 : -600
    score += densityPenalty
  }

  // IMPORTANT: Bonus for playing on boundaries (where action happens)
  // Level 2+: Prefer boundary play
  // Level 5: VERY strong preference for boundary
  if (config.level >= 2 && isOnBoundary(board, pos, player)) {
    const boundaryBonus = config.level >= 5 ? 60 : config.level >= 3 ? 35 : 20
    score += boundaryBonus
  }

  // First check if this move would be self-capture BEFORE capturing
  let capturedStones = 0
  if (wouldBeSelfCapture(newBoard, pos, player)) {
    // Check if it would capture opponent stones (which makes it legal)
    const { captured } = captureStones(newBoard, pos.row, pos.col, player)
    if (captured === 0) {
      return -1000 // Self-capture without capturing opponent = illegal
    }
    // Otherwise it's a valid capturing move, continue scoring
    capturedStones = captured
    score += captured * config.captureWeight
  } else {
    // Not self-capture, check if it captures anyway
    const { board: boardAfterCapture, captured } = captureStones(newBoard, pos.row, pos.col, player)
    capturedStones = captured
    score += captured * config.captureWeight

    // Evaluate the group health after placing
    const group = getGroup(boardAfterCapture, pos.row, pos.col)
    score += evaluateGroupHealth(boardAfterCapture, group)
  }

  // GROUP TRACKING: Find opponent groups and evaluate attack opportunities
  // This is key to tactics vs strategy differentiation by level
  if (config.level >= 1) {
    const opponentGroups = findOpponentGroups(board, player)

    // Check if this position attacks any weak opponent group
    for (const groupInfo of opponentGroups) {
      const isAttackPoint = groupInfo.attackPoints.some(
        ap => ap.row === pos.row && ap.col === pos.col
      )

      if (isAttackPoint) {
        const attackValue = evaluateAttackValue(board, groupInfo, player, config.level)
        score += attackValue
      }
    }
  }

  // Shape evaluation (level 2+)
  // Level 2: Basic awareness of bad shapes
  // Level 3+: Strong emphasis on good shapes, avoids bad shapes
  // Level 5: Perfect shape judgment
  if (config.level >= 2) {
    const shapeEval = evaluateShapes(board, pos, player, moveCount)

    // Bad shapes penalty
    if (shapeEval.isBadShape) {
      const penaltyMultiplier = config.level >= 5 ? 2.0 : config.level >= 3 ? 1.5 : 1.0
      score -= shapeEval.penalty * penaltyMultiplier
    }

    // Good shapes bonus (especially in opening)
    if (shapeEval.isGoodShape) {
      const bonusMultiplier = config.level >= 5 ? 2.0 : config.level >= 3 ? 1.5 : 1.0
      score += shapeEval.bonus * bonusMultiplier
    }
  }

  // Basic Instinct patterns (level 2+)
  // Level 1: No basic instinct
  // Level 2: Sometimes follows patterns (60-70%)
  // Level 3+: Scrupulously follows patterns (100%)
  if (config.level >= 2) {
    const instinctBonus = evaluateBasicInstinct(board, pos, player, config.level)
    score += instinctBonus
  }

  // Connection evaluation (level 2+)
  if (config.level >= 2) {
    const connectionBonus = evaluateConnection(board, pos, player)
    // Level 2: sometimes connect
    // Level 3+: strongly prefer connection
    const connectionWeight = config.level >= 3 ? 1.5 : 0.7
    score += connectionBonus * connectionWeight
  }

  // Evaluate influence (level 2+)
  if (config.level >= 2) {
    const influence = evaluateInfluence(board, pos, player)
    score += influence * config.influenceWeight
  }

  // Evaluate territorial security (level 3+)
  if (config.level >= 3) {
    const security = evaluateTerritorialSecurity(board, pos, player)
    score += security * (config.territoryWeight / 10)
  }

  // Joseki evaluation (level 3+, early/mid game)
  // Master level applies joseki knowledge for longer
  const josekiMoveLimit = config.level >= 5 ? 30 : 20
  if (config.level >= 3 && moveCount < josekiMoveLimit) {
    const josekiBonus = evaluateJoseki(board, pos, player, config.level)
    score += josekiBonus
  }

  // Chinese opening pattern (level 4+, opening)
  if (config.level >= 4 && moveCount < 10) {
    const fusekiBonus = evaluateChineseOpening(board, pos, moveCount)
    // Master level values good fuseki more
    score += config.level >= 5 ? fusekiBonus * 2 : fusekiBonus
  }

  // Ladder evaluation (level 4+)
  // Avoid playing into bad ladders, reward ladder breakers
  if (config.level >= 4) {
    const ladderResult = detectLadder(board, pos, player, 6)

    if (ladderResult.isLadder) {
      if (ladderResult.works) {
        // Starting a working ladder - good!
        score += 80
      } else {
        // Starting a ladder that doesn't work (opponent escapes) - bad!
        score -= 120
      }
    }

    // Check if this is a ladder breaker
    if (isLadderBreaker(board, pos, player)) {
      score += config.level >= 5 ? 40 : 25
    }
  }

  // Territory positioning: Corners > Sides > Center (金角銀辺草腹)
  // This is fundamental Go strategy
  const distToEdge = Math.min(
    pos.row,
    pos.col,
    board.size - 1 - pos.row,
    board.size - 1 - pos.col
  )

  // Corners (0-2 from edge) are most valuable
  if (distToEdge === 0) {
    score += 5 // On edge
  } else if (distToEdge === 1) {
    score += 15 // One space from edge
  } else if (distToEdge === 2) {
    score += 25 // Two spaces from edge (ideal)
  } else if (distToEdge === 3) {
    score += 20 // Three spaces (still good)
  } else if (distToEdge === 4) {
    score += 10 // Four spaces (decent)
  }
  // Center gets no bonus (or even penalty in opening)

  // Opening: strongly prefer corners and approaches
  if (moveCount < 10) {
    // Penalize center play in opening
    const center = (board.size - 1) / 2
    const distToCenter = Math.abs(pos.row - center) + Math.abs(pos.col - center)
    if (distToCenter < board.size / 4) {
      // Too close to center in opening
      score -= 15
    }
  }

  // Corner and wall control evaluation (level 2+)
  // Level 2: Learn corners/walls are good
  // Level 3-4: Maximize corner/wall control
  // Level 5: Know when to ignore (flexible, sometimes center is better)
  if (config.level >= 2) {
    const cornerWallValue = evaluateCornerWallControl(board, pos, player, config.level)
    score += cornerWallValue
  }

  // INVASION/REDUCTION: Master level (4-5) actively invades opponent territory
  // This makes master level more combative and spread across all quarters
  if (config.level >= 4) {
    const invasionValue = evaluateInvasionReduction(board, pos, player, config.level)
    score += invasionValue

    const reductionValue = evaluateReduction(board, pos, player, config.level)
    score += reductionValue
  }

  // Add random factor based on config
  score += Math.random() * config.randomness * 10

  return score
}

function wouldBeSelfCapture(board: Board, pos: Position, player: 'black' | 'white'): boolean {
  // Get the group that includes this newly placed stone
  const group = getGroup(board, pos.row, pos.col)

  // Count liberties of the entire group
  const liberties = countLiberties(board, group)

  // Self-capture if the group has no liberties
  return liberties === 0
}
