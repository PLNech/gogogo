/**
 * Self-Play Training System for Go AI
 *
 * This module implements a recursive self-play training system that continuously improves
 * the AI by playing games against itself and learning from the outcomes.
 *
 * The system maintains a pool of trained AI configurations and evolves them through:
 * 1. Self-play tournaments between different AI versions
 * 2. Performance evaluation and ranking
 * 3. Progressive improvement of weaker AI configurations
 * 4. Adaptive training based on performance metrics
 */

import { GoMCTS } from './mcts'
import { getAIDecision, getAIMove } from './simpleAI'
import { AIConfig, AI_PRESETS } from './types'
import { createBoard, placeStone, captureStones, countTerritory } from '../go/board'
import { MatchResult, useAIStatsStore } from '../../state/aiStatsStore'
import { v4 as uuidv4 } from 'uuid'

export interface SelfPlayConfig {
  // Number of games to play in each training session
  gamesPerSession: number

  // Number of training sessions to run
  numSessions: number

  // Maximum moves per game
  maxMoves: number

  // Minimum win rate threshold for considering an AI "improved"
  minWinRateThreshold: number

  // How often to evaluate and update AI configurations
  evaluationInterval: number

  // Whether to use progressive training (improving weaker AIs)
  progressiveTraining: boolean
}

export interface TrainingProgress {
  sessionId: number
  aiConfigs: AIConfig[]
  bestConfig: AIConfig
  worstConfig: AIConfig
  avgWinRate: number
  totalGamesPlayed: number
  timestamp: number
}

export class SelfPlayTrainer {
  private config: SelfPlayConfig
  private aiPool: AIConfig[]
  private trainingProgress: TrainingProgress[] = []

  constructor(config: Partial<SelfPlayConfig> = {}) {
    this.config = {
      gamesPerSession: config.gamesPerSession || 100,
      numSessions: config.numSessions || 10,
      maxMoves: config.maxMoves || 200,
      minWinRateThreshold: config.minWinRateThreshold || 0.5,
      evaluationInterval: config.evaluationInterval || 10,
      progressiveTraining: config.progressiveTraining !== undefined ? config.progressiveTraining : true
    }

    // Initialize with the preset configurations
    this.aiPool = Object.values(AI_PRESETS).map(config => ({ ...config }))
  }

  /**
   * Run a complete self-play training session
   */
  async runTraining(): Promise<TrainingProgress[]> {
    console.log('Starting self-play training...')

    for (let session = 0; session < this.config.numSessions; session++) {
      console.log(`Starting training session ${session + 1}/${this.config.numSessions}`)

      // Play games between all AI configurations in the pool
      const sessionResults = await this.playSession(session)

      // Update AI pool based on results
      await this.updateAIPool(sessionResults)

      // Record progress
      const progress = this.recordProgress(session, sessionResults)
      this.trainingProgress.push(progress)

      console.log(`Session ${session + 1} completed. Best config level: ${progress.bestConfig.level}`)

      // Every few sessions, evaluate overall performance
      if ((session + 1) % this.config.evaluationInterval === 0) {
        console.log(`Evaluating overall performance after session ${session + 1}`)
        await this.evaluatePerformance()
      }
    }

    console.log('Self-play training completed.')
    return this.trainingProgress
  }

  /**
   * Play a single training session with current AI pool
   */
  private async playSession(sessionId: number): Promise<Record<string, any>> {
    const results: Record<string, any> = {}
    const sessionGames = this.config.gamesPerSession

    // Create a mapping for all AI configurations to track results
    for (const config of this.aiPool) {
      const signature = this.getConfigSignature(config)
      results[signature] = {
        wins: 0,
        losses: 0,
        draws: 0,
        totalScore: 0,
        totalCaptures: 0,
        totalGames: 0
      }
    }

    // Play games between all pairs of AI configurations
    for (let i = 0; i < sessionGames; i++) {
      // Select two random AIs from the pool
      const ai1 = this.aiPool[Math.floor(Math.random() * this.aiPool.length)]
      const ai2 = this.aiPool[Math.floor(Math.random() * this.aiPool.length)]

      // Play a game
      const gameResult = await this.playSingleGame(ai1, ai2, sessionId, i)

      // Record results for both AIs
      const sig1 = this.getConfigSignature(ai1)
      const sig2 = this.getConfigSignature(ai2)

      if (gameResult.winner === 'black') {
        results[sig1].wins++
        results[sig2].losses++
      } else if (gameResult.winner === 'white') {
        results[sig1].losses++
        results[sig2].wins++
      } else {
        results[sig1].draws++
        results[sig2].draws++
      }

      results[sig1].totalScore += gameResult.blackScore
      results[sig2].totalScore += gameResult.whiteScore
      results[sig1].totalCaptures += gameResult.blackCaptures
      results[sig2].totalCaptures += gameResult.whiteCaptures

      results[sig1].totalGames++
      results[sig2].totalGames++
    }

    return results
  }

  /**
   * Play a single game between two AI configurations
   */
  private async playSingleGame(
    ai1Config: AIConfig,
    ai2Config: AIConfig,
    sessionId: number,
    gameNum: number
  ): Promise<MatchResult> {
    // Create a 9x9 board for the game
    const board = createBoard(9)
    let currentPlayer: 'black' | 'white' = 'black'
    let moveCount = 0
    let previousBoard = null

    // Track captures for each player
    const captures = { black: 0, white: 0 }

    // Play the game until it's over or max moves reached
    while (moveCount < this.config.maxMoves) {
      // Get AI decision for current player
      const aiDecision = getAIDecision(
        board,
        currentPlayer,
        captures,
        currentPlayer === 'black' ? ai1Config : ai2Config,
        moveCount
      )

      if (aiDecision.action === 'pass') {
        // Pass turn
        currentPlayer = currentPlayer === 'black' ? 'white' : 'black'
        moveCount++
        continue
      }

      // Try to place the stone
      const newBoard = placeStone(
        board,
        aiDecision.position!.row,
        aiDecision.position!.col,
        currentPlayer,
        previousBoard
      )

      if (!newBoard) {
        // Invalid move, try again or pass
        currentPlayer = currentPlayer === 'black' ? 'white' : 'black'
        moveCount++
        continue
      }

      // Apply captures
      const captureResult = captureStones(
        newBoard,
        aiDecision.position!.row,
        aiDecision.position!.col,
        currentPlayer
      )

      // Update captures
      captures[currentPlayer] += captureResult.captured

      // Update board state
      previousBoard = board
      board.stones = captureResult.board.stones

      // Switch turns
      currentPlayer = currentPlayer === 'black' ? 'white' : 'black'
      moveCount++

      // Check if game is over (no more empty positions)
      const emptyPositions = this.getEmptyPositions(board)
      if (emptyPositions.length === 0) {
        break
      }
    }

    // Calculate final scores
    const territory = countTerritory(board)
    const blackScore = territory.black
    const whiteScore = territory.white

    // Determine winner
    let winner: 'black' | 'white' | 'draw' = 'draw'
    if (blackScore > whiteScore) {
      winner = 'black'
    } else if (whiteScore > blackScore) {
      winner = 'white'
    }

    // Create match result
    const result: MatchResult = {
      id: `session_${sessionId}_game_${gameNum}_${uuidv4()}`,
      timestamp: Date.now(),
      boardSize: 9,
      maxMoves: this.config.maxMoves,
      winner,
      blackScore,
      whiteScore,
      blackCaptures: captures.black,
      whiteCaptures: captures.white,
      moveCount,
      blackConfig: ai1Config,
      whiteConfig: ai2Config
    }

    // Record the match result
    useAIStatsStore.getState().addMatch(result)

    return result
  }

  /**
   * Update the AI pool based on performance results
   */
  private async updateAIPool(results: Record<string, any>): Promise<void> {
    // Get current stats for each AI config
    const configStats = new Map<string, any>()

    for (const [signature, stats] of Object.entries(results)) {
      const totalGames = stats.totalGames
      const winRate = totalGames > 0 ? stats.wins / totalGames : 0

      configStats.set(signature, {
        signature,
        wins: stats.wins,
        losses: stats.losses,
        draws: stats.draws,
        totalGames,
        winRate,
        avgScore: totalGames > 0 ? stats.totalScore / totalGames : 0,
        avgCaptures: totalGames > 0 ? stats.totalCaptures / totalGames : 0
      })
    }

    // If progressive training is enabled, improve weaker AIs
    if (this.config.progressiveTraining) {
      await this.improveWeakAIs(configStats)
    }

    // Potentially add new configurations based on performance
    await this.addNewConfigs(configStats)
  }

  /**
   * Improve weaker AI configurations based on performance
   */
  private async improveWeakAIs(configStats: Map<string, any>): Promise<void> {
    // Find the weakest AI configurations
    const sortedConfigs = Array.from(configStats.values())
      .sort((a, b) => a.winRate - b.winRate)

    // Improve the bottom 30% of configurations
    const improveCount = Math.max(1, Math.floor(sortedConfigs.length * 0.3))

    for (let i = 0; i < improveCount; i++) {
      const config = this.findConfigBySignature(sortedConfigs[i].signature)
      if (config) {
        // Increase difficulty level and adjust weights
        const improvedConfig = this.improveConfig(config)
        this.replaceConfig(config, improvedConfig)
      }
    }
  }

  /**
   * Add new AI configurations based on successful performance
   */
  private async addNewConfigs(configStats: Map<string, any>): Promise<void> {
    // Find high-performing configurations
    const highPerformers = Array.from(configStats.values())
      .filter(stats => stats.winRate >= this.config.minWinRateThreshold)
      .sort((a, b) => b.winRate - a.winRate)

    // Add a new configuration based on the best performer
    if (highPerformers.length > 0 && this.aiPool.length < 20) {
      const bestConfig = this.findConfigBySignature(highPerformers[0].signature)
      if (bestConfig) {
        const newConfig = this.createAdvancedConfig(bestConfig)
        this.aiPool.push(newConfig)
      }
    }
  }

  /**
   * Improve an AI configuration by increasing difficulty
   */
  private improveConfig(config: AIConfig): AIConfig {
    // Create a copy of the configuration
    const improved = { ...config }

    // Increase difficulty level
    if (improved.level < 5) {
      improved.level = (improved.level + 1) as 1 | 2 | 3 | 4 | 5
    }

    // Increase search depth
    improved.searchDepth = Math.min(5, improved.searchDepth + 1)

    // Reduce randomness slightly
    improved.randomness = Math.max(0.01, improved.randomness - 0.05)

    // Increase weights for better scoring
    improved.captureWeight = Math.min(1000, improved.captureWeight + 50)
    improved.territoryWeight = Math.min(100, improved.territoryWeight + 5)
    improved.influenceWeight = Math.min(100, improved.influenceWeight + 3)

    return improved
  }

  /**
   * Create a more advanced configuration based on a good performer
   */
  private createAdvancedConfig(baseConfig: AIConfig): AIConfig {
    const advanced = { ...baseConfig }

    // Create a new level (if possible)
    if (advanced.level < 5) {
      advanced.level = (advanced.level + 1) as 1 | 2 | 3 | 4 | 5
    } else {
      // Or create a new configuration with even better parameters
      advanced.level = 5
    }

    // Boost all parameters significantly
    advanced.searchDepth = Math.min(5, advanced.searchDepth + 2)
    advanced.randomness = Math.max(0.01, advanced.randomness - 0.1)
    advanced.captureWeight = Math.min(1000, advanced.captureWeight + 100)
    advanced.territoryWeight = Math.min(100, advanced.territoryWeight + 10)
    advanced.influenceWeight = Math.min(100, advanced.influenceWeight + 8)

    return advanced
  }

  /**
   * Replace a configuration in the pool
   */
  private replaceConfig(oldConfig: AIConfig, newConfig: AIConfig): void {
    const index = this.aiPool.findIndex(c => this.configEquals(c, oldConfig))
    if (index !== -1) {
      this.aiPool[index] = newConfig
    }
  }

  /**
   * Record training progress
   */
  private recordProgress(sessionId: number, results: Record<string, any>): TrainingProgress {
    // Calculate overall statistics
    let totalWins = 0
    let totalLosses = 0
    let totalDraws = 0
    let totalGames = 0
    let totalScore = 0
    let totalCaptures = 0
    let bestConfig: AIConfig | null = null
    let worstConfig: AIConfig | null = null
    let maxWinRate = -1
    let minWinRate = Infinity

    // Process all configurations
    for (const [signature, stats] of Object.entries(results)) {
      const winRate = stats.totalGames > 0 ? stats.wins / stats.totalGames : 0
      totalWins += stats.wins
      totalLosses += stats.losses
      totalDraws += stats.draws
      totalGames += stats.totalGames
      totalScore += stats.totalScore
      totalCaptures += stats.totalCaptures

      if (winRate > maxWinRate) {
        maxWinRate = winRate
        bestConfig = this.findConfigBySignature(signature)
      }

      if (winRate < minWinRate) {
        minWinRate = winRate
        worstConfig = this.findConfigBySignature(signature)
      }
    }

    const avgWinRate = totalGames > 0 ? totalWins / totalGames : 0

    return {
      sessionId,
      aiConfigs: this.aiPool,
      bestConfig: bestConfig || AI_PRESETS[1],
      worstConfig: worstConfig || AI_PRESETS[1],
      avgWinRate,
      totalGamesPlayed: totalGames,
      timestamp: Date.now()
    }
  }

  /**
   * Evaluate overall performance of the training system
   */
  private async evaluatePerformance(): Promise<void> {
    console.log('Evaluating overall performance...')

    // Get leaderboard from stats store
    const leaderboard = useAIStatsStore.getState().getLeaderboard()

    if (leaderboard.length > 0) {
      console.log('Top performing AI configurations:')
      leaderboard.slice(0, 5).forEach((stats, index) => {
        console.log(`  ${index + 1}. ${stats.configSignature} - Win Rate: ${(stats.winRate).toFixed(2)}% (${stats.totalGames} games)`)
      })
    }

    // Log current AI pool
    console.log(`Current AI pool size: ${this.aiPool.length}`)
    console.log('AI configurations in pool:')
    this.aiPool.forEach((config, index) => {
      console.log(`  ${index + 1}. Level ${config.level} - Depth: ${config.searchDepth}, Randomness: ${config.randomness.toFixed(2)}`)
    })
  }

  /**
   * Get signature for a configuration
   */
  private getConfigSignature(config: AIConfig): string {
    return `L${config.level}_C${config.captureWeight}_T${config.territoryWeight}_I${config.influenceWeight}_R${config.randomness}`
  }

  /**
   * Find configuration by signature
   */
  private findConfigBySignature(signature: string): AIConfig | null {
    for (const config of this.aiPool) {
      if (this.getConfigSignature(config) === signature) {
        return config
      }
    }
    return null
  }

  /**
   * Check if two configurations are equal
   */
  private configEquals(config1: AIConfig, config2: AIConfig): boolean {
    return (
      config1.level === config2.level &&
      config1.searchDepth === config2.searchDepth &&
      config1.randomness === config2.randomness &&
      config1.captureWeight === config2.captureWeight &&
      config1.territoryWeight === config2.territoryWeight &&
      config1.influenceWeight === config2.influenceWeight
    )
  }

  /**
   * Get empty positions on the board
   */
  private getEmptyPositions(board: any): any[] {
    const positions = []
    for (let row = 0; row < board.size; row++) {
      for (let col = 0; col < board.size; col++) {
        if (board.stones[row][col] === null) {
          positions.push({ row, col })
        }
      }
    }
    return positions
  }

  /**
   * Get current training progress
   */
  getProgress(): TrainingProgress[] {
    return this.trainingProgress
  }

  /**
   * Get current AI pool
   */
  getAIPool(): AIConfig[] {
    return this.aiPool
  }
}