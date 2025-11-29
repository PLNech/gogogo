/**
 * Self-Play Training System for Go AI - Agentic Loop with CLI Visualization
 *
 * This module implements a recursive self-play training system that continuously improves
 * the AI through an agentic loop that:
 * 1. Runs training epochs
 * 2. Systematically analyzes game outcomes
 * 3. Builds analysis tools based on findings
 * 4. Repeats until convergence or maximum iterations
 * 5. Provides real-time CLI visualization during training
 */

import { GoMCTS } from './mcts'
import { getAIDecision, getAIMove } from './simpleAI'
import { AIConfig, AI_PRESETS } from './types'
import { createBoard, placeStone, captureStones, countTerritory } from '../go/board'
import { MatchResult, useAIStatsStore } from '../../state/aiStatsStore'
import { v4 as uuidv4 } from 'uuid'
import fs from 'fs'

export interface TrainingEpoch {
  epochNumber: number
  aiConfigs: AIConfig[]
  results: Record<string, any>
  timestamp: number
  convergenceMetric: number
}

export interface ConvergenceCriteria {
  minWinRate: number
  maxEpochs: number
  stabilityWindow: number
  minImprovement: number
}

export class SelfPlayAgenticTrainer {
  private aiPool: AIConfig[]
  private trainingEpochs: TrainingEpoch[] = []
  private convergenceCriteria: ConvergenceCriteria
  private currentEpoch: number = 0
  private convergenceHistory: number[] = []
  private isTraining: boolean = false

  constructor(convergenceCriteria?: Partial<ConvergenceCriteria>) {
    this.convergenceCriteria = {
      minWinRate: convergenceCriteria?.minWinRate || 0.75,
      maxEpochs: convergenceCriteria?.maxEpochs || 50,
      stabilityWindow: convergenceCriteria?.stabilityWindow || 5,
      minImprovement: convergenceCriteria?.minImprovement || 0.02
    }

    // Initialize with the preset configurations
    this.aiPool = Object.values(AI_PRESETS).map(config => ({ ...config }))
  }

  /**
   * Run the complete agentic training loop with CLI visualization
   */
  async runAgenticLoopWithVisualization(): Promise<{ success: boolean; finalEpochs: TrainingEpoch[] }> {
    console.log('\n🚀 Starting Self-Play Agentic Training Loop');
    console.log('=====================================================');

    this.isTraining = true;
    let converged = false;
    let epochCount = 0;

    while (!converged && epochCount < this.convergenceCriteria.maxEpochs) {
      console.log(`\n🔄 Epoch ${epochCount + 1} of ${this.convergenceCriteria.maxEpochs}`);
      console.log('-----------------------------------------------------');

      // Run training epoch
      const epoch = await this.runTrainingEpoch(epochCount);
      this.trainingEpochs.push(epoch);

      // Visualize epoch results
      this.visualizeEpoch(epoch);

      // Analyze convergence
      converged = this.checkConvergence(epoch);

      epochCount++;

      // Update CLI status
      this.updateTrainingStatus(epochCount, converged);

      if (converged) {
        console.log('\n🎉 Training has converged!');
        break;
      }
    }

    console.log('\n🏁 Training loop completed.');
    console.log('=====================================================');
    this.printFinalSummary();

    this.isTraining = false;

    return {
      success: converged || epochCount >= this.convergenceCriteria.maxEpochs,
      finalEpochs: this.trainingEpochs
    };
  }

  /**
   * Run a single training epoch with progress tracking
   */
  private async runTrainingEpoch(epochNumber: number): Promise<TrainingEpoch> {
    console.log(`🎮 Playing games for Epoch ${epochNumber + 1}...`);

    // Play games between all AI configurations in the pool
    const results = await this.playEpochGames(epochNumber);

    // Update AI pool based on results
    await this.updateAIPool(results);

    // Calculate convergence metric
    const avgWinRate = this.calculateAvgWinRate(results);

    // Record epoch data
    const epoch: TrainingEpoch = {
      epochNumber,
      aiConfigs: this.aiPool,
      results,
      timestamp: Date.now(),
      convergenceMetric: avgWinRate
    };

    return epoch;
  }

  /**
   * Play games for the entire epoch with progress tracking
   */
  private async playEpochGames(epochNumber: number): Promise<Record<string, any>> {
    const results: Record<string, any> = {};
    const gamesPerPair = 20; // Number of games per AI pair

    // Initialize results tracking for each configuration
    for (const config of this.aiPool) {
      const signature = this.getConfigSignature(config);
      results[signature] = {
        wins: 0,
        losses: 0,
        draws: 0,
        totalScore: 0,
        totalCaptures: 0,
        totalGames: 0,
        mctsData: [] as any[]
      };
    }

    // Play games between all pairs of AI configurations
    const totalPairs = this.aiPool.length * (this.aiPool.length + 1) / 2;
    let processedPairs = 0;

    console.log(`📊 Playing ${totalPairs} unique AI matchups, ${gamesPerPair} games each`);

    for (let i = 0; i < this.aiPool.length; i++) {
      for (let j = i; j < this.aiPool.length; j++) {
        const ai1 = this.aiPool[i];
        const ai2 = this.aiPool[j];

        // Play multiple games between this pair
        console.log(`  🔁 Playing ${gamesPerPair} games between AI ${i+1} (L${ai1.level}) vs AI ${j+1} (L${ai2.level})`);

        for (let game = 0; game < gamesPerPair; game++) {
          const gameResult = await this.playSingleGame(ai1, ai2, epochNumber, game);

          // Record results for both AIs
          const sig1 = this.getConfigSignature(ai1);
          const sig2 = this.getConfigSignature(ai2);

          if (gameResult.winner === 'black') {
            results[sig1].wins++;
            results[sig2].losses++;
          } else if (gameResult.winner === 'white') {
            results[sig1].losses++;
            results[sig2].wins++;
          } else {
            results[sig1].draws++;
            results[sig2].draws++;
          }

          results[sig1].totalScore += gameResult.blackScore;
          results[sig2].totalScore += gameResult.whiteScore;
          results[sig1].totalCaptures += gameResult.blackCaptures;
          results[sig2].totalCaptures += gameResult.whiteCaptures;
          results[sig1].totalGames++;
          results[sig2].totalGames++;

          // Store MCTS data for analysis
          if (gameResult.blackConfig.mctsData) {
            results[sig1].mctsData.push(gameResult.blackConfig.mctsData);
          }
          if (gameResult.whiteConfig.mctsData) {
            results[sig2].mctsData.push(gameResult.whiteConfig.mctsData);
          }
        }

        processedPairs++;
        // Show progress
        const progress = ((processedPairs / totalPairs) * 100).toFixed(1);
        console.log(`    Progress: ${progress}% (${processedPairs}/${totalPairs} pairs)`);
      }
    }

    console.log('✅ Completed all games for this epoch');
    return results;
  }

  /**
   * Play a single game between two AI configurations
   */
  private async playSingleGame(
    ai1Config: AIConfig,
    ai2Config: AIConfig,
    epochNumber: number,
    gameNum: number
  ): Promise<MatchResult> {
    // Create a 9x9 board for the game
    const board = createBoard(9);
    let currentPlayer: 'black' | 'white' = 'black';
    let moveCount = 0;
    let previousBoard = null;

    // Track captures for each player
    const captures = { black: 0, white: 0 };

    // Play the game until it's over or max moves reached
    while (moveCount < 200) {
      // Get AI decision for current player
      const aiDecision = getAIDecision(
        board,
        currentPlayer,
        captures,
        currentPlayer === 'black' ? ai1Config : ai2Config,
        moveCount
      );

      if (aiDecision.action === 'pass') {
        // Pass turn
        currentPlayer = currentPlayer === 'black' ? 'white' : 'black';
        moveCount++;
        continue;
      }

      // Try to place the stone
      const newBoard = placeStone(
        board,
        aiDecision.position!.row,
        aiDecision.position!.col,
        currentPlayer,
        previousBoard
      );

      if (!newBoard) {
        // Invalid move, try again or pass
        currentPlayer = currentPlayer === 'black' ? 'white' : 'black';
        moveCount++;
        continue;
      }

      // Apply captures
      const captureResult = captureStones(
        newBoard,
        aiDecision.position!.row,
        aiDecision.position!.col,
        currentPlayer
      );

      // Update captures
      captures[currentPlayer] += captureResult.captured;

      // Update board state
      previousBoard = board;
      board.stones = captureResult.board.stones;

      // Switch turns
      currentPlayer = currentPlayer === 'black' ? 'white' : 'black';
      moveCount++;

      // Check if game is over (no more empty positions)
      const emptyPositions = this.getEmptyPositions(board);
      if (emptyPositions.length === 0) {
        break;
      }
    }

    // Calculate final scores
    const territory = countTerritory(board);
    const blackScore = territory.black;
    const whiteScore = territory.white;

    // Determine winner
    let winner: 'black' | 'white' | 'draw' = 'draw';
    if (blackScore > whiteScore) {
      winner = 'black';
    } else if (whiteScore > blackScore) {
      winner = 'white';
    }

    // Create match result
    const result: MatchResult = {
      id: `epoch_${epochNumber}_game_${gameNum}_${uuidv4()}`,
      timestamp: Date.now(),
      boardSize: 9,
      maxMoves: 200,
      winner,
      blackScore,
      whiteScore,
      blackCaptures: captures.black,
      whiteCaptures: captures.white,
      moveCount,
      blackConfig: ai1Config,
      whiteConfig: ai2Config
    };

    // Record the match result
    useAIStatsStore.getState().addMatch(result);

    return result;
  }

  /**
   * Update the AI pool based on performance results
   */
  private async updateAIPool(results: Record<string, any>): Promise<void> {
    // Get current stats for each AI config
    const configStats = new Map<string, any>();

    for (const [signature, stats] of Object.entries(results)) {
      const totalGames = stats.totalGames;
      const winRate = totalGames > 0 ? stats.wins / totalGames : 0;

      configStats.set(signature, {
        signature,
        wins: stats.wins,
        losses: stats.losses,
        draws: stats.draws,
        totalGames,
        winRate,
        avgScore: totalGames > 0 ? stats.totalScore / totalGames : 0,
        avgCaptures: totalGames > 0 ? stats.totalCaptures / totalGames : 0
      });
    }

    // Improve weaker AIs
    await this.improveWeakAIs(configStats);

    // Add new configurations based on performance
    await this.addNewConfigs(configStats);
  }

  /**
   * Improve weaker AI configurations based on performance
   */
  private async improveWeakAIs(configStats: Map<string, any>): Promise<void> {
    // Find the weakest AI configurations
    const sortedConfigs = Array.from(configStats.values())
      .sort((a, b) => a.winRate - b.winRate);

    // Improve the bottom 30% of configurations
    const improveCount = Math.max(1, Math.floor(sortedConfigs.length * 0.3));

    if (improveCount > 0) {
      console.log(`🔧 Improving ${improveCount} weaker AI configurations...`);
    }

    for (let i = 0; i < improveCount; i++) {
      const config = this.findConfigBySignature(sortedConfigs[i].signature);
      if (config) {
        // Increase difficulty level and adjust weights
        const improvedConfig = this.improveConfig(config);
        this.replaceConfig(config, improvedConfig);

        console.log(`  ➕ Improved AI from level ${config.level} to level ${improvedConfig.level}`);
      }
    }
  }

  /**
   * Add new AI configurations based on successful performance
   */
  private async addNewConfigs(configStats: Map<string, any>): Promise<void> {
    // Find high-performing configurations
    const highPerformers = Array.from(configStats.values())
      .filter(stats => stats.winRate >= 0.6) // Minimum 60% win rate
      .sort((a, b) => b.winRate - a.winRate);

    // Add a new configuration based on the best performer
    if (highPerformers.length > 0 && this.aiPool.length < 20) {
      const bestConfig = this.findConfigBySignature(highPerformers[0].signature);
      if (bestConfig) {
        const newConfig = this.createAdvancedConfig(bestConfig);
        this.aiPool.push(newConfig);
        console.log(`🆕 Added new AI configuration (level ${newConfig.level})`);
      }
    }
  }

  /**
   * Improve an AI configuration by increasing difficulty
   */
  private improveConfig(config: AIConfig): AIConfig {
    // Create a copy of the configuration
    const improved = { ...config };

    // Increase difficulty level
    if (improved.level < 5) {
      improved.level = (improved.level + 1) as 1 | 2 | 3 | 4 | 5;
    }

    // Increase search depth
    improved.searchDepth = Math.min(5, improved.searchDepth + 1);

    // Reduce randomness slightly
    improved.randomness = Math.max(0.01, improved.randomness - 0.05);

    // Increase weights for better scoring
    improved.captureWeight = Math.min(1000, improved.captureWeight + 50);
    improved.territoryWeight = Math.min(100, improved.territoryWeight + 5);
    improved.influenceWeight = Math.min(100, improved.influenceWeight + 3);

    return improved;
  }

  /**
   * Create a more advanced configuration based on a good performer
   */
  private createAdvancedConfig(baseConfig: AIConfig): AIConfig {
    const advanced = { ...baseConfig };

    // Create a new level (if possible)
    if (advanced.level < 5) {
      advanced.level = (advanced.level + 1) as 1 | 2 | 3 | 4 | 5;
    } else {
      // Or create a new configuration with even better parameters
      advanced.level = 5;
    }

    // Boost all parameters significantly
    advanced.searchDepth = Math.min(5, advanced.searchDepth + 2);
    advanced.randomness = Math.max(0.01, advanced.randomness - 0.1);
    advanced.captureWeight = Math.min(1000, advanced.captureWeight + 100);
    advanced.territoryWeight = Math.min(100, advanced.territoryWeight + 10);
    advanced.influenceWeight = Math.min(100, advanced.influenceWeight + 8);

    return advanced;
  }

  /**
   * Replace a configuration in the pool
   */
  private replaceConfig(oldConfig: AIConfig, newConfig: AIConfig): void {
    const index = this.aiPool.findIndex(c => this.configEquals(c, oldConfig));
    if (index !== -1) {
      this.aiPool[index] = newConfig;
    }
  }

  /**
   * Check for convergence criteria
   */
  private checkConvergence(currentEpoch: TrainingEpoch): boolean {
    // Store current win rate for convergence tracking
    this.convergenceHistory.push(currentEpoch.convergenceMetric);

    // If we don't have enough data, continue training
    if (this.convergenceHistory.length < this.convergenceCriteria.stabilityWindow) {
      return false;
    }

    // Check if we have enough recent epochs to evaluate
    const recentHistory = this.convergenceHistory.slice(-this.convergenceCriteria.stabilityWindow);

    // Calculate variance in recent win rates
    const avg = recentHistory.reduce((a, b) => a + b, 0) / recentHistory.length;
    const variance = recentHistory.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / recentHistory.length;
    const stdDev = Math.sqrt(variance);

    // Check if we've reached target win rate
    if (currentEpoch.convergenceMetric >= this.convergenceCriteria.minWinRate) {
      console.log(`🎯 Reached target win rate of ${this.convergenceCriteria.minWinRate * 100}%`);
      return true;
    }

    // Check if variance is low enough (stable performance)
    const isStable = stdDev < 0.05; // Less than 5% variance

    // Check if improvement is minimal
    if (this.convergenceHistory.length >= 2) {
      const lastImprovement = Math.abs(
        this.convergenceHistory[this.convergenceHistory.length - 1] -
        this.convergenceHistory[this.convergenceHistory.length - 2]
      );

      if (lastImprovement < this.convergenceCriteria.minImprovement) {
        console.log(`📉 Minimal improvement detected (${lastImprovement.toFixed(4)})`);
        return true;
      }
    }

    // If stable and we've seen enough epochs, consider it converged
    if (isStable && this.convergenceHistory.length >= this.convergenceCriteria.stabilityWindow) {
      console.log(`📊 Convergence achieved: Stable performance with variance ${stdDev.toFixed(4)}`);
      return true;
    }

    return false;
  }

  /**
   * Visualize the results of a training epoch
   */
  private visualizeEpoch(epoch: TrainingEpoch): void {
    console.log(`📈 Epoch ${epoch.epochNumber + 1} Results:`);

    // Display AI pool status
    console.log('   AI Pool Status:');
    this.aiPool.forEach((config, index) => {
      console.log(`     AI ${index + 1}: Level ${config.level} (Depth: ${config.searchDepth}, Rand: ${config.randomness.toFixed(2)})`);
    });

    // Display convergence metric
    console.log(`   Average Win Rate: ${(epoch.convergenceMetric * 100).toFixed(2)}%`);

    // Show top performers
    const topConfigs = this.getTopPerformingConfigs(epoch.results);
    if (topConfigs.length > 0) {
      console.log('   Top Performing AIs:');
      topConfigs.slice(0, 3).forEach((config, index) => {
        console.log(`     ${index + 1}. ${config.signature} - Win Rate: ${(config.winRate * 100).toFixed(1)}%`);
      });
    }
  }

  /**
   * Update training status in CLI
   */
  private updateTrainingStatus(epoch: number, converged: boolean): void {
    const statusLine = `Status: Epoch ${epoch}/${this.convergenceCriteria.maxEpochs}`;
    if (converged) {
      console.log(`   ${statusLine} - CONVERGED!`);
    } else {
      console.log(`   ${statusLine} - Training...`);
    }
  }

  /**
   * Print final training summary
   */
  private printFinalSummary(): void {
    console.log('\n📊 FINAL SUMMARY');
    console.log('=====================================================');

    if (this.trainingEpochs.length > 0) {
      const finalEpoch = this.trainingEpochs[this.trainingEpochs.length - 1];
      console.log(`Total Epochs: ${this.trainingEpochs.length}`);
      console.log(`Final Average Win Rate: ${(finalEpoch.convergenceMetric * 100).toFixed(2)}%`);
      console.log(`Final AI Pool Size: ${this.aiPool.length}`);

      // Show final AI configurations
      console.log('\nFinal AI Configurations:');
      this.aiPool.forEach((config, index) => {
        console.log(`  ${index + 1}. Level ${config.level} - Depth: ${config.searchDepth}, Rand: ${config.randomness.toFixed(2)}`);
      });

      // Show convergence history
      if (this.convergenceHistory.length > 0) {
        console.log(`\nConvergence History (last 5): ${this.convergenceHistory.slice(-5).map(v => (v * 100).toFixed(1)).join(', ')}`);
      }
    } else {
      console.log('No training epochs completed.');
    }
  }

  /**
   * Get top performing configurations
   */
  private getTopPerformingConfigs(results: Record<string, any>): any[] {
    const configs = [];

    for (const [signature, stats] of Object.entries(results)) {
      const winRate = stats.totalGames > 0 ? stats.wins / stats.totalGames : 0;
      configs.push({
        signature,
        winRate,
        wins: stats.wins,
        totalGames: stats.totalGames
      });
    }

    return configs.sort((a, b) => b.winRate - a.winRate);
  }

  /**
   * Calculate average win rate across all configurations
   */
  private calculateAvgWinRate(results: Record<string, any>): number {
    let totalWins = 0;
    let totalGames = 0;

    for (const [_, stats] of Object.entries(results)) {
      totalWins += stats.wins;
      totalGames += stats.totalGames;
    }

    return totalGames > 0 ? totalWins / totalGames : 0;
  }

  /**
   * Get signature for a configuration
   */
  private getConfigSignature(config: AIConfig): string {
    return `L${config.level}_C${config.captureWeight}_T${config.territoryWeight}_I${config.influenceWeight}_R${config.randomness}`;
  }

  /**
   * Find configuration by signature
   */
  private findConfigBySignature(signature: string): AIConfig | null {
    for (const config of this.aiPool) {
      if (this.getConfigSignature(config) === signature) {
        return config;
      }
    }
    return null;
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
    );
  }

  /**
   * Get empty positions on the board
   */
  private getEmptyPositions(board: any): any[] {
    const positions = [];
    for (let row = 0; row < board.size; row++) {
      for (let col = 0; col < board.size; col++) {
        if (board.stones[row][col] === null) {
          positions.push({ row, col });
        }
      }
    }
    return positions;
  }

  /**
   * Get current training progress
   */
  getTrainingEpochs(): TrainingEpoch[] {
    return this.trainingEpochs;
  }

  /**
   * Get current AI pool
   */
  getAIPool(): AIConfig[] {
    return this.aiPool;
  }

  /**
   * Export training results to file
   */
  exportResults(filename: string = 'training_results.json'): void {
    const exportData = {
      trainingEpochs: this.trainingEpochs,
      finalAIPool: this.aiPool,
      convergenceCriteria: this.convergenceCriteria,
      timestamp: new Date().toISOString()
    };

    fs.writeFileSync(filename, JSON.stringify(exportData, null, 2));
    console.log(`💾 Training results exported to ${filename}`);
  }
}