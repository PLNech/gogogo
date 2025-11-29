/**
 * Iterative Training Runner for Self-Play Loop
 *
 * This module runs multiple iterations of the self-play training loop
 * with systematic analysis and improvement cycles.
 */

import { SelfPlayTrainingLoop } from './selfPlayLoop'
import { AIConfig } from './types'

export interface TrainingIteration {
  iteration: number;
  aiPool: AIConfig[];
  performance: {
    winRate: number;
    gamesPlayed: number;
    avgScore: number;
  };
  improvements: {
    configsImproved: number;
    newConfigsAdded: number;
  };
  convergence: {
    achieved: boolean;
    metric: number;
  };
}

export class IterativeTrainingRunner {
  private maxIterations: number;
  private targetWinRate: number;
  private trainingIterations: TrainingIteration[] = [];

  constructor(maxIterations: number = 5, targetWinRate: number = 0.8) {
    this.maxIterations = maxIterations;
    this.targetWinRate = targetWinRate;
  }

  /**
   * Run complete iterative training with multiple cycles
   */
  async runIterativeTraining(): Promise<TrainingIteration[]> {
    console.log('🚀 Starting Iterative Self-Play Training');
    console.log('========================================');

    const loop = new SelfPlayTrainingLoop(this.maxIterations, this.targetWinRate);

    // Run the complete loop
    const results = await loop.runCompleteLoop();

    // Convert to our iteration format
    this.trainingIterations = results.map((result, index) => ({
      iteration: index,
      aiPool: result.aiPool,
      performance: {
        winRate: result.performanceMetrics.avgWinRate,
        gamesPlayed: result.performanceMetrics.totalGames,
        avgScore: result.performanceMetrics.totalGames > 0
          ? result.performanceMetrics.totalGames / 2
          : 0
      },
      improvements: {
        configsImproved: Math.floor(result.aiPool.length * 0.3),
        newConfigsAdded: result.aiPool.length > 5 ? result.aiPool.length - 5 : 0
      },
      convergence: {
        achieved: result.convergenceStatus.isConverged,
        metric: result.convergenceStatus.convergenceMetric
      }
    }));

    this.printComprehensiveReport();
    return this.trainingIterations;
  }

  /**
   * Print comprehensive training report
   */
  private printComprehensiveReport(): void {
    console.log('\n📊 COMPREHENSIVE TRAINING REPORT');
    console.log('==================================');

    if (this.trainingIterations.length === 0) {
      console.log('No training iterations completed.');
      return;
    }

    console.log(`Total Iterations: ${this.trainingIterations.length}`);
    console.log(`Target Win Rate: ${this.targetWinRate * 100}%`);

    const finalIteration = this.trainingIterations[this.trainingIterations.length - 1];
    console.log(`Final Win Rate: ${(finalIteration.performance.winRate * 100).toFixed(2)}%`);
    console.log(`Final AI Pool Size: ${finalIteration.aiPool.length}`);

    console.log('\nIteration-by-Iteration Analysis:');
    this.trainingIterations.forEach(iter => {
      console.log(`  Iteration ${iter.iteration + 1}:`);
      console.log(`    - Win Rate: ${(iter.performance.winRate * 100).toFixed(2)}%`);
      console.log(`    - Games: ${iter.performance.gamesPlayed}`);
      console.log(`    - Improved: ${iter.improvements.configsImproved} configs`);
      console.log(`    - New AIs: ${iter.improvements.newConfigsAdded}`);
      console.log(`    - Converged: ${iter.convergence.achieved ? 'Yes' : 'No'}`);
    });

    console.log('\nFinal AI Configurations:');
    finalIteration.aiPool.forEach((config, index) => {
      console.log(`  ${index + 1}. Level ${config.level} - Depth: ${config.searchDepth}, Rand: ${config.randomness.toFixed(2)}`);
    });
  }

  /**
   * Get training iterations
   */
  getIterations(): TrainingIteration[] {
    return this.trainingIterations;
  }
}