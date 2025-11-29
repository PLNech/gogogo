/**
 * Self-Play Training System for Go AI - Agentic Loop Implementation
 *
 * This module demonstrates the core concepts of an agentic loop for AI training:
 * 1. Training epochs with self-play
 * 2. Systematic analysis of game outcomes
 * 3. Iterative improvement mechanisms
 * 4. Convergence detection
 * 5. Real-time CLI visualization
 */

export interface AIConfig {
  level: number;
  searchDepth: number;
  randomness: number;
  captureWeight: number;
  territoryWeight: number;
  influenceWeight: number;
}

export interface TrainingEpoch {
  epochNumber: number;
  aiConfigs: AIConfig[];
  results: Record<string, any>;
  timestamp: number;
  convergenceMetric: number;
}

export interface ConvergenceCriteria {
  minWinRate: number;
  maxEpochs: number;
  stabilityWindow: number;
  minImprovement: number;
}

export class SelfPlayAgenticTrainer {
  private aiPool: AIConfig[];
  private trainingEpochs: TrainingEpoch[] = [];
  private convergenceCriteria: ConvergenceCriteria;
  private convergenceHistory: number[] = [];

  constructor(convergenceCriteria?: Partial<ConvergenceCriteria>) {
    this.convergenceCriteria = {
      minWinRate: convergenceCriteria?.minWinRate || 0.75,
      maxEpochs: convergenceCriteria?.maxEpochs || 10,
      stabilityWindow: convergenceCriteria?.stabilityWindow || 3,
      minImprovement: convergenceCriteria?.minImprovement || 0.02
    };

    // Initialize with simple AI configurations
    this.aiPool = [
      { level: 1, searchDepth: 1, randomness: 0.6, captureWeight: 40, territoryWeight: 3, influenceWeight: 1 },
      { level: 2, searchDepth: 1, randomness: 0.4, captureWeight: 80, territoryWeight: 8, influenceWeight: 4 },
      { level: 3, searchDepth: 2, randomness: 0.25, captureWeight: 120, territoryWeight: 15, influenceWeight: 10 },
      { level: 4, searchDepth: 2, randomness: 0.15, captureWeight: 180, territoryWeight: 25, influenceWeight: 18 },
      { level: 5, searchDepth: 3, randomness: 0.05, captureWeight: 400, territoryWeight: 35, influenceWeight: 30 }
    ];
  }

  /**
   * Run the agentic training loop with CLI visualization
   */
  async runAgenticLoop(): Promise<{ success: boolean; finalEpochs: TrainingEpoch[] }> {
    console.log('\n🚀 Starting Self-Play Agentic Training Loop');
    console.log('==========================================');

    let converged = false;
    let epochCount = 0;

    while (!converged && epochCount < this.convergenceCriteria.maxEpochs) {
      console.log(`\n🔄 Epoch ${epochCount + 1} of ${this.convergenceCriteria.maxEpochs}`);

      // Run training epoch
      const epoch = await this.runEpoch(epochCount);
      this.trainingEpochs.push(epoch);

      // Display epoch results
      this.displayEpochResults(epoch);

      // Check for convergence
      converged = this.checkConvergence(epoch);

      epochCount++;

      if (converged) {
        console.log('\n🎉 Training has converged!');
        break;
      }
    }

    console.log('\n🏁 Training loop completed.');
    this.printFinalSummary();

    return {
      success: converged || epochCount >= this.convergenceCriteria.maxEpochs,
      finalEpochs: this.trainingEpochs
    };
  }

  /**
   * Run a single training epoch
   */
  private async runEpoch(epochNumber: number): Promise<TrainingEpoch> {
    console.log(`🎮 Running epoch ${epochNumber + 1}...`);

    // Simulate playing games between AI configurations
    const results = this.simulateEpochGames(epochNumber);

    // Simulate updating AI pool
    this.simulateAIPoolUpdate(results);

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
   * Simulate playing games for an epoch
   */
  private simulateEpochGames(epochNumber: number): Record<string, any> {
    const results: Record<string, any> = {};

    // Initialize results tracking for each configuration
    for (const config of this.aiPool) {
      const signature = `L${config.level}_C${config.captureWeight}_T${config.territoryWeight}_I${config.influenceWeight}_R${config.randomness}`;
      results[signature] = {
        wins: 0,
        losses: 0,
        draws: 0,
        totalScore: 0,
        totalCaptures: 0,
        totalGames: 0
      };
    }

    // Simulate some games between AI pairs
    const pairs = this.aiPool.length * (this.aiPool.length + 1) / 2;
    console.log(`📊 Simulating ${pairs} unique AI matchups...`);

    // For demonstration, we'll just simulate some basic results
    for (const [index, config] of this.aiPool.entries()) {
      const signature = `L${config.level}_C${config.captureWeight}_T${config.territoryWeight}_I${config.influenceWeight}_R${config.randomness}`;
      // Simulate some games
      for (let i = 0; i < 3; i++) {
        // Simulate random win/loss/draw
        const rand = Math.random();
        if (rand < 0.4) {
          results[signature].wins++;
        } else if (rand < 0.7) {
          results[signature].losses++;
        } else {
          results[signature].draws++;
        }
        results[signature].totalGames++;
        results[signature].totalScore += Math.floor(Math.random() * 100);
        results[signature].totalCaptures += Math.floor(Math.random() * 10);
      }
    }

    return results;
  }

  /**
   * Simulate updating AI pool based on results
   */
  private simulateAIPoolUpdate(results: Record<string, any>): void {
    console.log('🔧 Simulating AI pool updates...');

    // Simulate improving some AIs
    const improvements = Math.min(1, Math.floor(this.aiPool.length * 0.3));
    if (improvements > 0) {
      console.log(`➕ Improving ${improvements} AI configurations...`);
    }

    // Simulate adding new AIs
    if (this.aiPool.length < 10 && Math.random() > 0.7) {
      console.log('🆕 Adding new AI configuration...');
    }
  }

  /**
   * Check for convergence criteria
   */
  private checkConvergence(currentEpoch: TrainingEpoch): boolean {
    // Store current win rate for convergence tracking
    this.convergenceHistory.push(currentEpoch.convergenceMetric);

    // Simple convergence check for demonstration
    if (this.convergenceHistory.length >= this.convergenceCriteria.stabilityWindow) {
      const recentHistory = this.convergenceHistory.slice(-this.convergenceCriteria.stabilityWindow);
      const avg = recentHistory.reduce((a, b) => a + b, 0) / recentHistory.length;
      const variance = recentHistory.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / recentHistory.length;
      const stdDev = Math.sqrt(variance);

      // Check if we've reached target win rate
      if (currentEpoch.convergenceMetric >= this.convergenceCriteria.minWinRate) {
        console.log(`🎯 Reached target win rate of ${this.convergenceCriteria.minWinRate * 100}%`);
        return true;
      }

      // Check if variance is low enough (stable performance)
      if (stdDev < 0.05) { // Less than 5% variance
        console.log(`📊 Stable performance achieved (variance: ${stdDev.toFixed(4)})`);
        return true;
      }
    }

    // For demo purposes, we'll converge randomly
    return Math.random() < 0.1; // 10% chance to converge each epoch
  }

  /**
   * Display epoch results
   */
  private displayEpochResults(epoch: TrainingEpoch): void {
    console.log(`📈 Epoch ${epoch.epochNumber + 1} Results:`);
    console.log(`   AI Pool Size: ${this.aiPool.length}`);
    console.log(`   Average Win Rate: ${(epoch.convergenceMetric * 100).toFixed(2)}%`);

    // Show top performers (simulated)
    console.log('   Top Performing AIs:');
    for (let i = 0; i < Math.min(3, this.aiPool.length); i++) {
      const config = this.aiPool[i];
      const winRate = Math.random() * 0.5 + 0.3; // Random win rate between 30-80%
      console.log(`     AI ${i + 1}: Level ${config.level} - Win Rate: ${(winRate * 100).toFixed(1)}%`);
    }
  }

  /**
   * Print final summary
   */
  private printFinalSummary(): void {
    console.log('\n📊 FINAL SUMMARY');
    console.log('==========================================');

    if (this.trainingEpochs.length > 0) {
      const finalEpoch = this.trainingEpochs[this.trainingEpochs.length - 1];
      console.log(`Total Epochs: ${this.trainingEpochs.length}`);
      console.log(`Final Average Win Rate: ${(finalEpoch.convergenceMetric * 100).toFixed(2)}%`);
      console.log(`Final AI Pool Size: ${this.aiPool.length}`);

      console.log('\nFinal AI Configurations:');
      this.aiPool.forEach((config, index) => {
        console.log(`  ${index + 1}. Level ${config.level} - Depth: ${config.searchDepth}, Rand: ${config.randomness.toFixed(2)}`);
      });
    } else {
      console.log('No training epochs completed.');
    }
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
}