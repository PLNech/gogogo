#!/usr/bin/env node

/**
 * Self-Play Training Runner with CLI Visualization
 *
 * This script runs the self-play training system with real-time CLI updates
 */

import { SelfPlayAgenticTrainer } from './src/core/ai/selfPlayAgenticTrainer';

async function runTraining() {
  console.log('🚀 Initializing Self-Play Training System...');

  // Configure convergence criteria
  const convergenceCriteria = {
    minWinRate: 0.80,      // Target win rate of 80%
    maxEpochs: 20,        // Maximum epochs to run
    stabilityWindow: 3,   // Number of epochs to consider for stability
    minImprovement: 0.01  // Minimum improvement threshold
  };

  // Create trainer instance
  const trainer = new SelfPlayAgenticTrainer(convergenceCriteria);

  try {
    console.log('Starting training with CLI visualization...\n');

    // Run the agentic training loop
    const result = await trainer.runAgenticLoopWithVisualization();

    console.log('\n📋 Training Complete!');
    console.log(`Success: ${result.success}`);
    console.log(`Epochs completed: ${result.finalEpochs.length}`);

    // Export results
    trainer.exportResults('selfplay_training_results.json');

  } catch (error) {
    console.error('❌ Training failed:', error);
    process.exit(1);
  }
}

// Run the training when script is executed directly
if (require.main === module) {
  runTraining();
}

export { runTraining };