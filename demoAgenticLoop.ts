#!/usr/bin/env node

/**
 * Demo Script for Agentic Loop Training System
 *
 * Demonstrates the core concepts of the agentic training loop
 */

import { SelfPlayAgenticTrainer } from './src/core/ai/agenticTrainer'

async function demonstrateAgenticLoop() {
  console.log('🚀 Demo: Self-Play Agentic Training Loop');
  console.log('=========================================');

  console.log('\n📋 Demonstrating the core concepts of agentic training:');
  console.log('1. Training epochs with self-play games');
  console.log('2. Systematic analysis of outcomes');
  console.log('3. Iterative improvement mechanisms');
  console.log('4. Convergence detection');
  console.log('5. Real-time CLI visualization');

  console.log('\n🎯 Running demonstration...');

  // Create a trainer for demonstration
  const trainer = new SelfPlayAgenticTrainer({
    minWinRate: 0.7,
    maxEpochs: 5,
    stabilityWindow: 2,
    minImprovement: 0.01
  });

  try {
    // Run the demonstration
    const result = await trainer.runAgenticLoop();

    console.log('\n✅ Demonstration completed successfully!');
    console.log(`   Result: ${result.success ? 'Converged' : 'Max epochs reached'}`);
    console.log(`   Epochs: ${result.finalEpochs.length}`);

  } catch (error) {
    console.error('❌ Demo failed:', error);
    process.exit(1);
  }
}

// Run the demonstration when script is executed directly
if (require.main === module) {
  demonstrateAgenticLoop();
}

export { demonstrateAgenticLoop };