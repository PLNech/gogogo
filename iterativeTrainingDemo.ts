#!/usr/bin/env node

/**
 * Iterative Training Demo for Self-Play Loop
 *
 * This demonstrates the complete iterative training process
 * with multiple cycles of run/review/dev/iterate
 */

import { IterativeTrainingRunner } from './src/core/ai/iterativeTrainingRunner'

async function runIterativeTrainingDemo() {
  console.log('🚀 Iterative Self-Play Training Demo');
  console.log('=====================================');

  console.log('\n📋 Training Cycle Process:');
  console.log('1. RUN: Execute training epoch with AI self-play');
  console.log('2. REVIEW: Analyze game outcomes and performance');
  console.log('3. DEV: Implement improvements to AI configurations');
  console.log('4. ITERATE: Repeat with enhanced AI for next cycle');

  console.log('\n🎯 Running 5-cycle iterative training...');

  // Create training runner
  const runner = new IterativeTrainingRunner(5, 0.8);

  try {
    // Run the complete iterative training
    const iterations = await runner.runIterativeTraining();

    console.log('\n✅ Iterative training completed successfully!');
    console.log(`   Total cycles: ${iterations.length}`);
    console.log(`   Final performance: ${(iterations[iterations.length-1]?.performance.winRate * 100 || 0).toFixed(2)}% win rate`);

    return iterations;
  } catch (error) {
    console.error('❌ Training failed:', error);
    process.exit(1);
  }
}

// Run the demo when script is executed directly
if (require.main === module) {
  runIterativeTrainingDemo();
}

export { runIterativeTrainingDemo };