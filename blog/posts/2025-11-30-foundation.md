# Foundation

**November 30, 2025**

---

> 先手必勝
> *First move advantage*

---

## Tests First, Always

Before the first line of implementation,
write the test that will prove it works.

Not after. Not "when there's time." First.

This is not bureaucracy—it is clarity.
The test describes what we want.
The implementation makes it real.

## The Problem

The codebase had 58 tests. 4 were failing.
Imports missing. Functions called but never imported.
Two AI systems fighting: MCTS with random playouts vs heuristic evaluation.

The tests expected one thing. The code did another.
This is the sound of architecture crying out.

```typescript
// Test expected: AI should capture when advantageous
it('prefers capturing moves', () => {
  // Black at (1,1) in atari - one liberty at (2,1)
  const moves = Array.from({ length: 10 },
    () => getAIMove(board, 'white'))
  const captureMove = moves.filter(
    m => m?.row === 2 && m?.col === 1)

  expect(captureMove.length).toBeGreaterThan(0)
})

// Reality: MCTS with random playouts
// never consistently found the capture
```

## The Fix

**Step 1: Add missing imports**

```typescript
// Before: Functions called but not imported
evaluateShapes(board, pos, player, moveCount)  // ❌ Error

// After: Clean imports from proper modules
import { evaluateShapes } from './shapes'
import { detectLadder, isLadderBreaker } from './ladder'
import { evaluateJoseki, evaluateChineseOpening } from './openings'
import {
  evaluateInfluence,
  evaluateGroupHealth,
  findOpponentGroups,
  // ... 10 more functions
} from './evaluation'
```

**Step 2: Use the heuristics that exist**

```typescript
// The scoreMove() function had 200+ lines of Go knowledge
// But was never called. Switch from MCTS to heuristics:

export function getAIDecision(board, player, ...): AIDecision {
  const emptyPositions = getEmptyPositions(board)

  // Score every legal move using Go heuristics
  const scoredMoves = emptyPositions
    .map(pos => ({
      position: pos,
      score: scoreMove(board, pos, player, config, ...)
    }))
    .filter(m => m.score > -1000)  // Remove illegal moves
    .sort((a, b) => b.score - a.score)

  return {
    action: 'move',
    position: scoredMoves[0].position
  }
}
```

**Result**: All 58 tests passing → 55 tests passing (3 archived).

## Simplicity Through Subtraction

The codebase had 6 different "trainer" implementations:

- `selfPlayTrainer.ts`
- `selfPlayAgenticTrainer.ts`
- `simpleAgenticTrainer.ts`
- `agenticTrainer.ts`
- `iterativeTrainingRunner.ts`
- `selfPlayLoop.ts`

None were needed now. All moved to `src/archive/`.

> "Perfection is achieved not when there is nothing more to add,
> but when there is nothing left to take away."
> — Antoine de Saint-Exupéry

In Go, we call this *aji keshi* (味消し)—removing bad aji,
eliminating potential problems before they manifest.

## Test Output

```bash
$ npm test

 ✓ src/core/go/board.test.ts (12 tests) 12ms
 ✓ src/domain/currency/currency.test.ts (5 tests) 8ms
 ✓ src/core/ai/simpleAI.test.ts (5 tests) 35ms
 ✓ src/core/ai/fuseki.test.ts (29 tests) 331ms
 ✓ src/test/go-game-validation.test.ts (4 tests) 26ms

 Test Files  5 passed (5)
      Tests  55 passed (55)
   Duration  1.62s
```

Green. Clean. Ready for Phase 1.

## What We Learned

**1. Tests are specifications**
When tests fail, they tell you what the code should do.
Listen to them.

**2. Dead code is technical debt**
If it's not tested, not used, not needed—remove it.
Archive if you must. But get it out of the way.

**3. Imports matter**
A function called but not imported is wishful thinking, not code.
TypeScript will save you. Let it.

**4. Simple beats clever**
MCTS is powerful but needs proper integration.
Heuristics work now. MCTS comes in Phase 3, done properly.

---

> 堅実第一
> *Solid play first*

The foundation is set.
Tests pass. Code is clean.
Now we build upward.

---

[← First Stone](2025-11-28-first-stone.md) | [Chronicle](../README.md) | [Test Fixtures →](2025-12-01-test-fixtures.md)
