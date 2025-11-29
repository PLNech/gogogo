# First Stone

**November 28, 2025**

---

> 空即是色
> *Emptiness is form*

---

Before the first move, there is only potential.
The board lies empty—not barren, but pregnant with possibility.
Each intersection awaits its purpose.

This is not a project about defeating humans at Go.
Nor about building the strongest possible engine.
It is about understanding—making visible what was hidden.

The primary goal: **become stronger**.
The method: build a teacher that plays, that challenges, that reveals.

## Why Go?

Go is deceptively simple. Two colors. One rule: surround to capture.
Yet from this simplicity emerges complexity that humbles.
It took decades for computers to play at amateur dan level.

Go teaches patience. It teaches the value of influence over immediate gain.
It teaches that sometimes the best move is not the aggressive one.
These lessons apply beyond the board.

## The Journey

We begin with chaos: a codebase that grew without clear direction.
Tests failing. Imports missing. Five different AI implementations,
none quite working, all fighting for dominance.

The path forward is not addition, but subtraction.
Not piling features, but revealing essence.
Not moving fast, but moving deliberately.

> "In Go, the first move is not the most important.
> The most important move is the one that makes the position clear."

## The Architecture

We build on three pillars:

1. **Board Representation** - Seeing the game as machines see it
2. **Heuristic Policy** - Encoding Go intuition as testable code
3. **Tree Search (MCTS)** - Reading ahead through possibilities

Each pillar supports the next. Each can be tested independently.
Each will be documented with working code.

```typescript
// The board in its simplest form
export interface Board {
  size: number
  stones: Stone[][]  // null | 'black' | 'white'
}

// Empty board: infinite potential
export function createBoard(size: number): Board {
  const stones: Stone[][] = []
  for (let i = 0; i < size; i++) {
    stones[i] = []
    for (let j = 0; j < size; j++) {
      stones[i]![j] = null
    }
  }
  return { size, stones }
}
```

From this foundation, everything grows.

## The Method

Every feature will be built test-first.
Every deliverable will be accompanied by a blog post.
Every post will contain working code and demonstration.

This is not just documentation—it is proof.
Proof that the feature exists.
Proof that it works.
Proof that we understand what we built.

## What's Next

**Phase 0: Foundation**

- Fix the broken tests
- Remove what doesn't serve
- Establish test infrastructure
- Create fixtures for standard positions

Only then can we build.
First: clean the ground.
Then: lay the foundation.
Finally: raise the structure.

---

> 碁は対話
> *Go is conversation*

This project is a conversation between human and machine,
between intuition and analysis,
between the ancient game and modern techniques.

The board awaits.
Let us begin.

---

[← Back to Chronicle](../README.md)
