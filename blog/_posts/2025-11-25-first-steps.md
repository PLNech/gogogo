---
layout: post
title: "First Steps"
date: 2025-11-25
phase: "Foundation"
excerpt: "The naive approach. Simple heuristics. A teacher that barely crawls."
---

> 初心忘るべからず
> *Never forget beginner's mind*

---

The first AI was born.

It did not think. It reacted.

Random moves. Capture when possible. Avoid filling eyes.

```typescript
function simpleAI(board: Board): Move {
  const captures = findCaptureMoves(board);
  if (captures.length > 0) {
    return random(captures);
  }

  const safe = findSafeMoves(board);
  return random(safe);
}
```

Primitive. Beautiful. A first breath.

---

## What It Could Do

- Place stones legally
- Capture when an opponent group had one liberty
- Avoid obvious self-atari
- Pass when no good moves remained

---

## What It Could Not Do

Everything else.

No sense of territory. No concept of influence.

No understanding that the corner is stronger than the center.

It played like a drunk stumbling through a garden, occasionally stepping on flowers.

---

## MCTS Arrives

[Monte Carlo Tree Search]({{ '/glossary#mcts' | relative_url }}).

The algorithm that changed computer Go.

![MCTS Tree]({{ '/images/mcts-tree.png' | relative_url }})

Each simulation a tiny universe. Each backpropagation a lesson learned.

The tree grows. The AI begins to *search*.

---

## The UCB Formula

$$UCB = \frac{w_i}{n_i} + c \sqrt{\frac{\ln N}{n_i}}$$

The [Upper Confidence Bound]({{ '/glossary#ucb' | relative_url }}). Exploitation vs exploration.

The eternal balance.

Win rate pulls toward known goods. The exploration term whispers: *what haven't you tried?*

---

## First Games

The MCTS AI beat the random AI.

A small victory. A first step on a long path.

But when we watched it play, something was wrong.

It built walls. Strange, useless walls.

![Wall Problem]({{ '/images/wall-problem.png' | relative_url }})

Stones placed in straight lines. Territory hemorrhaging everywhere.

The heuristics were broken. The evaluations blind.

---

> 石の下を見よ
> *Look beneath the stones*

The surface showed progress. Beneath: foundations made of sand.

Tomorrow we dig deeper.
