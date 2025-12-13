---
layout: post
title: "Neurosymbolic Harmony"
date: 2024-12-13
phase: "Synthesis"
excerpt: "Neural networks learn patterns. Symbolic systems calculate. Together: tactical precision."
---

> 和を以て貴しと為す
> *Harmony is to be valued*

---

The neural network learned to see.

But seeing is not calculating.

A master sees a ladder. Then reads it out, stone by stone.

The network saw patterns. It could not read.

---

## The Tactical Gap

| Tactic | Neural Accuracy | Required |
|--------|-----------------|----------|
| Simple capture | ~70% | 99% |
| Ladder | ~60% | exact |
| Snapback | ~40% | exact |
| Ko | ~50% | exact |

Patterns approximate. Tactics require proof.

---

## The Insight

Neural networks excel at *what kind of position*.

Symbolic systems excel at *what exactly happens*.

Why choose?

---

## Hybrid Architecture

```
┌─────────────────────────────────┐
│      Neural Network             │
│   (pattern recognition)         │
│                                 │
│   "This looks like a ladder"    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│    Tactical Analyzer            │
│   (symbolic verification)       │
│                                 │
│   "Let me check: B1, W2, B3..." │
│   "Yes, ladder works."          │
└─────────────────────────────────┘
```

The network proposes. The analyzer verifies.

---

## Snapback Detection

The classic problem:

```
  . B B B .
  B . X . B    ← White throws in at X
  . B B B .
```

White plays X. Captured immediately.

But after capture, Black has one liberty—at X.

White recaptures. *Snapback*.

---

The old algorithm checked everything. O(n²).

The new algorithm: **Atari-First**.

```python
def detect_snapback(board, move):
    # Only check groups already in atari
    for group in opponent_groups_in_atari():
        if only_liberty(group) == move:
            # Simulate capture
            if new_liberties(group) == 1:
                return True  # Snapback!
    return False
```

O(atari_groups). Usually zero or one.

---

## Ladder Tracing

Ladders are deterministic.

The defender runs diagonally. The attacker chases.

If a breaker stone exists on that diagonal: escape.

If edge reached: capture.

```python
def ladder_works(board, group):
    direction = escape_direction(group)
    for pos in diagonal_scan(group, direction):
        if board[pos] == defender:
            return False  # Broken
        if pos.is_edge():
            return True   # Captured
```

One diagonal scan. O(board_size).

No recursion needed.

---

## Results

Before tactical analyzer:

```
Capture:   2/3
Escape:    1/2
Ladder:    0/2
Snapback:  0/1
───────────────
Total:     5/10
```

After:

```
Capture:   3/3  ✓
Escape:    2/2  ✓
Ladder:    1/2  ↑
Snapback:  1/1  ✓
───────────────
Total:     8/10
```

**+60% relative improvement.**

---

## The Key Discovery

Multiplicative boosts fail when priors are wrong.

```
K10 (correct): 1.26% × 2.0 = 2.52%
M9 (wrong):    4.68% × 1.0 = 4.68%

M9 still wins.
```

**Additive boosts** overcome bad priors:

```
K10 (correct): 1.26% + 10% = 11.26%
M9 (wrong):    4.68% + 0%  = 4.68%

K10 wins.
```

```python
if boost > 1.0:
    policy[idx] += additive_weight * (boost - 1.0)
```

Simple. Effective.

---

## A Game Watched

```
Move 25: Capture 2 stones       boost ×12.0
Move 41: Snapback detected      boost ×161.7
Move 45: 7-stone capture        boost ×24.8

Final: Black +50.5
```

The hybrid plays with intention.

---

## What We Learned

Neural networks learn *intuition*.

Symbolic systems provide *certainty*.

Neither alone suffices.

Together: something approaching understanding.

---

> 切磋琢磨
> *Mutual polishing*

The neural and symbolic polish each other.

Pattern informs calculation. Calculation validates pattern.

The journey continues.

---

## What's Next

- Full integration with self-play training
- More tactical patterns (ko, seki, bent four)
- Score distribution heads for precise endgame

The machine does not yet see as masters see.

But it begins to glimpse.
