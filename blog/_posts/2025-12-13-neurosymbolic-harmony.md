---
layout: post
title: "Neurosymbolic Harmony"
date: 2025-12-13
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
| [Ladder]({{ '/glossary#ladder' | relative_url }}) | ~60% | exact |
| [Snapback]({{ '/glossary#snapback' | relative_url }}) | ~40% | exact |
| [Ko]({{ '/glossary#ko' | relative_url }}) | ~50% | exact |

Patterns approximate. Tactics require proof.

---

## The Insight

Neural networks excel at *what kind of position*.

Symbolic systems excel at *what exactly happens*.

Why choose?

---

## Hybrid Architecture

![Hybrid Architecture]({{ '/images/hybrid-architecture.png' | relative_url }})

The network proposes. The analyzer verifies.

---

## Snapback Detection

The classic problem:

{% include goboard.html id="snapback" size=5 black="bb,cb,db,bc,dc,bd,cd,dd" mark="cc" caption="White throws in at the marked point. What happens?" %}

White plays center. Captured immediately.

But after capture, Black has one liberty—at center.

White recaptures. *Snapback*.

---

The old algorithm checked everything. [O(n²)]({{ '/glossary#big-o-notation' | relative_url }}).

The new algorithm: **[Atari]({{ '/glossary#atari' | relative_url }})-First**.

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

![Ladder Pattern]({{ '/images/ladder-pattern.png' | relative_url }})

If a breaker stone exists on that diagonal: escape. If edge reached: capture.

One diagonal scan. O(board_size). No recursion needed.

---

## Results

![Tactical Results]({{ '/images/tactical-results.png' | relative_url }})

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

{% include goboard.html id="showcase" size=9 sgf="(;GM[1]FF[4]CA[UTF-8]SZ[9]PB[GoGoGo]PW[GoGoGo]RE[W+3.5];B[cb];W[dc];B[ed];W[eb];B[dd];W[fc];B[fe];W[ec];B[ee];W[ce];B[gd];W[db];B[df];W[aa];B[fa];W[ba];B[gb];W[hc];B[bb];W[ea];B[de];W[dg];B[hd];W[bg];B[cg];W[he];B[fb];W[cd];B[ab];W[ge];B[eg];W[ei];B[ch];W[gc];B[ff];W[ia];B[ca];W[aa];B[cc];W[ic])" caption="Watch a game from our hybrid neural-symbolic engine. Use the controls below to step through." %}

White wins by 3.5 points.

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
