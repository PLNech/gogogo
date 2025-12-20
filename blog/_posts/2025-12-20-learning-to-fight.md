---
layout: post
title: "Learning to Fight"
date: 2025-12-20
phase: "Synthesis"
excerpt: "Evolving instinct priorities through 2000 Atari Go games. All eight proverbs confirmed."
---

> 戦いを学ぶ — *Learn fighting*

The eight instincts exist. We encoded them. But which matters *most*?

Sensei's Library assigns weights. Intuition assigns weights. Go proverbs weight "extend from atari" as critical.

But proverbs are human wisdom. What does the game itself say?

---

## The Experiment

Atari Go. First capture wins. Pure tactics. No territory to muddy the signal.

We play thousands of games. Each move, we detect which instincts fire. We track whether following them leads to victory.

```
For each game:
  For each move:
    detected = detect_instincts(position)
    actual_move = model_plays()
    followed = actual_move in detected.moves

  After game:
    For each instinct event:
      if followed and won: record_win
      if followed and lost: record_loss
      if ignored and won: record_win_without
      if ignored and lost: record_loss_without
```

The advantage is simple: `win_rate(followed) - win_rate(ignored)`.

Positive = following helps.
Negative = following hurts.

---

## Results (2000 games)

| Instinct | Advantage | Fired | Verdict |
|----------|-----------|-------|---------|
| hane_vs_tsuke | **+13.2%** | 54,397 | 🏆 Champion |
| extend_from_atari | +9.7% | 19,337 | ✅ Confirmed |
| block_the_thrust | +9.6% | 59,893 | ✅ Confirmed |
| block_the_angle | +3.5% | 64,273 | ✅ Works |
| connect_vs_peep | +3.4% | 65,163 | ✅ "Even a moron" |
| stretch_from_bump | +3.2% | 52,845 | ✅ Slight positive |
| stretch_from_kosumi | +3.0% | 79,665 | ✅ Slight positive |
| hane_at_head_of_two | +1.9% | 45,171 | ✅ Strategic |

---

## Key Findings

### The Hane Champion

`hane_vs_tsuke` at **+13.2%** is the clear champion. When opponent attaches, hane creates cutting points. In Atari Go, cuts kill.

### All Eight Proverbs Confirmed

With correct pattern detection and 2000 games, **all eight instincts show positive advantage**. The proverbs are right.

### The Tactical Triad

Three instincts dominate in Atari Go's tactical environment:

1. **hane_vs_tsuke** (+13.2%) — Create cutting points
2. **extend_from_atari** (+9.7%) — Escape capture
3. **block_the_thrust** (+9.6%) — Prevent invasions

These are the survival instincts. In a game where first capture wins, survival is everything.

### Correct Detection Matters

Earlier runs showed `hane_at_head_of_two` as slightly negative (-0.9%). But our pattern detector was wrong — it triggered on *any* two opponent stones, not the true 2v2 confrontation pattern from Sensei's Library.

With correct detection (requiring our two stones parallel to their two), the instinct is positive (+1.9%). The proverb was never wrong — our code was.

One experiment is a question, not an answer. But correct questions lead to correct answers.

---

## Weight Evolution

```python
advantage = follow_winrate - ignore_winrate
adjustment = advantage * (1 + (0.5 - follow_rate))
weight = weight + lr * adjustment
```

Final weights stayed close to initial (Sensei's wisdom was good!):
- `hane_vs_tsuke`: 1.50 → **1.55** (earned its boost)
- `block_the_thrust`: 2.00 → **2.02** (confirmed)
- Others: minimal change (±0.03)

---

## Implications

1. **Sensei was right.** All eight proverbs confirmed by data.

2. **Humility in measurement.** When data contradicts wisdom, question the experiment first.

3. **Correct detection matters.** The pattern must match the proverb exactly. 2v2 is not "any two stones."

4. **Hane is king** (in fights). When contact happens, wrap around. Don't hesitate.

---

> 碁は戦なり — *Go is war*

In Atari Go, we learn to fight before we learn to live.
