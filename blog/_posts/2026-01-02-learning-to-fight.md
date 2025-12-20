---
layout: post
title: "Learning to Fight"
date: 2026-01-02
phase: "Synthesis"
excerpt: "What if the proverbs are wrong? Evolving instinct priorities through Atari Go."
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
| hane_vs_tsuke | **+12.1%** | 75,201 | 🏆 Champion |
| extend_from_atari | +5.2% | 36,985 | ✅ Confirmed |
| block_the_thrust | +3.7% | 38,171 | ✅ Confirmed |
| block_the_angle | +3.5% | 76,785 | ✅ Works |
| connect_vs_peep | +2.7% | 54,683 | ✅ "Even a moron" |
| stretch_from_bump | +2.1% | 39,619 | ✅ Slight positive |
| stretch_from_kosumi | +2.0% | 76,397 | ✅ Slight positive |
| hane_at_head_of_two | **-0.9%** | 68,468 | ⚠️ Only negative |

---

## Key Findings

### The Hane Champion

`hane_vs_tsuke` at **+12.1%** is 4× more impactful than the next instinct. When opponent attaches, hane creates cutting points. In Atari Go, cuts kill.

### All Proverbs (Mostly) Confirmed

Early results with 50 games showed noise. With 2000 games, seven of eight instincts show positive advantage. The proverbs are right.

### The Humble Exception

`hane_at_head_of_two` is the only negative (-0.9%).

But wait—is the proverb wrong, or is our test wrong?

Hane at the head of two is *strategic*. It pressures groups over many moves. It builds attack. It creates weak points that matter 50 moves later.

In Atari Go? First capture wins. There is no "50 moves later."

**The instinct isn't wrong. Our battlefield is too small.**

This teaches humility. Tactical Go (Atari) and strategic Go (full game) reward different wisdom. Some proverbs need space to breathe.

---

## Weight Evolution

```python
advantage = follow_winrate - ignore_winrate
adjustment = advantage * (1 + (0.5 - follow_rate))
weight = weight + lr * adjustment
```

Final weights stayed close to initial (Sensei's wisdom was good!):
- `hane_vs_tsuke`: 1.50 → **1.56** (earned its boost)
- Others: minimal change (±0.03)

---

## Implications

1. **Sensei was right.** Seven of eight proverbs confirmed by data.

2. **Humility in measurement.** When data contradicts wisdom, question the experiment first.

3. **Tactics ≠ Strategy.** Atari Go tests tactical instincts. Full Go rewards patience.

4. **Hane is king** (in fights). When contact happens, wrap around. Don't hesitate.

---

> 碁は戦なり — *Go is war*

In Atari Go, we learn to fight before we learn to live.
