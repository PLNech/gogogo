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

## Early Results (50 games)

| Instinct | Follow Advantage | Surprise? |
|----------|-----------------|-----------|
| hane_vs_tsuke | **+11.4%** | Huge! |
| block_the_thrust | **+7.3%** | Confirms proverb |
| connect_vs_peep | +4.1% | "Even a moron connects" |
| block_the_angle | +1.9% | Slight positive |
| stretch_from_kosumi | +0.8% | Nearly neutral |
| extend_from_atari | -2.0% | Wait, what? |
| hane_at_head_of_two | -2.1% | Hmm |
| stretch_from_bump | **-10.3%** | Proverb is wrong? |

---

## Surprising Findings

### The Hane Effect

`hane_vs_tsuke` dominates. When opponent attaches, wrapping around with hane creates cutting points. In tactical Atari Go, those cuts kill.

### "Extend from Atari" - Overrated?

In full Go, extending saves your stones. But in Atari Go, if you're in atari, you're already losing the tactical race. Better to counterattack than defend.

### "Stretch from Bump" - Just Wrong?

The proverb says: when opponent bumps with support, stretch away. But our data shows -10% win rate. Perhaps in Atari Go, contact fighting trumps running?

---

## The Weights Evolve

We update weights based on advantage:

```python
advantage = follow_winrate - ignore_winrate
adjustment = advantage * (1 + (0.5 - follow_rate))
weight = weight + lr * adjustment
```

Instincts that help but are ignored get bigger boosts. The model learns what matters.

---

## Implications

1. **Context matters.** Atari Go weights differ from full Go weights.

2. **Proverbs are heuristics.** They encode human wisdom, but wisdom has limits.

3. **Self-play reveals truth.** Let the game teach the game.

4. **Transfer learning opportunity.** Train on Atari Go, transfer tactics to full Go.

---

## Next Steps

Running 2000+ games to get statistically significant results. Then:

- Compare Atari Go weights vs full Go weights
- Use learned weights in instinct curriculum
- See if tactical accuracy improves

The proverbs point the way. The data walks the path.

---

> 碁は戦なり — *Go is war*

In Atari Go, we learn to fight before we learn to live.
