---
layout: post
title: "The Adaptive Curriculum"
date: 2025-12-24
phase: "Foundation"
excerpt: "Teaching fundamentals. The weight decays as mastery grows."
---

> 急がば回れ
> *If in a hurry, take the roundabout path.*

---

We had a problem.

The model couldn't see captures. Zero percent accuracy on the most basic instinct.

Self-play with random initialization creates a vicious cycle:
- Random model never captures
- Captures never appear in training data
- Model never learns to capture

The network learns from patterns it sees.

If it never sees a capture, it never learns one.

---

The solution is simple. Ancient, even.

**Teach fundamentals first.**

But how do you teach a neural network?

---

We wire the instincts directly into the loss:

$$L = L_{policy} + L_{value} + \lambda(t) \cdot L_{instinct}$$

The instinct loss penalizes missed fundamentals: captures, escapes, connections.

But here's the key insight:

$$\lambda(t) = \lambda_0 \times (1 - accuracy)$$

The weight **decays as mastery grows**.

---

High lambda early: "Learn to see a stone in atari."

Low lambda late: "You've graduated. Play freely."

The model transitions from student to player.

Naturally. Adaptively.

---

```python
# During training step
instinct_loss, _ = curriculum.compute_loss(boards, log_policy)

# The weight adapts to benchmark accuracy
curriculum.update_lambda(current_accuracy)

# Total loss includes adaptive instinct term
loss = policy_loss + value_loss + instinct_loss
```

---

![Training Curves](/assets/images/curriculum_training.png)

The results speak.

**Before**: 0% capture, 5.7% overall.

**After 5 epochs**: 100% capture, 100% overall.

Lambda fell from 1.4 to 0.05.

The curriculum worked itself out of a job.

---

The four quadrants tell the story:

1. **Loss curves** — Instinct loss dominates early, policy takes over
2. **Lambda decay** — Weight adapts as mastery grows
3. **Instinct accuracy** — Captures: 0% → 100%
4. **Training accuracy** — Model learns the positions

---

This is curriculum learning made adaptive.

No manual scheduling. No epoch thresholds.

The model teaches itself when it's ready to move on.

---

> 習うより慣れよ
> *Practice makes perfect* — literally "get used to it rather than learn it"

We don't teach rules.

We show examples until the rules become instinct.

---

Next: wire this into self-play.

Fundamentals learned here.

Strategy learned there.

The best of both worlds.

---

*Code: `training/instinct_loss.py`, `training/train_curriculum.py`*
