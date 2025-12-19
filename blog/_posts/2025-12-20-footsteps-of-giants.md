---
layout: post
title: "Footsteps of Giants"
date: 2025-12-20
phase: "Research"
excerpt: "On the papers that light our path, the doors we chose not to open, and the questions that remain."
---

> 学びて思わざれば則ち罔し — *Learning without thinking is labor lost*

We have a model. It plays. Badly, but it plays.

Now we ask: can it learn to play on any board?

![Research Evolution](/gogogo/images/research-evolution.png)

---

## The Pioneers

In 2016, [Silver et al.](https://www.nature.com/articles/nature16961) at DeepMind did the impossible. AlphaGo defeated Lee Sedol. The ancient game fell to neural networks and Monte Carlo tree search.

A year later, [AlphaZero](https://arxiv.org/abs/1712.01815) learned from nothing. No human games. Pure self-play. Tabula rasa to superhuman in hours.

We do not stand on their shoulders. We step in their footsteps, following paths they carved through wilderness.

---

## The Pathfinder

Then came [David Wu](https://arxiv.org/abs/1902.10565) with KataGo.

Where AlphaZero required thousands of TPUs, KataGo asked: *what if we were clever instead of rich?*

His [methods document](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) is a gift to the field. Open. Detailed. Reproducible.

One finding stopped us mid-scroll:

> "KataGo trains on sizes from 7x7 to 19x19 simultaneously... it appears to extrapolate quite effectively to larger boards, playing at a high level on board sizes in the 20s and 30s with no additional training."

A single network. All sizes. Extrapolation beyond training.

And this:

> "Mixed-size training is not particularly expensive. At least in early training up to human pro level, KataGo seems to learn at least equally fast and perhaps slightly faster than training on 19x19 alone."

Faster. Not slower. *Faster.*

The small boards provide quick feedback. Knowledge transfers upward. The network learns general rules for how size affects play.

But Wu didn't stop there. The [methods document](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) reads like a cookbook of clever ideas:

- **Shaped Dirichlet noise**: Don't explore uniformly. Concentrate noise on moves the policy already likes. Find the blind spots.
- **Policy surprise weighting**: When MCTS discovers moves the policy missed, train harder on those positions.
- **Auxiliary soft policy**: A second head predicting a softened distribution. Forces the network to distinguish between "good" and "slightly less good."

Each trick: 5-20 Elo. Combined: 40-90 Elo. The compound interest of careful engineering.

---

## The New Wave

At NeurIPS 2024, [Rigaux and Cazenave](https://arxiv.org/abs/2410.23753) asked a different question.

CNNs see grids. But Go is a graph. Intersections connected by lines. Pieces moving along edges between nodes.

Their architecture, AlphaGateau, treats the board as topology rather than image.

Results:

| Architecture | Parameters | Learning Speed |
|--------------|------------|----------------|
| CNN (AlphaZero-style) | 2.2M | Baseline |
| GNN (AlphaGateau) | 1.0M | 10× faster |

Fewer parameters. Faster learning. And this:

> "The model, when trained on a smaller 5×5 variant of chess, is able to be quickly fine-tuned to play on regular 8×8 chess."

The graph doesn't care about grid dimensions.

---

## Doors We Closed

Research is as much about what doesn't work.

**Padding and cropping**: Take a 9×9 model, pad the input to 19×19, crop the output. Simple. Obvious.

Wrong.

Go requires whole-board vision. A 9×9 window sees 22% of a 19×19 game. You cannot understand a moyo through a keyhole. No serious work uses this approach.

We close this door.

**Coordinate embeddings alone**: Add explicit (row, col) features. Let the network know where it stands.

But KataGo succeeds without them. Convolutions learn position implicitly from the edges. The network knows where the walls are.

This door remains ajar. Perhaps useful. Not essential.

---

## Doors That Open

**Multi-scale training**: KataGo's path. Train on all sizes at once. Let small boards teach patience. Let large boards teach vision. Knowledge flows between scales.

This door is wide open. Others have walked through. We will follow.

**Curriculum learning**: Start small. Graduate to larger. [Recent work](https://www.researchgate.net/publication/381884459_Multi-Agent_Training_for_Pommerman_Curriculum_Learning_and_Population-based_Self-Play_Approach) shows self-play itself provides curriculum—"a perfect ordering of difficulty."

But structured progression may accelerate further. 9×9 until competent. Then 13×13. Then the full board.

**Graph neural networks**: The elegant path. Board size becomes irrelevant. Topology is topology.

But the tooling is young. Browser inference uncertain. This door leads somewhere beautiful, but the path is less traveled.

We note its location. Perhaps later.

---

## Questions That Remain

Can we measure transfer versus fresh training in a small experiment? One day of compute. Two models. Does 9×9 pre-training help 19×19?

Is self-play sufficient for all sizes, or do we need professional games at each scale?

GNN versus CNN: is the implementation cost worth the generalization? For us, with our single GPU, our limited time?

These questions have no answers yet. Only experiments.

---

## What We Know Now

| Approach | Verdict | Source |
|----------|---------|--------|
| Padding/cropping | Skip | No SOTA uses it |
| Multi-scale training | Proven | KataGo |
| Transfer/curriculum | Proven | AlphaGateau |
| Graph networks | Emerging | NeurIPS 2024 |
| Coordinate embeddings | Maybe | Works without |

---

## The Work Ahead

We have logged these paths in our plan. Multi-scale training first—the proven road. Curriculum learning as acceleration. GNN as a separate experiment, when time permits.

The model plays on 9×9. Soon, perhaps, it will play on any size.

Or it won't. And we will learn why.

Either way, we follow the footsteps. Silver, Schrittwieser, Huang. Wu. Rigaux, Cazenave. Names in papers. Minds that opened doors.

Late at night, the questions multiply. The GPU hums. Someone reads, someone thinks, and slowly, something like understanding emerges.

> 道は近きに在りて、遠きに求む — *The Way is near, yet we seek it afar*

The papers are read. The doors are mapped. The path forward is clearer than before.

Now we walk it.

---

**Sources:**
- [Mastering the Game of Go with Deep Neural Networks and Tree Search](https://www.nature.com/articles/nature16961) — Silver et al., 2016
- [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815) — Silver et al., 2017
- [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565) — Wu, 2019
- [KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) — Wu, ongoing
- [Enhancing Chess RL with Graph Representation](https://arxiv.org/abs/2410.23753) — Rigaux & Cazenave, NeurIPS 2024
- [Curriculum Learning for RL Domains](https://jmlr.org/papers/volume21/20-212/20-212.pdf) — Narvekar et al., 2020
