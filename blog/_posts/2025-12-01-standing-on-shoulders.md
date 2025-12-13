---
layout: post
title: "Standing on Shoulders"
date: 2025-12-01
phase: "Research"
excerpt: "KataGo's secrets. Ownership heads. The 50x speedup that changed everything."
---

> 温故知新
> *Study the old to understand the new*

---

We read the KataGo paper.

Then we read it again.

The answer was there all along.

---

## The Key Insight

> "With only a final binary result, the neural net can only guess at what aspect of the board position caused the loss. By contrast, with an **ownership target**, the neural net receives direct feedback on which area of the board was mispredicted."
>
> — KataGo Paper §4.1

Not one bit per game.

*361 bits per position*.

Every intersection tells the network: you were wrong here.

---

## The Speedup

| Technique | Speedup |
|-----------|---------|
| Playout Cap Randomization | 1.37× |
| Forced Playouts + Pruning | 1.25× |
| Global Pooling | 1.60× |
| Opponent Move Prediction | 1.30× |
| **Ownership + Score Targets** | **1.65×** |

Combined: ~50× faster than AlphaGo Zero.

Compute democratized.

---

## Ownership: The Breakthrough

![Ownership Map]({{ '/images/ownership-map.png' | relative_url }})

Every stone. Every territory. Every dame point.

The network learns *where* it was wrong, not just *that* it was wrong.

```python
def ownership_loss(predicted, target):
    # 361 predictions per position
    # Each one a lesson
    return bce(predicted, target).sum() / 361
```

The gradient flows precisely where understanding was lacking.

---

## What We Implemented

**Ownership Head**
```python
class OwnershipHead(nn.Module):
    def forward(self, x):
        return self.conv(x)  # (batch, 1, 19, 19)
```

One convolution. Tanh output. Per-point prediction.

Simple. Powerful.

**Opponent Move Prediction**

Force the network to model the opponent.

*What would they do here?*

Understanding the opponent is understanding the game.

**Curriculum Learning**

Tactical positions first. Captures. Atari. Life and death.

Learn the urgent before the subtle.

---

## The New Architecture

```
Input:  27 planes × 19 × 19

Backbone:
  6 ResBlocks × 128 filters
  Global Pooling at layers 3, 6

Heads:
  Policy → 362 (board + pass)
  Value → win probability
  Ownership → 361 predictions ← NEW
  Opponent → 362 predictions ← NEW
```

2.2 million parameters.

Fits comfortably on one GPU.

---

## Training Curves

![Training curves with ownership]({{ '/images/training_curves.png' | relative_url }})

Policy loss: descending.

Value loss: descending.

Ownership loss: *actually learning*.

The sparse feature problem—solved.

---

## First Results

![Accuracy Comparison]({{ '/images/accuracy-comparison.png' | relative_url }})

The network sees again.

---

> 水は方円の器に随う
> *Water takes the shape of its container*

Better containers. Better water.

The architecture shaped the learning.

Now: can we push further?
