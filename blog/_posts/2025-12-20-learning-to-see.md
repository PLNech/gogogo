---
layout: post
title: "Learning to See"
date: 2025-12-20
phase: "Foundation"
excerpt: "Sensei's 8 Basic Instincts. Before we run, we crawl."
---

> 初心忘るべからず
> *Never forget the beginner's mind.*

---

We built 1449 tests today.

Not clever tests. Not deep reading. Not life-and-death.

[Sensei's 8 Basic Instincts](/glossary#basic-instincts) — patterns masters play without thinking.

![The 8 Basic Instincts](/images/eight-basic-instincts.png)

---

## The Eight

| # | Instinct | Japanese | When |
|---|----------|----------|------|
| 1 | [Extend from Atari](/glossary#extend-from-atari) | アタリから伸びよ | Stone in atari → extend |
| 2 | [Hane vs Tsuke](/glossary#hane-vs-tsuke) | ツケにはハネ | Opponent attaches → hane |
| 3 | [Hane at Head of Two](/glossary#hane-at-head-of-two) | 二子の頭にハネ | Two stones in row → play above |
| 4 | [Stretch from Kosumi](/glossary#stretch-from-kosumi) | コスミから伸びよ | Diagonal attach → stretch away |
| 5 | [Block the Angle](/glossary#block-the-angle) | カケにはオサエ | Angle attack → block |
| 6 | [Connect vs Peep](/glossary#connect-vs-peep) | ノゾキにはツギ | Opponent peeps → connect |
| 7 | [Block the Thrust](/glossary#block-the-thrust) | ツキアタリには | Opponent thrusts → block |
| 8 | [Stretch from Bump](/glossary#stretch-from-bump) | ブツカリから伸びよ | Supported attach → stretch |

---

A child learns them in their first lessons.

Our model scored **1.3%**.

![Benchmark Results](/images/learning-to-see-benchmark.png)

Zero on extend. Zero on hane. Zero on connect.

Worse than random.

Worse than a child.

---

There's a French song. *Le Déconservatoire* by [Les Voleurs de Swing](https://fr.wikipedia.org/wiki/Les_Voleurs_de_swing).

The conservatory teaches rules. Scales. Theory. Discipline.

You lose freedom. You drill structure into bone.

It feels like death.

But only those who master the rules earn the right to break them.

Django learned technique. Then he forgot it. Then he played.

---

We wanted to teach our model *joseki*. *Fuseki*. The grand patterns.

Hubris.

It cannot see a stone in atari.

---

So we go back.

Eight instincts. Five board sizes. 1449 positions.

Corners. Edges. Center. Black to play. White to play.

The humblest curriculum.

---

Great artists steal.

But first, they copy.

Stroke by stroke. Stone by stone.

---

The beauty isn't in the rules.

It's in what comes after.

When the hand moves before thought.

When structure becomes style.

When we earn the right to unlearn.

---

But not yet.

First, we learn to see.

---

```
INSTINCT BENCHMARK RESULTS
============================================================
  extend_atari    :  0.0% (0/44)
  hane_tsuke      :  0.0% (0/50)
  hane_head_two   :  5.6% (2/36)
  stretch_kosumi  :  0.0% (0/48)
  block_angle     :  5.6% (2/36)
  connect_peep    :  0.0% (0/50)
  block_thrust    :  0.0% (0/24)
  stretch_bump    :  0.0% (0/27)
  OVERALL         :  1.3% (4/315)
============================================================
```

This is where we start.

---

> 下手の考え休むに似たり
> *A weak player's thinking resembles resting.*

We are weak. We admit it.

Now we work.
