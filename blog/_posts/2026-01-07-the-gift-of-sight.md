---
layout: post
title: "The Gift of Sight"
date: 2026-01-07
phase: "Synthesis"
excerpt: "Building a dashboard to see what our AI sees. Observability as meditation."
---

> 見ることは信じること — *To see is to believe*

---

We trained in darkness.

Numbers scrolling. Loss decreasing. Games generating.

But we could not *see*.

---

## The Blindness

Our model played thousands of games. We had metrics:
- 17.6% benchmark accuracy
- Captures: 0%
- Ladders: 20%

Numbers. Abstractions. Shadows on the cave wall.

What did the games *look* like? Where did it struggle? When did it hesitate?

We didn't know. We couldn't know.

---

## The Dashboard

So we built eyes.

![Game Analysis Dashboard](/gogogo/images/dashboard-game.png)

*Select a move to see details* — the slider becomes a time machine.

Six charts. One game. Every heartbeat visible:

**Victory Probability** — the black-white split, breath of the game. Who's winning? The answer flows like ink.

**Live Groups** — the armies on the field. Rising, falling, merging, dying.

**Score Estimate** — the territory balance, positive for black, negative for white.

**Captures** — the casualties. Each step upward, a tactical victory.

**Minimum Liberties** — danger. Red X marks where groups gasp for air.

**Policy Entropy** — uncertainty. High early (many choices), low late (one path remains).

---

## Zoom In, Zoom Out

The slider is the key.

Move 23: a capture. The liberties chart spikes down.

Move 47: ko begins. The entropy jumps — suddenly many moves matter.

Move 72: the tide turns. Victory probability crosses 50%.

Tactics live in single moves. Strategy emerges across the arc.

This is Go's eternal tension: the local and the global. The stone and the whole board.

Now we can see both.

---

## Thirty-Six Views

Hokusai painted Mount Fuji thirty-six times. Same mountain. Different angles, seasons, distances.

Each view revealed something the others could not.

![Game Comparison](/gogogo/images/dashboard-compare.png)

Our **Compare** view overlays games like Hokusai's prints. Two trajectories. Same board size. Different outcomes.

Where do they diverge? Move 35 — one game's black surges ahead, the other collapses.

*Comparison is a virtue in anything we strive to create.*

Without it, we see only what is. With it, we see what could have been.

---

## 観察は智慧の始まり

*Observation is the beginning of wisdom.*

The dashboard footer speaks truth. We cannot improve what we cannot see.

Now we see:
- Which games are close, which are blowouts
- When the model hesitates (high entropy)
- Where tactical failures happen (liberties = 1)
- How capture momentum shifts the score

---

## The Stack

For the curious:

```
FastAPI     — the spine
HTMX        — the nerves (no JS framework needed)
Tailwind    — the skin (warm paper aesthetic)
Plotly.js   — the eyes (interactive charts)
```

Lean. Deployable. Blog-compatible.

One command: `poetry run uvicorn dashboard.app:app --reload`

---

## What Remains

The dashboard shows metrics. But not the board itself.

We see the *vital signs* of the game. Not the *body*.

Next: render the board at each move. Show territory shading. Animate the stones appearing.

Let us watch our AI play, move by move, like reviewing a master's game.

---

> 千里の道も一歩から — *A journey of a thousand miles begins with a single step*

We took many steps today.

We gave ourselves the gift of sight.

Now the real learning begins.
