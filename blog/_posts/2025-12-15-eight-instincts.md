---
layout: post
title: "Eight Instincts"
date: 2025-12-15
phase: "Wisdom"
excerpt: "Test first, then implement. Define truth before seeking it. The 8 basic instincts meet TDD."
---

> 先に知る者は後に行く
> *First know, then act*

---

## The Ancient Pattern

Masters don't think. They see.

![Sensei's 8 Basic Instincts](/gogogo/images/eight-basic-instincts.png)

Thousands of games distilled into instinct. When a stone touches yours, the response is automatic. When two stones align, the hane appears.

These patterns have names. Eight fundamental responses, passed through generations.

The [8 Basic Instincts](https://senseis.xmp.net/?BasicInstinct).

---

## The Modern Method

How do you teach a machine to see?

First, define what seeing means.

```python
def test_extend_from_atari(self, analyzer, board):
    """Stone in atari should extend."""
    # Black stone with one liberty
    board.board[c, c] = 1
    board.board[c-1, c] = -1
    board.board[c+1, c] = -1
    board.board[c, c-1] = -1
    # Liberty at (c, c+1)

    result = analyzer.detect_extend_from_atari(
        board, (c, c+1)
    )
    assert result is True
```

The test defines truth. The implementation seeks it.

---

## Test-Driven Development

Write the test first.

Watch it fail.

Write the code.

Watch it pass.

---

This is not just methodology. This is epistemology.

The test is a **gradient toward truth**. Each assertion narrows the space of correct implementations. Each failure illuminates the gap between intent and reality.

> 碁の手は無限なり、答えは一つ
> *Moves are infinite. Truth is one.*

---

## The Eight

| # | Instinct | Japanese | Boost |
|---|----------|----------|-------|
| 1 | Extend from atari | アタリから伸びよ | 3.0× |
| 2 | Hane against tsuke | ツケにはハネ | 1.5× |
| 3 | Hane at head of two | 二子の頭にハネ | 2.0× |
| 4 | Stretch from kosumi | コスミから伸びよ | 1.4× |
| 5 | Block the angle | カケにはオサエ | 1.5× |
| 6 | Connect against peep | ノゾキにはツギ | 2.5× |
| 7 | Block the thrust | ツキアタリには | 1.8× |
| 8 | Stretch from bump | ブツカリから伸びよ | 1.3× |

---

## Implementation

Each instinct becomes a detector:

```python
def detect_connect_against_peep(
    self, board: Board, move: Tuple[int, int]
) -> bool:
    """Detect: connect against a peep.

    Even a moron connects against a peep.
    — Go proverb
    """
    peeps = self.find_peeps(board)
    for _, cutting_point in peeps:
        if move == cutting_point:
            return True
    return False
```

Simple. Direct. Testable.

---

## The Gradient

Fourteen tests. Each one a question.

```
test_extend_from_atari ............... PASSED
test_hane_response_to_tsuke .......... PASSED
test_hane_at_head_of_two ............. PASSED
test_connect_against_peep ............ PASSED
...
======================== 14 passed in 0.11s ====
```

Each green check: the implementation aligns with intent.

Each failure would have been a gift—a precise pointer to where understanding diverged from truth.

---

## Philosophy of Testing

The Weiqi masters speak of the **perfect eye**.

Not just seeing the board. Seeing through it. Understanding before calculation.

TDD offers a path to the perfect eye for code:

1. **Define** what correct means (write test)
2. **Observe** the gap (watch it fail)
3. **Close** the gap (write implementation)
4. **Verify** alignment (watch it pass)

The test is not overhead. The test is the compass.

---

## Integration

The instincts flow into the tactical analyzer:

```python
def get_tactical_boost(self, board, move):
    boost = 1.0

    # Captures, ladders, snapbacks...
    # ...

    # Apply 8 Basic Instincts
    instinct_boost = self._get_instinct_boost(
        board, move
    )
    if instinct_boost > 1.0:
        boost *= instinct_boost

    return boost
```

Ancient wisdom meets modern search.

---

## What We Built

32 tests. All passing.

Eight instincts encoded. Centuries of Go knowledge made executable.

The neural network proposes moves. The symbolic system evaluates. The instincts guide both toward the path masters walk without thinking.

---

## The Lesson

Test first, then implement.

Define truth before seeking it.

The gradient guides. The test illuminates. The code follows.

> 道は近きにあり
> *The way is near*

It always was. We just needed to see it.

---

## What's Next

- Self-play training with instinct boosts
- Ownership prediction (KataGo's key insight)
- The game emerges from the patterns

The machine does not yet have instinct.

But it begins to recognize the shapes that instinct names.
