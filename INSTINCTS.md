# Sensei's 8 Basic Instincts

> Reference document for Go instinct patterns.
> Source: [senseis.xmp.net/BasicInstinct](https://senseis.xmp.net/?BasicInstinct)

**Used by:**
- `training/sensei_instincts.py` (Python detector)
- `src/ui/instincts/InstinctBattleground.tsx` (React visualization)
- AI training curriculum (instinct loss)

**Legend for patterns:**
- `B` = Black stone (us)
- `W` = White stone (opponent)
- `b` = Black stone (opponent, when we play White)
- `w` = White stone (us, when we play White)
- `.` = Empty intersection
- `*` = The instinct move
- `○` = Marked stone (support/trigger)

---

## 1. Extend from Atari (アタリから伸びよ)

**Trigger**: Your stone is in atari (1 liberty)
**Response**: Extend to gain liberties

```
. . . . .
. W W . .
. W B W .    ← Black in atari (surrounded on 3 sides)
. . * . .    ← Black EXTENDS to escape
. . . . .
```

**Why**: Extending creates 2+ liberties, escaping the threat.

**Atari Go Results**: +9.7% follow advantage, 19,337 times fired

---

## 2. Hane vs Tsuke (ツケにはハネ)

**Trigger**: Opponent attaches (tsuke) to your stone
**Response**: Wrap around with hane (diagonal)

```
. . . . .
. . B . .    ← Your stone
. . W . .    ← White ATTACHES directly
. . * . .    ← You play HANE (can be diagonal too)
. . . . .
```

Alternative diagonal hane:
```
. . . . .
. . B . .    ← Your stone
. . B W .    ← White attaches from side
. . . * .    ← Diagonal HANE wrapping around
. . . . .
```

**Why**: Hane gains influence and puts pressure on the attaching stone.

**Atari Go Results**: +13.2% follow advantage (CHAMPION!), 54,397 times fired

---

## 3. Hane at Head of Two (二子の頭にハネ)

**Trigger**: Two opposing groups of two stones facing each other
**Response**: Play at the HEAD of opponent's two stones

```
. . . . .
. . * . .    ← Play at HEAD of White's two
. B W . .    ← Black's stone faces White's first
. B W . .    ← Black's stone faces White's second
. . . . .
```

From Sensei's Library diagram:
```
    . x .
    ● ○ .    ← x marks the head, where Black plays
    ● ○ .    ← Two Black (●) face two White (○)
    . . .
```

**Why**: Playing at the head:
1. Wraps your two stones into strong influence
2. Pushes opponent down with limited expansion
3. Creates cutting points and weaknesses for opponent

**Key insight**: This is a 2v2 confrontation pattern, NOT about isolated stones!

**Atari Go Results**: +1.9% follow advantage (Strategic), 45,171 times fired

---

## 4. Stretch from Kosumi (コスミから伸びよ)

**Trigger**: Opponent plays diagonal contact (kosumi-tsuke)
**Response**: Stretch away from the contact

```
. . . . .
. . W . .    ← Opponent's diagonal contact
. . . B .    ← Your stone (diagonal to W)
. . . * .    ← STRETCH away
. . . . .
```

**Why**: Stretching maintains connection and creates good shape.

**Atari Go Results**: +3.0% follow advantage, 79,665 times fired

---

## 5. Block the Angle (カケにはオサエ)

**Trigger**: Opponent approaches with knight's move (keima/angle play)
**Response**: Block diagonally

```
. . . . .
. . . B .    ← Your stone
. . * . .    ← BLOCK diagonally
. W . . .    ← Opponent's knight's move approach
. . . . .
```

From Sensei's screenshots (diagonal block):
```
. . . . .
. . . ● .    ← Your stone (Black)
. . ① . .    ← White plays block (numbered = move order)
. ○ . . .    ← After opponent's approach
. . . . .
```

**Why**: Strengthens your stone and weakens opponent's approach. Also blocks opponent's path into your area.

**Atari Go Results**: +3.5% follow advantage, 64,273 times fired

---

## 6. Connect vs Peep (ノゾキにはツギ)

**Trigger**: Opponent peeps at your cutting point
**Response**: Connect immediately ("Even a moron connects against a peep")

```
. . . . .
. B . B .    ← Your two stones with a gap
. . * . .    ← CONNECT the gap
. . W . .    ← Opponent PEEPS at cutting point
. . . . .
```

From Sensei's screenshot:
```
. . . . .
. ● ② ● .    ← Two Black stones, Black connects at ②
. . ① . .    ← White peeps at ①
. . . . .
```

**Why**: If you don't connect, opponent cuts and you lose.

**Atari Go Results**: +3.4% follow advantage, 65,163 times fired

---

## 7. Block the Thrust (ツキアタリには)

**Trigger**: Opponent thrusts into your formation
**Response**: Block the thrust

```
. . . . .
. . ② . .    ← Black BLOCKS at ②
. B ① . .    ← White THRUSTS at ① into formation
. B . . .    ← Your stones in column
. . . . .
```

From Sensei's screenshot:
```
. . . . .
. ● ② . .    ← Black ● with block at ②
. ● ① . .    ← White thrusts at ①
. . ○ . .    ← White's supporting stone
. . . . .
```

**Why**:
- If opponent cuts after the block, they get cut in return
- Forces opponent to choose which side to cut

**Atari Go Results**: +9.6% follow advantage, 59,893 times fired

---

## 8. Stretch from Bump (ブツカリから伸びよ)

**Trigger**: Opponent bumps against your stone, AND they have support
**Response**: Stretch away (don't hane)

```
. . . . .
. . ② . .    ← Black STRETCHES at ②
. ○ ① . .    ← White support (○), Black bumps at ①
. . W . .    ← White's supporting stone
. . . . .
```

From Sensei's screenshot:
```
. . . . .
. . ② . .    ← Black stretches
○ ① ● . .    ← Circle = White support, ① = bump, ● = Black
. . . . .
```

**Why**: Normally attachments are answered with hane. But when opponent's attachment is strengthened by a supporting stone, hane leads to crosscut trouble. Stretch maintains stability.

**Atari Go Results**: +3.2% follow advantage, 52,845 times fired

---

## Summary Table

| # | Instinct | Japanese | Advantage | Times Fired | Verdict |
|---|---|---|---|---|---|
| 1 | Hane vs Tsuke | ツケにはハネ | **+13.2%** | 54,397 | Champion |
| 2 | Extend from Atari | アタリから伸びよ | +9.7% | 19,337 | Confirmed |
| 3 | Block the Thrust | ツキアタリには | +9.6% | 59,893 | Confirmed |
| 4 | Block the Angle | カケにはオサエ | +3.5% | 64,273 | Works |
| 5 | Connect vs Peep | ノゾキにはツギ | +3.4% | 65,163 | Even a moron |
| 6 | Stretch from Bump | ブツカリから伸びよ | +3.2% | 52,845 | Slight positive |
| 7 | Stretch from Kosumi | コスミから伸びよ | +3.0% | 79,665 | Slight positive |
| 8 | Hane at Head of Two | 二子の頭にハネ | +1.9% | 45,171 | Strategic |

---

## Pattern Detection Notes

For pattern matching in code, detect:

1. **Atari situations**: Count liberties, if 1 → extend_from_atari
2. **Attachment (tsuke)**: Opponent places adjacent to our stone → hane_vs_tsuke
3. **2v2 parallel**: Two groups of 2 stones parallel → hane_at_head_of_two
4. **Diagonal contact**: Opponent diagonal to us → stretch_from_kosumi
5. **Knight's move approach**: Opponent at (±1, ±2) or (±2, ±1) → block_the_angle
6. **Peep at cutting point**: Gap between our stones, opponent adjacent to gap → connect_vs_peep
7. **Thrust into formation**: Opponent plays between our stones → block_the_thrust
8. **Bump with support**: Opponent contacts AND has adjacent support stone → stretch_from_bump

---

## Philosophical Note

> "One experiment is a question, not an answer."

With correct pattern detection (2v2 confrontation, not just "any two stones"), all eight instincts show positive advantage in Atari Go. The proverbs were right — our detection was wrong.

We reproduce. Vary board sizes. Let empiric tell if wisdom was right — but with time.
