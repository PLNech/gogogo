# SNAP.md - Snapback & Ladder Detection Research

## Problem Statement

Current TacticalAnalyzer fails on:
- **Snapback detection**: 0/1 tests passing
- **Ladder continuation**: 1/2 tests passing

Goal: Efficient symbolic detection for single-GPU training where NN evaluation is the bottleneck, not tactical search.

---

## Part 1: Snapback Analysis

### What IS a Snapback?

```
Sequence:
1. White throws in (suicide)     →  W captured immediately
2. Black captures the stone      →  Black group now has 1 liberty
3. White recaptures Black group  →  SNAPBACK complete
```

Key insight: **The throw-in point becomes Black's only liberty after capture.**

### Thesis: Forward Simulation

Current approach:
```python
for each legal move:
    simulate placement
    if captured:
        check opponent libs after capture
        if libs == 1: SNAPBACK
```

**Problems:**
- O(n²) per position (check all moves, simulate each)
- Expensive group/liberty recalculation
- Most moves aren't snapbacks - wasted computation

### Antithesis: Pattern Matching

Alternative: recognize snapback SHAPES
```
Classic patterns:
  B B      B B B      B . B
  B . ←    B . B ←    B B B
  B B      B B B        ↑
```

**Problems:**
- Finite patterns miss novel positions
- Rotation/reflection variants multiply patterns
- Doesn't generalize to arbitrary shapes

### Synthesis: Liberty Graph Analysis

**Key observation**: Snapback is a GRAPH PROPERTY, not a move property.

```
Definition: Move M is a snapback iff:
1. M is the ONLY liberty of opponent group G
2. After M is played and captured, G has exactly 1 liberty (at M)
```

This means:
- Only check moves that ARE a group's only liberty (atari!)
- Only groups already in atari can be snapback targets
- Dramatically reduces search space

### Proposed Algorithm: Atari-First Snapback

```python
def find_snapbacks(board, player):
    snapbacks = []

    # Only check opponent groups in ATARI
    for group in opponent_groups_in_atari(board):
        liberty = get_single_liberty(group)  # Only 1 liberty

        # Would our stone there be captured?
        if would_be_captured(board, liberty, player):
            # After capture, check if group has 1 lib
            if libs_after_capture(board, group, liberty) == 1:
                snapbacks.append(liberty)

    return snapbacks
```

**Complexity**: O(groups_in_atari) << O(all_moves)

---

## Part 2: Ladder Analysis

### Current Implementation Issues

`trace_ladder()` does full recursive simulation:
- Expensive for long ladders
- May timeout or hit depth limit
- Doesn't cache results

### Thesis: Full Simulation

```python
def trace_ladder(board, group, attacker):
    # Recursive: extend, chase, extend, chase...
    # Until: escape (3+ libs) or capture (0 libs)
```

**Problem**: O(board_size) depth × O(branching) width

### Antithesis: Ladder Breaker Scan

Key insight: **Ladders work unless a "ladder breaker" exists.**

```
Ladder direction is DETERMINISTIC:
- Defender runs diagonally
- If friendly stone exists on diagonal → ladder broken
```

```python
def ladder_works(board, group, attacker):
    direction = compute_ladder_direction(group)

    # Scan diagonal for breakers
    for pos in diagonal_path(group, direction):
        if board[pos] == defender_color:
            return False  # Ladder broken
        if pos.is_edge():
            return True   # Captured at edge

    return True  # No breaker found
```

**Complexity**: O(board_size) - single diagonal scan!

### Synthesis: Hybrid with Caching

1. **Quick check**: Scan diagonal for breakers
2. **If unclear**: Fall back to simulation
3. **Cache**: Store result by Zobrist hash

```python
def trace_ladder_fast(board, group, attacker):
    cache_key = board.zobrist_hash()
    if cache_key in ladder_cache:
        return ladder_cache[cache_key]

    # Quick diagonal scan
    result = diagonal_ladder_check(board, group)

    if result is None:  # Unclear
        result = trace_ladder_recursive(board, group, attacker)

    ladder_cache[cache_key] = result
    return result
```

---

## Part 3: Implementation Plan

### Phase 1: Snapback (Priority: HIGH)

**Step 1.1**: Implement `opponent_groups_in_atari()`
- Iterate board once, find all groups with 1 liberty
- Return only opponent's groups

**Step 1.2**: Implement `libs_after_capture()`
- Simulate capture without full board copy
- Count liberties of capturing group

**Step 1.3**: Update `detect_snapback()`
- Use atari-first algorithm
- Only check atari groups, not all moves

**Step 1.4**: Add unit tests
- Test various snapback shapes
- Test false positives (looks like snapback but isn't)

### Phase 2: Ladder Optimization (Priority: MEDIUM)

**Step 2.1**: Implement `compute_ladder_direction()`
- Given group in atari, determine escape direction
- Direction is perpendicular to attacker stones

**Step 2.2**: Implement `diagonal_ladder_check()`
- Scan diagonal from group position
- Return True (works), False (broken), None (unclear)

**Step 2.3**: Add Zobrist caching
- Cache ladder results per position
- Clear cache on board change

**Step 2.4**: Benchmark
- Compare speed: full simulation vs diagonal scan
- Measure cache hit rate

### Phase 3: Integration (Priority: LOW)

**Step 3.1**: Update TacticalAnalyzer
- Use fast snapback detection
- Use fast ladder detection

**Step 3.2**: Update HybridAgent
- Higher boost for detected snapbacks
- Higher boost for working ladder moves

**Step 3.3**: Re-run tactical tests
- Target: 9/10 or better
- Snapback: 1/1
- Ladders: 2/2

---

## Part 4: Performance Considerations

### Single-GPU Constraint

NN inference is the bottleneck (~10ms per batch).
Tactical search must be << 1ms to not add latency.

**Targets:**
- Snapback detection: < 0.1ms per position
- Ladder check: < 0.5ms per position
- Total tactical overhead: < 1ms

### Memory

- Ladder cache: ~100KB (10K entries × 10 bytes)
- Group liberty tracking: O(board_size²) = ~1KB for 19×19

### Parallelization

Tactical analysis is CPU-bound, NN is GPU-bound.
Can run in parallel:
```
Thread 1 (GPU): NN batch inference
Thread 2 (CPU): Tactical analysis for next batch
```

---

## Part 5: Research Questions

### Open Questions

1. **Snapback generalization**: Can we detect "delayed snapbacks" (capture in 2+ moves)?

2. **Ladder races**: When both players have ladders, who wins?

3. **Net vs Ladder**: Nets (geta) are similar to ladders. Same detection?

4. **Learning tactical features**: Can we train NN to output "snapback probability" as auxiliary head?

### Future Directions

1. **MCTS integration**: Use tactical knowledge for move ordering in MCTS

2. **Proof-number search**: For life/death problems, use PNS with tactical pruning

3. **Endgame solver**: Exact endgame solving with tactical shortcuts

---

## References

- [Erta, "Computer Go Algorithms"](https://webdocs.cs.ualberta.ca/~games/go/) - Liberty counting
- [Müller, "Computer Go"](https://www.springer.com/gp/book/9780387001630) - Tactical search
- [Silver et al., "AlphaGo"](https://www.nature.com/articles/nature16961) - MCTS + NN integration
- [KataGo](https://github.com/lightvector/KataGo) - Ownership prediction for tactical positions
