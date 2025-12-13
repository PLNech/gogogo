# Neurosymbolic Go AI Architecture

> "The hand must train the eye, and the eye must guide the hand."
> — Go proverb

## The Problem

Pure neural networks trained on expert games + self-play fail to generalize on tactical sequences:

```
Example: "Two-capture dead group" (Snapback)
- Black plays A, White must respond
- Black's stone captured, but now White has single eye
- Black plays B, captures entire group (20+ stones)

Network sees: "Some stones in atari" → Value: +0.1
Reality: Forced sequence → Value: +0.95
```

**Root cause**: Neural networks are pattern matchers, not calculators. They learn correlations, not causation. Tactical sequences require *reading* - verifying concrete move sequences.

## The Solution: Hybrid Architecture

### 1. Neural Component (Pattern Recognition)

What neural networks are good at:
- Strategic patterns (joseki, fuseki, shape)
- Influence and territory estimation
- Ownership prediction
- "Soft" positional judgment

Keep the existing KataGo-style network:
```
Input: 27 planes (stones, liberties, history, etc.)
Output:
  - Policy P(a|s): move probabilities
  - Value V(s): position evaluation
  - Ownership O(s): territory prediction
```

### 2. Symbolic Component (Tactical Verification)

What symbolic search is good at:
- **Ladders**: 100% deterministic, can verify in O(n)
- **Capture races**: Count liberties, verify outcomes
- **Snapbacks**: Detect throw-in patterns
- **Life/death**: Small-scale search for two eyes
- **Ko fights**: Track ko threats and outcomes

New module: `TacticalAnalyzer`
```python
class TacticalAnalyzer:
    """Symbolic tactical verification."""

    def detect_ladder(self, board, group) -> Optional[bool]:
        """Is this group in a working ladder? Returns True/False/None (unclear)"""

    def verify_capture_sequence(self, board, move, depth=4) -> float:
        """Search capture outcomes up to depth. Returns expected captures."""

    def detect_snapback(self, board, move) -> Optional[int]:
        """Does this move enable a snapback? Returns captured stones."""

    def evaluate_life_death(self, board, group, depth=6) -> float:
        """Is this group alive/dead/unsettled? Returns -1/+1/0."""
```

### 3. Integration: Hybrid MCTS

**A. Policy Blending**

```python
def compute_hybrid_prior(board, neural_policy, tactical_analyzer):
    """Combine neural policy with tactical boosts."""

    prior = neural_policy.copy()

    for move in legal_moves:
        # Boost moves that capture
        captures = tactical_analyzer.verify_capture_sequence(board, move)
        if captures > 0:
            prior[move] *= (1 + captures * 0.5)  # Scale by captured stones

        # Boost moves that save groups via ladder break
        if tactical_analyzer.is_ladder_breaker(board, move):
            prior[move] *= 2.0

        # Penalize moves that create dead groups
        if tactical_analyzer.creates_dead_group(board, move):
            prior[move] *= 0.01  # Near-zero

    return normalize(prior)
```

**B. Value Refinement**

```python
def refine_value(board, neural_value, tactical_analyzer, mcts_node):
    """Refine neural value with tactical search for critical positions."""

    # Check if position is "tactical" (has atari, low liberties, etc.)
    if not is_tactical_position(board):
        return neural_value  # Trust neural for calm positions

    # For tactical positions, do deeper verification
    tactical_value = tactical_analyzer.deep_search(board, depth=8)

    # Blend: trust tactics more when confident
    if abs(tactical_value) > 0.8:  # High confidence tactical result
        return 0.7 * tactical_value + 0.3 * neural_value
    else:
        return 0.5 * tactical_value + 0.5 * neural_value
```

**C. Selective Deepening**

```python
def should_deepen_search(mcts_node, tactical_analyzer):
    """Decide if we need more simulations on this node."""

    # High-value tactical positions deserve more search
    if tactical_analyzer.has_capture_opportunity(mcts_node.board):
        return True

    # Uncertain value + tactical activity = needs verification
    if mcts_node.value_variance > 0.3 and has_atari(mcts_node.board):
        return True

    return False
```

## Implementation Phases

### Phase 1: Tactical Feature Engineering (Quick Win)

Add explicit tactical features to neural network input:

```python
# New input planes (add to existing 27)
TACTICAL_PLANES = {
    'ladder_threatened': plane_28,      # Our groups in ladder danger
    'ladder_escape': plane_29,          # Moves that break ladders
    'snapback_opportunity': plane_30,   # Throw-in patterns
    'capture_in_1': plane_31,           # Immediate capture moves
    'capture_in_2': plane_32,           # 2-move capture sequences
    'group_life_status': plane_33,      # -1 dead, 0 unsettled, +1 alive
}
```

### Phase 2: TacticalAnalyzer Module

Port and enhance TypeScript heuristics to Python:

```python
# training/tactics.py

class TacticalAnalyzer:
    """Symbolic tactical analysis for Go positions."""

    def __init__(self, board_size: int = 19):
        self.board_size = board_size

    def detect_ladder(self, board: Board, start_group: List[Tuple[int,int]]) -> Optional[bool]:
        """
        Trace ladder to determine if group is captured.
        Returns True (captured), False (escapes), None (unclear).
        """
        # Exhaustive ladder tracing - deterministic!
        ...

    def verify_capture_sequence(
        self,
        board: Board,
        move: Tuple[int, int],
        depth: int = 4
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Alpha-beta search on capture sequences.
        Returns (expected_captures, principal_variation).
        """
        ...

    def detect_snapback(self, board: Board, move: Tuple[int, int]) -> int:
        """
        Check if move creates snapback opportunity.
        Returns number of stones captured by snapback (0 if none).
        """
        ...

    def life_death_search(
        self,
        board: Board,
        group: List[Tuple[int,int]],
        depth: int = 6
    ) -> float:
        """
        Minimax search for life/death status.
        Returns -1 (dead), +1 (alive), or intermediate value.
        """
        ...
```

### Phase 3: Hybrid MCTS Integration

```python
# training/hybrid_mcts.py

class HybridMCTS(MCTS):
    """MCTS with tactical verification."""

    def __init__(self, model, config, tactical_analyzer):
        super().__init__(model, config)
        self.tactics = tactical_analyzer

    def evaluate_leaf(self, board: Board) -> Tuple[np.ndarray, float]:
        """Hybrid evaluation: neural + tactical."""

        # Get neural evaluation
        neural_policy, neural_value = self._neural_eval(board)

        # Apply tactical adjustments
        policy = self._adjust_policy_tactically(board, neural_policy)
        value = self._adjust_value_tactically(board, neural_value)

        return policy, value

    def _adjust_policy_tactically(self, board, neural_policy):
        """Boost/penalize moves based on tactical analysis."""
        adjusted = neural_policy.copy()

        for move_idx in range(len(adjusted)):
            move = self._idx_to_move(move_idx)
            if move is None:  # pass
                continue

            # Check for immediate captures
            captures = self.tactics.count_captures(board, move)
            if captures > 0:
                adjusted[move_idx] *= (1 + captures * 0.3)

            # Check for ladder saves
            if self.tactics.saves_ladder(board, move):
                adjusted[move_idx] *= 2.0

            # Check for snapbacks
            snapback_captures = self.tactics.detect_snapback(board, move)
            if snapback_captures > 0:
                adjusted[move_idx] *= (1 + snapback_captures * 0.5)

            # Penalize creating dead groups
            if self.tactics.creates_dead_group(board, move):
                adjusted[move_idx] *= 0.01

        return adjusted / adjusted.sum()  # Re-normalize

    def _adjust_value_tactically(self, board, neural_value):
        """Refine value for tactical positions."""

        # Only refine if position is tactical
        if not self.tactics.is_tactical(board):
            return neural_value

        # Deep tactical search for uncertain positions
        tactical_value = self.tactics.deep_evaluate(board, depth=6)

        # Weight tactical result more when it's confident
        confidence = abs(tactical_value)
        if confidence > 0.7:
            return 0.6 * tactical_value + 0.4 * neural_value
        else:
            return 0.4 * tactical_value + 0.6 * neural_value
```

### Phase 4: Training Integration

**A. Use Tactical Features as Auxiliary Targets**

```python
def compute_tactical_targets(board, final_result):
    """Generate tactical supervision signals."""
    return {
        'ladder_groups': detect_all_ladders(board),
        'capture_moves': find_capture_moves(board),
        'life_death': evaluate_all_groups(board),
    }
```

**B. Curriculum: Tactical Positions First**

Already implemented! But enhance with:
- Filter for positions with verified tactical outcomes
- Weight positions by tactical complexity

**C. Self-Play with Tactical Verification**

```python
def generate_self_play_game(model, tactics):
    """Self-play with tactical consistency checking."""

    mcts = HybridMCTS(model, tactics)

    for move_num in range(max_moves):
        # Search with tactical enhancement
        policy = mcts.search(board)
        move = select_move(policy)

        # CRITICAL: Verify tactical moves match search result
        if tactics.is_tactical_move(board, move):
            verified_outcome = tactics.verify_sequence(board, move, depth=8)
            # Store verified outcome as training target
            training_data.append((board, policy, verified_outcome))
```

## Expected Benefits

| Aspect | Pure Neural | Neurosymbolic |
|--------|-------------|---------------|
| Ladder reading | ~60% accuracy | ~99% accuracy |
| Snapback detection | Often missed | Deterministic |
| Capture races | Value network guesses | Verified outcomes |
| Training data quality | Noisy self-play | Tactically verified |
| Simulations needed | 800+ for tactics | 200 + deep search |

## Key Insight

The neural network learns **what kinds of positions** are tactical.
The symbolic component verifies **specific outcomes**.

Together:
1. NN says "this looks like a ladder position" (pattern recognition)
2. Symbolic traces the ladder to verify (calculation)
3. Training signal is now **clean** - no noise from tactical mistakes

## Files to Create

```
training/
├── tactics.py          # TacticalAnalyzer class
├── hybrid_mcts.py      # HybridMCTS with tactical integration
├── tactical_features.py # New input planes for tactics
└── verify_training.py  # Script to verify tactical positions in dataset
```

## References

1. **AlphaGo (original)**: Used handcrafted features + neural + MCTS
2. **KataGo**: Ladder detection as input features (not just learned)
3. **SAI (Symbolic AI Improvement)**: Hybrid symbolic-neural for games
4. **Proof-Number Search**: For verified life/death analysis

---

*"The strong player plays with shape; the master plays with reading."*
