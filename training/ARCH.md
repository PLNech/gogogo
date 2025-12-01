# Neural Network Architecture Documentation

> Every decision grounded in research. No cargo-culting.

---

## KataGo Key Insight (50x Speedup over AlphaZero)

> "With only a final binary result, the neural net can only guess at what aspect of the board position caused the loss. By contrast, with an ownership target, the neural net receives direct feedback on which area of the board was mispredicted, with large errors and gradients localized to the mispredicted area."
> — [KataGo Paper §4.1](https://arxiv.org/abs/1902.10565)

**Speedup breakdown** (from ablation studies, Table 2):

| Technique | Speedup Factor |
|-----------|----------------|
| Playout Cap Randomization | 1.37× |
| Forced Playouts + Policy Target Pruning | 1.25× |
| Global Pooling | 1.60× |
| Auxiliary Policy Targets (opponent move) | 1.30× |
| Auxiliary Ownership + Score Targets | **1.65×** |
| Game-specific Features | 1.55× |
| **Combined** | **~9× (conservative)** |

---

## Network Architecture

### Base Architecture: Pre-activation ResNet

| Component | Choice | Rationale | Source |
|-----------|--------|-----------|--------|
| **Backbone** | Pre-activation ResNet | BN→ReLU→Conv→BN→ReLU→Conv. Better gradient flow. | [He et al. 2016](https://arxiv.org/abs/1603.05027) |
| **Dual heads** | Policy + Value (+ Ownership) | Shared features, multi-task learning | [AlphaGo Zero](https://www.nature.com/articles/nature24270) |
| **Global pooling** | Mean + ScaledMean + Max | Captures non-local patterns (ko, ladders, winning/losing adjustments) | [KataGo §3.3](https://arxiv.org/abs/1902.10565) |

### KataGo Progressive Sizing

| Size | Params | When to Switch | Strength |
|------|--------|----------------|----------|
| 6×96 | ~1M | Start | Strong amateur |
| 10×128 | ~2.5M | 0.75 days | Strong professional |
| 15×192 | ~8M | 1.75 days | Superhuman |
| 20×256 | ~24M | 7.5 days | Superhuman+ |

**Our choice**: 6×128 (~2.2M params) - between KataGo's first two sizes.

### Current Configuration

```
Input:  27 planes × 19 × 19  (tactical features)
        or 17 planes × 19 × 19  (basic features)

Backbone:
  - Initial Conv: 5×5 (KataGo) or 3×3 (ours), input → 128 filters
  - 6 ResBlocks × 128 filters (pre-activation)
    - Blocks 3, 6: + GlobalPoolingBias
  - Total: ~2.2M parameters

Policy Head:
  - Conv 1×1 → 32 channels (P)
  - Conv 1×1 → 32 channels (G) → GlobalPoolingBias to P
  - BN → ReLU → Conv 1×1 → 2 channels (policy + opponent_policy)
  - Pass move: FC from pooled G → 2 values

Value Head:
  - Conv 1×1 → 32 channels (V)
  - GlobalPool(V) → 96 values
  - FC → 64 → FC → outputs (win/loss/draw, score_mean, score_std, etc.)

Ownership Head (NEW - HIGH PRIORITY):
  - Conv 1×1 → 1 channel
  - Tanh → [-1, +1] per point
```

### Global Pooling Bias Structure (KataGo §3.3, Appendix A)

```python
class GlobalPoolingBias(nn.Module):
    """
    KataGo's key innovation for non-local patterns.

    Given tensors X (b×b×cX) and G (b×b×cG):
    1. BN + ReLU on G
    2. GlobalPool(G) → 3*cG values (mean, scaled_mean, max)
    3. FC → cX biases
    4. Add biases channelwise to X
    """
    def __init__(self, c_g, c_x, board_size=19):
        super().__init__()
        self.bn = nn.BatchNorm2d(c_g)
        self.fc = nn.Linear(3 * c_g, c_x)
        self.b_avg = 14.0  # (9+19)/2

    def global_pool(self, g):
        # g: (batch, c_g, h, w)
        b = g.shape[-1]  # board width
        mean = g.mean(dim=(2, 3))                           # (batch, c_g)
        scaled_mean = mean * (b - self.b_avg) / 10.0        # board-size scaling
        max_pool = g.amax(dim=(2, 3))                       # (batch, c_g)
        return torch.cat([mean, scaled_mean, max_pool], dim=1)  # (batch, 3*c_g)

    def forward(self, x, g):
        g = F.relu(self.bn(g))
        pooled = self.global_pool(g)           # (batch, 3*c_g)
        biases = self.fc(pooled)               # (batch, c_x)
        return x + biases.unsqueeze(-1).unsqueeze(-1)  # broadcast to spatial dims
```

**Why it helps**: "In a wide variety of strategy games, strong players, when winning, alter their local move preferences to favor 'simple' options, whereas when losing they seek 'complication'. Global pooling allows convolutional nets to internally condition on such global context." (§3.3)

---

## Input Features

### KataGo Input Features (18 spatial + 10 global)

**Spatial (b×b×18):**

| # | Feature | Notes |
|---|---------|-------|
| 1 | Location is on board | For variable board sizes |
| 2 | Own/opponent stones | Standard |
| 3 | Stones with 1/2/3 liberties | **Liberty features** |
| 1 | Ko-illegal move | Superko handling |
| 5 | Last 5 move locations (one-hot) | History |
| 3 | Ladderable stones (0/1/2 turns ago) | **Ladder detection** |
| 1 | Moving here catches ladder | **Ladder tactics** |
| 2 | Pass-alive areas (own/opponent) | Life/death |

**Global (10 values):**

| # | Feature |
|---|---------|
| 5 | Which of last 5 moves were pass |
| 1 | Komi / 15.0 |
| 2 | Ko rules (simple/positional/situational) |
| 1 | Suicide allowed |
| 1 | Komi + board size parity |

### Our Features (27 planes)

| Plane | Description | Status |
|-------|-------------|--------|
| 0-1 | Own/opponent stones | ✓ |
| 2-16 | Move history (15 planes) | ✓ |
| 17-19 | Own groups: 1/2/3+ liberties | SPARSE |
| 20-22 | Opponent groups: 1/2/3+ liberties | SPARSE |
| 23 | Capture moves available | VERY SPARSE |
| 24 | Self-atari moves | VERY SPARSE |
| 25 | Eye-like points | SPARSE |
| 26 | Edge distance | DENSE |

**Problem**: Planes 17-25 are sparse → 21% accuracy vs 40% with basic features.

**Solutions** (in priority order):
1. Ownership auxiliary target (361× more signal)
2. Opponent move auxiliary target
3. Curriculum learning (tactical positions first)

---

## Loss Functions (KataGo Appendix B)

### Full KataGo Loss Function

```python
def katago_loss(outputs, targets):
    """
    KataGo's complete loss function.
    All weights from paper Appendix B.
    """
    loss = 0.0

    # 1. GAME OUTCOME (main value target)
    # Cross-entropy on {win, loss, no_result}
    # Weight: c_value = 1.5
    loss += 1.5 * F.cross_entropy(outputs['game_outcome'], targets['result'])

    # 2. POLICY (main policy target)
    # Cross-entropy on move distribution
    # Weight: 1.0
    loss += F.cross_entropy(outputs['policy'], targets['move'])

    # 3. OPPONENT POLICY (auxiliary - regularization)
    # Predicts opponent's next move
    # Weight: w_opp = 0.15
    loss += 0.15 * F.cross_entropy(outputs['opponent_policy'], targets['opponent_move'])

    # 4. OWNERSHIP (auxiliary - CRITICAL for credit assignment)
    # Per-point ownership in [-1, +1]
    # Weight: w_o = 1.5 / b² (scales with board size)
    # Cross-entropy treating as classification
    b = 19
    w_ownership = 1.5 / (b * b)
    ownership_target_binary = (targets['ownership'] + 1) / 2  # [-1,1] → [0,1]
    loss += w_ownership * F.binary_cross_entropy_with_logits(
        outputs['ownership'], ownership_target_binary, reduction='sum'
    )

    # 5. SCORE BELIEF PDF (auxiliary)
    # One-hot encoding of final score
    # Weight: w_spdf = 0.02
    loss += 0.02 * F.cross_entropy(outputs['score_pdf'], targets['final_score_bucket'])

    # 6. SCORE BELIEF CDF (auxiliary)
    # Cumulative distribution - pushes mass near correct score
    # Weight: w_scdf = 0.02
    pred_cdf = outputs['score_pdf'].cumsum(dim=1)
    target_cdf = targets['score_cdf']
    loss += 0.02 * ((pred_cdf - target_cdf) ** 2).sum(dim=1).mean()

    # 7. SCORE MEAN SELF-PREDICTION (regularization)
    # μ̂_s should match mean of score_pdf
    # Weight: w_sbreg = 0.004
    score_values = torch.arange(-400, 401)  # possible scores
    implied_mean = (F.softmax(outputs['score_pdf'], dim=1) * score_values).sum(dim=1)
    loss += 0.004 * F.huber_loss(outputs['score_mean'], implied_mean, delta=10.0)

    # 8. L2 REGULARIZATION
    # Weight: c_L2 = 3e-5
    # (applied via optimizer weight_decay)

    return loss
```

### Simplified Version for Supervised Pre-training

```python
def supervised_loss_with_ownership(outputs, targets, board_size=19):
    """
    Practical loss for supervised pre-training.
    Adds ownership to existing policy + value losses.
    """
    # Policy loss (main)
    policy_loss = F.cross_entropy(outputs['policy'], targets['move'])

    # Value loss (main)
    value_loss = 1.5 * F.mse_loss(outputs['value'], targets['outcome'])

    # Ownership loss (auxiliary - CRITICAL)
    # targets['ownership']: (batch, 19, 19) in [-1, +1]
    # outputs['ownership']: (batch, 1, 19, 19) logits
    w_ownership = 1.5 / (board_size ** 2)
    ownership_target = (targets['ownership'] + 1) / 2  # → [0, 1]
    ownership_loss = w_ownership * F.binary_cross_entropy_with_logits(
        outputs['ownership'].squeeze(1),
        ownership_target,
        reduction='sum'
    ) / targets['ownership'].shape[0]  # mean over batch

    # Opponent move loss (auxiliary - regularization)
    # Weight: 0.15
    opponent_loss = 0.15 * F.cross_entropy(
        outputs['opponent_policy'],
        targets['opponent_move']
    )

    return policy_loss + value_loss + ownership_loss + opponent_loss
```

### Loss Weight Summary

| Loss | Weight | Why |
|------|--------|-----|
| Policy | 1.0 | Main target |
| Value/Outcome | 1.5 | Noisy, needs higher weight |
| **Ownership** | **1.5/b²** | **Credit assignment** |
| Opponent policy | 0.15 | Regularization only |
| Score PDF | 0.02 | Fine-grained value |
| Score CDF | 0.02 | Pushes mass correctly |
| Score mean reg | 0.004 | Self-consistency |
| L2 | 3e-5 | Weight decay |

---

## Auxiliary Targets Implementation

### 1. Ownership Head (HIGHEST PRIORITY)

```python
class OwnershipHead(nn.Module):
    """
    Predicts final ownership of each board point.
    +1 = current player owns, -1 = opponent owns, 0 = neutral/dame

    "The neural net receives direct feedback on which area
    of the board was mispredicted" (KataGo §4.1)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # Returns logits, apply tanh for [-1, 1] prediction
        return self.conv(x)  # (batch, 1, 19, 19)


def compute_ownership_target(final_board_state):
    """
    Compute ownership from final game position.
    Uses Tromp-Taylor scoring (area scoring).

    Returns: (19, 19) tensor with values in {-1, 0, +1}
    """
    # final_board_state: numpy array where
    # 1 = black stone, -1 = white stone, 0 = empty

    ownership = np.zeros_like(final_board_state)

    # Stones are owned by their color
    ownership[final_board_state == 1] = 1   # Black
    ownership[final_board_state == -1] = -1  # White

    # Empty points: flood-fill to determine territory
    # A point is territory if ALL paths to edge go through one color
    for i in range(19):
        for j in range(19):
            if final_board_state[i, j] == 0:
                owner = flood_fill_territory(final_board_state, i, j)
                ownership[i, j] = owner  # 1, -1, or 0 (neutral/seki)

    return ownership


def flood_fill_territory(board, start_i, start_j):
    """
    Determine territory ownership via flood fill.
    Returns: 1 (black), -1 (white), or 0 (neutral)
    """
    visited = set()
    queue = [(start_i, start_j)]
    touches_black = False
    touches_white = False

    while queue:
        i, j = queue.pop(0)
        if (i, j) in visited:
            continue
        visited.add((i, j))

        if board[i, j] == 1:
            touches_black = True
            continue
        elif board[i, j] == -1:
            touches_white = True
            continue

        # Empty - explore neighbors
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < 19 and 0 <= nj < 19:
                queue.append((ni, nj))

    if touches_black and not touches_white:
        return 1  # Black territory
    elif touches_white and not touches_black:
        return -1  # White territory
    else:
        return 0  # Neutral (dame or seki)
```

### 2. Opponent Move Prediction (Easy Win)

```python
class DualPolicyHead(nn.Module):
    """
    Outputs both current player's policy AND opponent's next move.
    The opponent prediction is purely for regularization.

    "KataGo is the first to apply it to the AlphaZero process...
    produces a modest but clear benefit" (KataGo §3.4)
    """
    def __init__(self, in_channels, board_size=19):
        super().__init__()
        self.conv_p = nn.Conv2d(in_channels, 32, 1)
        self.conv_g = nn.Conv2d(in_channels, 32, 1)
        self.global_pool_bias = GlobalPoolingBias(32, 32)
        self.bn = nn.BatchNorm2d(32)
        # 2 channels: policy + opponent_policy
        self.final_conv = nn.Conv2d(32, 2, 1)
        # Pass move outputs
        self.pass_fc = nn.Linear(96, 2)  # 32*3 from global pool
        self.board_size = board_size

    def forward(self, x):
        p = self.conv_p(x)
        g = self.conv_g(x)
        p = self.global_pool_bias(p, g)
        p = F.relu(self.bn(p))
        spatial = self.final_conv(p)  # (batch, 2, 19, 19)

        # Flatten spatial policies
        batch = spatial.shape[0]
        policy = spatial[:, 0].view(batch, -1)      # (batch, 361)
        opp_policy = spatial[:, 1].view(batch, -1)  # (batch, 361)

        # Pass move from global pool
        pooled = self.global_pool_bias.global_pool(g)
        pass_logits = self.pass_fc(pooled)  # (batch, 2)

        # Concatenate: 361 board + 1 pass = 362
        policy = torch.cat([policy, pass_logits[:, 0:1]], dim=1)
        opp_policy = torch.cat([opp_policy, pass_logits[:, 1:2]], dim=1)

        return policy, opp_policy
```

### 3. Score Distribution (Optional, Lower Priority)

```python
class ScoreHead(nn.Module):
    """
    Predicts distribution over final score differences.
    Useful for:
    - More nuanced value estimation
    - Score-maximizing play (not just win/loss)

    Weight in loss: 0.02 (small, auxiliary)
    """
    def __init__(self, in_channels, max_score=400):
        super().__init__()
        self.max_score = max_score
        n_buckets = 2 * max_score + 1  # -400 to +400

        # Shared processing
        self.pool = GlobalPoolingBias(in_channels, in_channels)
        self.fc1 = nn.Linear(3 * in_channels, 256)

        # Score distribution (PDF)
        self.fc_pdf = nn.Linear(256 + 2, 64)  # +2 for score value, parity
        self.fc_pdf_out = nn.Linear(64, 1)

        # Score mean/std direct predictions
        self.fc_stats = nn.Linear(256, 2)  # mean, log_std

    def forward(self, x):
        pooled = self.pool.global_pool(x)
        h = F.relu(self.fc1(pooled))

        # Direct mean/std prediction
        stats = self.fc_stats(h)
        score_mean = stats[:, 0] * 20  # scale
        score_std = F.softplus(stats[:, 1]) * 20

        # Could also compute full PDF if needed
        # (omitted for efficiency in supervised training)

        return score_mean, score_std
```

---

## Training Techniques

### Playout Cap Randomization (Self-Play Only)

```python
"""
KataGo §3.1: Relieves tension between policy and value training.

Policy needs many playouts (800+) to improve beyond prior.
Value needs many games (fewer playouts per game) for data.

Solution:
- 25% of turns: full search (600-1000 nodes) → record for training
- 75% of turns: fast search (100-200 nodes) → just play, don't record
"""

class PlayoutCapRandomizer:
    def __init__(self, p_full=0.25, n_full=600, n_fast=100):
        self.p_full = p_full
        self.n_full = n_full
        self.n_fast = n_fast

    def get_playout_cap(self):
        if random.random() < self.p_full:
            return self.n_full, True  # (cap, record_for_training)
        return self.n_fast, False
```

### Forced Playouts + Policy Target Pruning (Self-Play Only)

```python
"""
KataGo §3.2: Decouples exploration from policy target.

Problem: Dirichlet noise forces playouts on bad moves,
         but we don't want to train policy to predict those.

Solution:
1. Force minimum playouts on noised moves: n_forced = sqrt(k * P(c) * total_N)
2. After search, PRUNE: subtract forced playouts from non-best moves
3. This removes exploration artifacts from training target
"""

def compute_policy_target(root_visits, forced_visits, best_child):
    """
    Prune forced playouts from policy target.

    Args:
        root_visits: dict of child -> visit count
        forced_visits: dict of child -> forced visit count
        best_child: child with most visits
    """
    target = {}
    best_puct = compute_puct(best_child, root_visits)

    for child, visits in root_visits.items():
        if child == best_child:
            target[child] = visits
        else:
            # Subtract forced playouts, but not below PUCT threshold
            pruned = visits
            while pruned > 1 and compute_puct(child, pruned) >= best_puct:
                pruned -= 1
            pruned = max(1, visits - forced_visits.get(child, 0))

            # Outright prune single-playout children
            if pruned <= 1:
                continue
            target[child] = pruned

    # Normalize to distribution
    total = sum(target.values())
    return {c: v/total for c, v in target.items()}
```

### Curriculum Learning (Supervised Pre-training)

```python
"""
Our addition: Train on tactical positions first.

Schedule:
- Epochs 1-3: 100% tactical positions (has atari or capture)
- Epochs 4-6: 70% tactical, 30% normal
- Epochs 7+: 50% tactical, 50% normal
"""

class CurriculumSampler(torch.utils.data.Sampler):
    def __init__(self, tactical_indices, normal_indices, tactical_ratio=1.0):
        self.tactical_indices = tactical_indices
        self.normal_indices = normal_indices
        self.tactical_ratio = tactical_ratio

    def set_epoch(self, epoch):
        if epoch <= 3:
            self.tactical_ratio = 1.0
        elif epoch <= 6:
            self.tactical_ratio = 0.7
        else:
            self.tactical_ratio = 0.5

    def __iter__(self):
        n_tactical = int(len(self) * self.tactical_ratio)
        n_normal = len(self) - n_tactical

        tactical = np.random.choice(self.tactical_indices, n_tactical, replace=True)
        normal = np.random.choice(self.normal_indices, n_normal, replace=True)

        indices = np.concatenate([tactical, normal])
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.tactical_indices) + len(self.normal_indices)
```

---

## Pareto Analysis: Is 6×128 Optimal?

**Empirical Benchmarks (RTX 4080, batch=64, 27 input planes):**

| Config | Params | Forward (ms) | Backward (ms) | Total (ms) |
|--------|--------|--------------|---------------|------------|
| 4×64 | 675K | 3.4 | 5.9 | 9.3 |
| 6×96 | 1.4M | 5.8 | 11.4 | 17.2 |
| **6×128** | **2.2M** | **6.7** | **17.4** | **24.1** |
| 8×128 | 2.8M | 9.0 | 22.6 | 31.6 |
| 6×192 | 4.5M | 14.1 | 37.3 | 51.4 |
| 10×128 | 3.4M | 11.0 | 28.0 | 39.0 |

**KataGo Progression** (for reference):

| Config | Days | Strength |
|--------|------|----------|
| 6×96 | 0-0.75 | Strong amateur |
| 10×128 | 0.75-1.75 | Professional |
| 15×192 | 1.75-7.5 | Superhuman |
| 20×256 | 7.5-19 | Superhuman+ |

**Decision**: Stay at 6×128. Add ownership head and auxiliary targets before scaling network.

**Scaling Path** (if accuracy plateaus after auxiliary targets):
1. First try: 8×128 (2.8M, +31% time) - adds depth
2. Then try: 10×128 (3.4M, +62% time) - more depth
3. Finally: 6×192 (4.5M, +113% time) - adds width

---

## Implementation Status

### Phase 1: Curriculum Learning ✅ COMPLETE
- [x] `has_tactical_activity()` in sgf_parser.py (planes 20, 23)
- [x] `CurriculumSampler` class with epoch-based weighting (100%→70%→50%)
- [x] `--curriculum` flag in train_supervised.py
- [x] Data loading with `include_tactical_mask`
- **Test**: `poetry run python train_supervised.py --tactical-features --curriculum`

### Phase 2: Ownership Head ✅ COMPLETE
- [x] `ownership_map()` in board.py (Tromp-Taylor flood-fill)
- [x] SGF parser returns ownership for all positions in game
- [x] `ownership_conv` head in model.py
- [x] `ProGameDataset` handles ownership with augmentation
- [x] `--ownership` flag added
- [x] Ownership loss in `train_step()` (weight: 1.5/b² with BCE)
- [x] Training loop unpacks ownership from batch
- **Test**: `poetry run python train_supervised.py --tactical-features --ownership`

### Phase 3: Opponent Move Auxiliary ✅ COMPLETE
- [x] Store opponent's next move in data pipeline (`include_opponent_move`)
- [x] Dual policy head (policy + opponent_policy_fc sharing conv features)
- [x] Add cross-entropy loss weighted by 0.15 (configurable via `--opponent-weight`)
- [x] `--opponent-move` flag added
- **Test**: `poetry run python train_supervised.py --tactical-features --opponent-move`

### Phase 4: Global Pooling Enhancement (LATER)
- Already have basic GlobalPoolingBlock
- [ ] Upgrade to KataGo's mean + scaled_mean + max
- [ ] Add to policy head

### Phase 5: Evaluate & Scale (LATER)
- [ ] Compare all variants on same val set
- [ ] Pick best combination
- [ ] If accuracy <40%, scale to 8×128 or 10×128

---

## Open Questions (Updated)

1. ~~Should we drop tactical features?~~ **No** - KataGo uses them. Problem is lack of auxiliary targets.

2. **Is ownership worth complexity?** **Yes** - 1.65× speedup in KataGo ablations. Highest impact single technique.

3. ~~Progressive network sizing?~~ **Later** - Focus on auxiliary targets first. KataGo saw bigger gains from targets than from scaling.

4. **Score distribution vs score mean?** Start with score mean only (2 outputs: μ, σ). Full PDF is optional.

---

## References

1. Wu, D. (2020). "Accelerating Self-Play Learning in Go." https://arxiv.org/abs/1902.10565 **← PRIMARY SOURCE**

2. Silver, D. et al. (2017). "Mastering the game of Go without human knowledge." Nature. https://www.nature.com/articles/nature24270

3. He, K. et al. (2016). "Identity Mappings in Deep Residual Networks." https://arxiv.org/abs/1603.05027

4. Hu, J. et al. (2018). "Squeeze-and-Excitation Networks." CVPR. (Related to global pooling)

5. Tian, Y. & Zhu, Y. (2016). "Better computer Go player with neural network and long-term prediction." ICLR. (Opponent move prediction origin)

6. Wu, T. et al. (2018). "Multi-labelled Value Networks for Computer Go." IEEE Trans. Games. (Ownership/score targets in supervised learning)

---

*Last updated: 2025-12-01*
*Source: KataGo paper full text analysis*

