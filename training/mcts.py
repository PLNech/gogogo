"""Monte Carlo Tree Search with neural network (batched for GPU efficiency)."""
import numpy as np
import torch
import math
import time
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from board import Board
from config import Config


class NNCache:
    """LRU cache for neural network evaluations.

    Caches (policy, value) results keyed by Zobrist hash.
    Provides up to 5.8× speedup by avoiding redundant NN calls.

    Source: [Speculative MCTS, NeurIPS 2024]

    Usage:
        - Within MCTS: same position visited multiple times
        - Between moves: tree reuse means many cached evaluations
    """

    def __init__(self, max_size: int = 100000):
        """Initialize cache.

        Args:
            max_size: Maximum number of entries (default 100K ~= 50MB)
        """
        self.max_size = max_size
        self.cache: OrderedDict[int, Tuple[np.ndarray, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, board_hash: int) -> Optional[Tuple[np.ndarray, float]]:
        """Get cached (policy, value) for position.

        Args:
            board_hash: Zobrist hash of position

        Returns:
            (policy, value) tuple if cached, None otherwise
        """
        if board_hash in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(board_hash)
            return self.cache[board_hash]
        self.misses += 1
        return None

    def put(self, board_hash: int, policy: np.ndarray, value: float):
        """Cache (policy, value) for position.

        Args:
            board_hash: Zobrist hash of position
            policy: Policy distribution from NN
            value: Value estimate from NN
        """
        if board_hash in self.cache:
            # Update existing and move to end
            self.cache.move_to_end(board_hash)
            self.cache[board_hash] = (policy, value)
        else:
            # Add new entry
            self.cache[board_hash] = (policy, value)
            # Evict oldest if over capacity
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> str:
        """Return cache statistics string."""
        return f"NNCache: {len(self.cache)}/{self.max_size} entries, {self.hit_rate:.1%} hit rate ({self.hits} hits, {self.misses} misses)"


class MCTSNode:
    """MCTS tree node."""

    def __init__(self, prior: float = 0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children: Dict[Tuple[int, int], 'MCTSNode'] = {}

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return len(self.children) > 0

    def select_child(self, c_puct: float) -> Tuple[Tuple[int, int], 'MCTSNode']:
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = self._ucb_score(child, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _ucb_score(self, child: 'MCTSNode', c_puct: float) -> float:
        """UCB score: Q + U where U = c * P * sqrt(N_parent) / (1 + N_child)."""
        prior_score = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
        value_score = -child.value  # Negamax: opponent's value is negative for us
        return value_score + prior_score

    def expand(self, board: Board, policy: np.ndarray):
        """Expand node with legal moves."""
        legal_moves = board.get_legal_moves()
        legal_moves.append((-1, -1))  # Pass

        for move in legal_moves:
            if move == (-1, -1):
                action_idx = board.size ** 2
            else:
                action_idx = move[0] * board.size + move[1]
            prior = policy[action_idx]
            self.children[move] = MCTSNode(prior)

    def update(self, value: float):
        """Update node statistics."""
        self.visit_count += 1
        self.value_sum += value


@dataclass
class PendingEval:
    """Leaf node waiting for neural network evaluation."""
    tensor: np.ndarray
    node: MCTSNode
    board: Board
    search_path: List[MCTSNode]
    board_hash: int  # Zobrist hash for cache lookup


class MCTS:
    """MCTS with batched neural network evaluation for GPU efficiency."""

    def __init__(self, model, config: Config, batch_size: int = 8,
                 cache_size: int = 100000, use_cache: bool = True):
        """Initialize MCTS.

        Args:
            model: Neural network model
            config: Configuration
            batch_size: Batch size for NN evaluation
            cache_size: Max entries in NN evaluation cache (default 100K)
            use_cache: Whether to use NN cache (default True)
        """
        self.model = model
        self.config = config
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.cache = NNCache(max_size=cache_size) if use_cache else None

    def _apply_root_policy_temp(self, policy: np.ndarray, move_number: int = 0) -> np.ndarray:
        """Apply softmax temperature to root policy (KataGo A.5).

        Higher temperature (>1) flattens the distribution, counteracting
        MCTS's tendency to sharpen already-preferred moves.

        Source: KataGo Methods - "softmax temperature slightly above 1"
        """
        # Select temperature based on game phase
        if move_number < self.config.root_policy_temp_early_moves:
            temp = self.config.root_policy_temp_early
        else:
            temp = self.config.root_policy_temp

        if temp == 1.0:
            return policy

        # Convert to logits, apply temperature, convert back
        # policy is already softmax output, so we take log first
        eps = 1e-8
        log_policy = np.log(policy + eps)
        scaled_logits = log_policy / temp
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        return exp_logits / (exp_logits.sum() + eps)

    def _add_dirichlet_noise(self, policy: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        """Add shaped Dirichlet noise to root policy (KataGo A.4).

        Shaped noise concentrates exploration on moves with higher policy,
        finding blind spots more efficiently than uniform noise.

        Source: KataGo Methods - "concentrates the other half of the alpha
        on the smaller subset of moves"
        """
        alpha = self.config.root_dirichlet_alpha
        frac = self.config.root_exploration_fraction

        # Only add noise to legal moves
        num_legal = legal_mask.sum()
        if num_legal == 0:
            return policy

        if self.config.shaped_dirichlet:
            # Shaped: concentrate noise on higher-policy moves
            # Split alpha: half uniform, half weighted by policy
            legal_policy = policy * legal_mask
            policy_sum = legal_policy.sum()

            if policy_sum > 0:
                # Normalized policy for legal moves
                norm_policy = legal_policy / policy_sum
                # Alpha per move: half uniform + half policy-weighted
                alpha_per_move = alpha * (0.5 / num_legal + 0.5 * norm_policy)
            else:
                alpha_per_move = np.full_like(policy, alpha / num_legal) * legal_mask

            # Sample from Dirichlet with shaped alpha
            # Only sample for legal moves
            legal_indices = np.where(legal_mask > 0)[0]
            legal_alphas = alpha_per_move[legal_indices]
            noise = np.zeros_like(policy)
            noise[legal_indices] = np.random.dirichlet(legal_alphas + 1e-8)
        else:
            # Uniform: standard Dirichlet noise
            noise = np.zeros_like(policy)
            legal_indices = np.where(legal_mask > 0)[0]
            noise[legal_indices] = np.random.dirichlet([alpha] * len(legal_indices))

        # Blend: (1 - frac) * policy + frac * noise
        noisy_policy = (1 - frac) * policy + frac * noise
        # Zero out illegal moves (policy may have non-zero values for illegal moves)
        noisy_policy = noisy_policy * legal_mask
        # Renormalize
        total = noisy_policy.sum()
        if total > 0:
            noisy_policy = noisy_policy / total

        return noisy_policy

    def _batch_predict(self, tensors: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Batch predict multiple positions at once."""
        if len(tensors) == 0:
            return np.array([]), np.array([])

        batch = np.stack(tensors)
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(batch).to(next(self.model.parameters()).device)
            policies, values = self.model(x)
            return torch.exp(policies).cpu().numpy(), values.cpu().numpy().flatten()

    def search(self, board: Board, verbose: bool = False, move_number: int = 0,
               add_noise: bool = True) -> np.ndarray:
        """Run batched MCTS and return visit count distribution.

        Args:
            board: Current board state
            verbose: Print timing info
            move_number: Current move number (for temperature scheduling)
            add_noise: Whether to add Dirichlet noise at root (for training)
        """
        start_time = time.time()
        root = MCTSNode()

        # Expand root (check cache first)
        root_hash = board.zobrist_hash()
        cached = self.cache.get(root_hash) if self.cache else None
        use_tactical = getattr(self.config, 'tactical_features', False) or self.config.input_planes == 27
        if cached:
            policy, value = cached
        else:
            policy, value = self._batch_predict([board.to_tensor(use_tactical_features=use_tactical)])
            policy = policy[0]
            value = value[0]
            if self.cache:
                self.cache.put(root_hash, policy, value)

        # KataGo A.5: Apply root policy temperature
        policy = self._apply_root_policy_temp(policy, move_number)

        # KataGo A.4: Add shaped Dirichlet noise at root (for exploration during training)
        if add_noise:
            # Build legal move mask
            action_size = board.size ** 2 + 1
            legal_mask = np.zeros(action_size, dtype=np.float32)
            for move in board.get_legal_moves():
                action_idx = move[0] * board.size + move[1]
                legal_mask[action_idx] = 1.0
            legal_mask[board.size ** 2] = 1.0  # Pass is always legal

            policy = self._add_dirichlet_noise(policy, legal_mask)

        root.expand(board, policy)

        simulations_done = 0
        while simulations_done < self.config.mcts_simulations:
            # Collect batch of leaves to evaluate
            pending: List[PendingEval] = []

            for _ in range(min(self.batch_size, self.config.mcts_simulations - simulations_done)):
                # Selection: traverse tree to find leaf
                node = root
                scratch_board = board.copy()
                search_path = [node]

                while node.expanded():
                    action, node = node.select_child(self.config.c_puct)
                    search_path.append(node)

                    if action == (-1, -1):
                        scratch_board.pass_move()
                    else:
                        scratch_board.play(action[0], action[1])

                    if scratch_board.is_game_over():
                        break

                if scratch_board.is_game_over():
                    # Terminal node - use actual score
                    score = scratch_board.score()
                    if scratch_board.current_player == 1:
                        value = 1.0 if score > 0 else (-1.0 if score < 0 else 0.0)
                    else:
                        value = 1.0 if score < 0 else (-1.0 if score > 0 else 0.0)

                    # Backprop immediately
                    for n in reversed(search_path):
                        n.update(value)
                        value = -value
                    simulations_done += 1
                else:
                    # Check cache before adding to pending batch
                    pos_hash = scratch_board.zobrist_hash()
                    cached = self.cache.get(pos_hash) if self.cache else None

                    if cached:
                        # Cache hit! Expand and backprop immediately
                        policy, value = cached
                        node.expand(scratch_board, policy)

                        for n in reversed(search_path):
                            n.update(value)
                            value = -value
                        simulations_done += 1
                    else:
                        # Cache miss - add to batch for NN evaluation
                        pending.append(PendingEval(
                            tensor=scratch_board.to_tensor(use_tactical_features=use_tactical),
                            node=node,
                            board=scratch_board,
                            search_path=search_path,
                            board_hash=pos_hash
                        ))

            # Batch evaluate all pending leaves (cache misses only)
            if pending:
                tensors = [p.tensor for p in pending]
                policies, values = self._batch_predict(tensors)

                for i, p in enumerate(pending):
                    # Store in cache
                    if self.cache:
                        self.cache.put(p.board_hash, policies[i], values[i])

                    # Expand node
                    p.node.expand(p.board, policies[i])

                    # Backprop
                    value = values[i]
                    for n in reversed(p.search_path):
                        n.update(value)
                        value = -value

                    simulations_done += 1

        if verbose:
            elapsed = time.time() - start_time
            cache_str = f", {self.cache.stats()}" if self.cache else ""
            print(f"      MCTS: {simulations_done} sims in {elapsed:.2f}s ({simulations_done/elapsed:.0f} sims/s){cache_str}")

        # Return visit distribution
        action_size = board.size ** 2 + 1
        visits = np.zeros(action_size, dtype=np.float32)

        for action, child in root.children.items():
            if action == (-1, -1):
                action_idx = board.size ** 2
            else:
                action_idx = action[0] * board.size + action[1]
            visits[action_idx] = child.visit_count

        if visits.sum() > 0:
            visits = visits / visits.sum()

        return visits

    def select_action(self, board: Board, temperature: float = 1.0) -> Tuple[int, int]:
        """Select action based on MCTS policy."""
        visits = self.search(board)

        if temperature == 0:
            action_idx = np.argmax(visits)
        else:
            visits_temp = visits ** (1 / temperature)
            visits_temp = visits_temp / (visits_temp.sum() + 1e-8)
            action_idx = np.random.choice(len(visits), p=visits_temp)

        if action_idx == board.size ** 2:
            return (-1, -1)
        else:
            return (action_idx // board.size, action_idx % board.size)
