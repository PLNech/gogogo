"""Hybrid MCTS: Neural Network + Symbolic Tactics.

Combines neural network pattern recognition with symbolic tactical
verification for stronger Go play.

The neural network provides:
- Policy prior (which moves to consider)
- Value estimation (who's winning)
- Strategic patterns (joseki, shape, influence)

The symbolic component provides:
- Ladder verification (deterministic)
- Capture sequence validation
- Snapback detection
- Life/death analysis

Usage:
    from hybrid_mcts import HybridMCTS
    from tactics import TacticalAnalyzer

    tactics = TacticalAnalyzer()
    mcts = HybridMCTS(model, config, tactics)
    policy = mcts.search(board)
"""

import numpy as np
import torch
import math
import time
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from board import Board
from config import Config
from tactics import TacticalAnalyzer
from mcts import MCTSNode, NNCache, PendingEval


class HybridMCTS:
    """MCTS with neural network + symbolic tactical enhancements."""

    def __init__(
        self,
        model,
        config: Config,
        tactics: TacticalAnalyzer,
        batch_size: int = 8,
        cache_size: int = 100000,
        use_cache: bool = True,
        tactical_weight: float = 0.3,
        tactical_threshold: float = 1.2
    ):
        """Initialize Hybrid MCTS.

        Args:
            model: Neural network model
            config: Configuration
            tactics: TacticalAnalyzer instance
            batch_size: Batch size for NN evaluation
            cache_size: Max entries in NN cache
            use_cache: Whether to use NN cache
            tactical_weight: How much to blend tactical adjustments (0-1)
            tactical_threshold: Minimum tactical boost to apply (>1.0)
        """
        self.model = model
        self.config = config
        self.tactics = tactics
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.cache = NNCache(max_size=cache_size) if use_cache else None
        self.tactical_weight = tactical_weight
        self.tactical_threshold = tactical_threshold

        # Stats tracking
        self.tactical_adjustments = 0
        self.total_evaluations = 0

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

    def _adjust_policy(self, board: Board, neural_policy: np.ndarray) -> np.ndarray:
        """Adjust neural policy with tactical boosts/penalties.

        Returns modified policy distribution.
        """
        adjusted = neural_policy.copy()
        any_adjustment = False

        # Get legal moves
        legal_moves = board.get_legal_moves()

        for move in legal_moves:
            if move == (-1, -1):
                continue  # Skip pass

            move_idx = move[0] * board.size + move[1]
            boost = self.tactics.get_tactical_boost(board, move)

            if boost > self.tactical_threshold or boost < 0.5:
                # Significant tactical signal
                adjusted[move_idx] *= boost
                any_adjustment = True

        if any_adjustment:
            self.tactical_adjustments += 1
            # Re-normalize
            if adjusted.sum() > 1e-8:
                adjusted = adjusted / adjusted.sum()
            else:
                # All moves penalized heavily - fall back to uniform
                adjusted = neural_policy.copy()

        return adjusted

    def _adjust_value(self, board: Board, neural_value: float) -> float:
        """Adjust neural value for tactical positions.

        For positions with clear tactical outcomes, blend neural value
        with tactical evaluation.
        """
        if not self.tactics.is_tactical_position(board):
            return neural_value

        # Find groups in danger
        player = board.current_player
        visited = set()
        tactical_value_sum = 0.0
        tactical_weight_sum = 0.0

        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != 0 and (r, c) not in visited:
                    group = board.get_group(r, c)
                    visited.update(group)

                    libs = board.count_liberties(group)
                    color = board.board[r, c]

                    if libs <= 2 and len(group) >= 2:
                        # Evaluate this group
                        status = self.tactics.evaluate_life_death(
                            board, group, depth=4
                        )

                        # Weight by group size
                        weight = len(group) / (board.size * board.size)

                        if color == player:
                            # Our group: alive is good
                            tactical_value_sum += status * weight
                        else:
                            # Opponent group: dead is good for us
                            tactical_value_sum -= status * weight

                        tactical_weight_sum += weight

        if tactical_weight_sum > 0.01:
            # Blend neural value with tactical assessment
            tactical_value = tactical_value_sum / tactical_weight_sum
            blend = self.tactical_weight * tactical_weight_sum  # Scale by importance
            blend = min(blend, 0.5)  # Cap at 50% tactical influence

            return (1 - blend) * neural_value + blend * tactical_value

        return neural_value

    def search(self, board: Board, verbose: bool = False) -> np.ndarray:
        """Run hybrid MCTS and return visit count distribution."""
        start_time = time.time()
        root = MCTSNode()
        self.total_evaluations = 0
        self.tactical_adjustments = 0

        # Expand root (check cache first)
        root_hash = board.zobrist_hash()
        cached = self.cache.get(root_hash) if self.cache else None

        use_tactical = getattr(self.config, 'tactical_features', False) or self.config.input_planes == 27

        if cached:
            policy, value = cached
            policy = self._adjust_policy(board, policy)
            root.expand(board, policy)
        else:
            policy, value = self._batch_predict([board.to_tensor(use_tactical_features=use_tactical)])
            policy = self._adjust_policy(board, policy[0])
            value = self._adjust_value(board, value[0])
            root.expand(board, policy)
            if self.cache:
                self.cache.put(root_hash, policy, value)

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
                        # Cache hit! Apply tactical adjustments and backprop
                        policy, value = cached
                        policy = self._adjust_policy(scratch_board, policy)
                        value = self._adjust_value(scratch_board, value)
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

            # Batch evaluate all pending leaves
            if pending:
                tensors = [p.tensor for p in pending]
                policies, values = self._batch_predict(tensors)
                self.total_evaluations += len(pending)

                for i, p in enumerate(pending):
                    # Apply tactical adjustments
                    policy = self._adjust_policy(p.board, policies[i])
                    value = self._adjust_value(p.board, values[i])

                    # Store adjusted values in cache
                    if self.cache:
                        self.cache.put(p.board_hash, policy, value)

                    # Expand node
                    p.node.expand(p.board, policy)

                    # Backprop
                    for n in reversed(p.search_path):
                        n.update(value)
                        value = -value

                    simulations_done += 1

        if verbose:
            elapsed = time.time() - start_time
            cache_str = f", {self.cache.stats()}" if self.cache else ""
            tac_str = f", {self.tactical_adjustments}/{self.total_evaluations} tactical"
            print(f"      HybridMCTS: {simulations_done} sims in {elapsed:.2f}s{cache_str}{tac_str}")

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


def create_hybrid_mcts(
    model,
    config: Config,
    tactical_weight: float = 0.3
) -> HybridMCTS:
    """Factory function to create HybridMCTS with default settings.

    Args:
        model: Neural network model
        config: Configuration
        tactical_weight: Blend weight for tactical adjustments (0-1)

    Returns:
        Configured HybridMCTS instance
    """
    tactics = TacticalAnalyzer(max_ladder_length=50)
    return HybridMCTS(
        model=model,
        config=config,
        tactics=tactics,
        batch_size=config.batch_size if hasattr(config, 'batch_size') else 8,
        tactical_weight=tactical_weight
    )
