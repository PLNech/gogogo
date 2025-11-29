"""Monte Carlo Tree Search with neural network."""
import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional
from board import Board
from config import Config


class MCTSNode:
    """MCTS tree node."""

    def __init__(self, prior: float = 0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children: Dict[Tuple[int, int], MCTSNode] = {}

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

        # Add pass move
        legal_moves.append((-1, -1))  # Pass

        for move in legal_moves:
            if move == (-1, -1):
                action_idx = board.size ** 2  # Pass index
            else:
                action_idx = move[0] * board.size + move[1]

            prior = policy[action_idx]
            self.children[move] = MCTSNode(prior)

    def update(self, value: float):
        """Update node statistics."""
        self.visit_count += 1
        self.value_sum += value


class MCTS:
    """MCTS with neural network evaluation."""

    def __init__(self, model, config: Config):
        self.model = model
        self.config = config

    def search(self, board: Board, verbose: bool = False) -> np.ndarray:
        """Run MCTS and return visit count distribution."""
        start_time = time.time()
        root = MCTSNode()

        # Expand root with neural network
        policy, value = self.model.predict(board.to_tensor())
        root.expand(board, policy)

        for sim in range(self.config.mcts_simulations):
            if verbose and sim > 0 and sim % 10 == 0:
                elapsed = time.time() - start_time
                print(f"      MCTS sim {sim}/{self.config.mcts_simulations} ({elapsed:.1f}s elapsed)")
            _  = sim  # Use the variable
            node = root
            scratch_board = board.copy()
            search_path = [node]

            # Selection
            while node.expanded():
                action, node = node.select_child(self.config.c_puct)
                search_path.append(node)

                if action == (-1, -1):
                    scratch_board.pass_move()
                else:
                    scratch_board.play(action[0], action[1])

                if scratch_board.is_game_over():
                    break

            # Expansion and evaluation
            if not scratch_board.is_game_over():
                policy, value = self.model.predict(scratch_board.to_tensor())
                node.expand(scratch_board, policy)
                value = value
            else:
                # Game over: use actual score
                score = scratch_board.score()
                if scratch_board.current_player == 1:
                    value = 1.0 if score > 0 else (-1.0 if score < 0 else 0.0)
                else:
                    value = 1.0 if score < 0 else (-1.0 if score > 0 else 0.0)

            # Backpropagation
            for node in reversed(search_path):
                node.update(value)
                value = -value  # Negamax

        # Return visit count distribution
        action_size = board.size ** 2 + 1
        visits = np.zeros(action_size, dtype=np.float32)

        for action, child in root.children.items():
            if action == (-1, -1):
                action_idx = board.size ** 2
            else:
                action_idx = action[0] * board.size + action[1]
            visits[action_idx] = child.visit_count

        # Normalize
        if visits.sum() > 0:
            visits = visits / visits.sum()

        return visits

    def select_action(self, board: Board, temperature: float = 1.0) -> Tuple[int, int]:
        """Select action based on MCTS policy."""
        visits = self.search(board)

        if temperature == 0:
            action_idx = np.argmax(visits)
        else:
            # Sample from distribution
            visits_temp = visits ** (1 / temperature)
            visits_temp = visits_temp / visits_temp.sum()
            action_idx = np.random.choice(len(visits), p=visits_temp)

        if action_idx == board.size ** 2:
            return (-1, -1)  # Pass
        else:
            return (action_idx // board.size, action_idx % board.size)
