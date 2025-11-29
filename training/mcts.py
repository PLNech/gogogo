"""Monte Carlo Tree Search with neural network (batched for GPU efficiency)."""
import numpy as np
import torch
import math
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from board import Board
from config import Config


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


class MCTS:
    """MCTS with batched neural network evaluation for GPU efficiency."""

    def __init__(self, model, config: Config, batch_size: int = 8):
        self.model = model
        self.config = config
        self.batch_size = batch_size

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

    def search(self, board: Board, verbose: bool = False) -> np.ndarray:
        """Run batched MCTS and return visit count distribution."""
        start_time = time.time()
        root = MCTSNode()

        # Expand root
        policy, value = self._batch_predict([board.to_tensor()])
        root.expand(board, policy[0])

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
                    # Add to batch for evaluation
                    pending.append(PendingEval(
                        tensor=scratch_board.to_tensor(),
                        node=node,
                        board=scratch_board,
                        search_path=search_path
                    ))

            # Batch evaluate all pending leaves
            if pending:
                tensors = [p.tensor for p in pending]
                policies, values = self._batch_predict(tensors)

                for i, p in enumerate(pending):
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
            print(f"      MCTS: {simulations_done} sims in {elapsed:.2f}s ({simulations_done/elapsed:.0f} sims/s)")

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
