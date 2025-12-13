"""Self-play game generation with optional tactical enhancement."""
import numpy as np
import time
from typing import List, Tuple, Optional
from board import Board
from mcts import MCTS
from config import Config


def get_game_stats(board: Board) -> dict:
    """Get current game statistics."""
    black_stones = np.sum(board.board == 1)
    white_stones = np.sum(board.board == -1)

    # Count groups
    visited = set()
    black_groups = 0
    white_groups = 0

    for r in range(board.size):
        for c in range(board.size):
            if (r, c) in visited or board.board[r, c] == 0:
                continue

            group = board.get_group(r, c)
            for pos in group:
                visited.add(pos)

            if board.board[r, c] == 1:
                black_groups += 1
            else:
                white_groups += 1

    # Territory estimate (simple version)
    score = board.score()

    return {
        'black_stones': black_stones,
        'white_stones': white_stones,
        'black_groups': black_groups,
        'white_groups': white_groups,
        'score': score,
        'empty': board.size ** 2 - black_stones - white_stones
    }


class GameRecord:
    """Record of a single game for training."""

    def __init__(self):
        self.states: List[np.ndarray] = []  # Board tensors
        self.policies: List[np.ndarray] = []  # MCTS policies
        self.current_players: List[int] = []  # Who played each move

    def add(self, state: np.ndarray, policy: np.ndarray, player: int):
        self.states.append(state)
        self.policies.append(policy)
        self.current_players.append(player)

    def finalize(self, winner: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Convert to training samples with final game result."""
        samples = []
        for state, policy, player in zip(self.states, self.policies, self.current_players):
            if winner == 0:
                value = 0.0
            elif player == winner:
                value = 1.0
            else:
                value = -1.0
            samples.append((state, policy, value))
        return samples


def play_game(model, config: Config, game_idx: int = 0, verbose: bool = False,
               batch_size: int = 8, use_hybrid: bool = False,
               tactical_weight: float = 0.3) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Play a single self-play game with batched MCTS.

    Args:
        model: Neural network model
        config: Configuration
        game_idx: Game index for logging
        verbose: Print progress
        batch_size: Batch size for MCTS
        use_hybrid: If True, use HybridMCTS with tactical enhancements
        tactical_weight: Weight for tactical adjustments (0-1)

    Returns:
        List of (state, policy, value) tuples for training
    """
    board = Board(config.board_size)

    if use_hybrid:
        from hybrid_mcts import HybridMCTS
        from tactics import TacticalAnalyzer
        tactics = TacticalAnalyzer()
        mcts = HybridMCTS(model, config, tactics, batch_size=batch_size,
                         tactical_weight=tactical_weight)
    else:
        mcts = MCTS(model, config, batch_size=batch_size)

    record = GameRecord()

    move_count = 0
    max_moves = config.board_size ** 2 * 2  # Reasonable limit

    if verbose:
        print(f"  [Game {game_idx+1}] Starting game...")

    while not board.is_game_over() and move_count < max_moves:
        # Get MCTS policy
        if verbose and move_count % 10 == 0:
            stats = get_game_stats(board)
            print(f"  [Game {game_idx+1}] Move {move_count}/{max_moves} | "
                  f"B:{stats['black_stones']}({stats['black_groups']}g) "
                  f"W:{stats['white_stones']}({stats['white_groups']}g) "
                  f"Score:{stats['score']:+.1f} Empty:{stats['empty']}")

        policy = mcts.search(board, verbose=False)  # Batched MCTS is already efficient

        # Record state before move
        record.add(
            board.to_tensor(),
            policy,
            board.current_player
        )

        # Select and play move
        temp = config.temperature if move_count < config.temp_threshold else 0.0
        action = mcts.select_action(board, temp)

        if action == (-1, -1):
            board.pass_move()
        else:
            board.play(action[0], action[1])

        move_count += 1

    # Determine winner
    score = board.score()
    if score > 0:
        winner = 1  # Black wins
    elif score < 0:
        winner = -1  # White wins
    else:
        winner = 0  # Draw

    if verbose:
        print(f"  [Game {game_idx+1}] Finished after {move_count} moves. Winner: {'Black' if winner == 1 else 'White' if winner == -1 else 'Draw'}")

    return record.finalize(winner)


def generate_games(model, config: Config, num_games: int, verbose: bool = True,
                   use_hybrid: bool = False, tactical_weight: float = 0.3) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Generate multiple self-play games.

    Args:
        model: Neural network model
        config: Configuration
        num_games: Number of games to generate
        verbose: Print progress
        use_hybrid: If True, use HybridMCTS with tactical enhancements
        tactical_weight: Weight for tactical adjustments (0-1)

    Returns:
        List of (state, policy, value) tuples for training
    """
    all_samples = []

    for i in range(num_games):
        samples = play_game(model, config, game_idx=i, verbose=verbose,
                           use_hybrid=use_hybrid, tactical_weight=tactical_weight)
        all_samples.extend(samples)
        print(f"Game {i+1}/{num_games}: {len(samples)} positions")

    return all_samples


class ReplayBuffer:
    """Circular buffer for training samples."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.position = 0

    def add(self, samples: List[Tuple[np.ndarray, np.ndarray, float]]):
        """Add samples to buffer."""
        for sample in samples:
            if len(self.buffer) < self.max_size:
                self.buffer.append(sample)
            else:
                self.buffer[self.position] = sample
            self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch for training."""
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)

        states = np.array([self.buffer[i][0] for i in indices])
        policies = np.array([self.buffer[i][1] for i in indices])
        values = np.array([self.buffer[i][2] for i in indices])

        return states, policies, values

    def __len__(self):
        return len(self.buffer)
