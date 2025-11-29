"""Self-play game generation."""
import numpy as np
from typing import List, Tuple
from board import Board
from mcts import MCTS
from config import Config


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


def play_game(model, config: Config, game_idx: int = 0, verbose: bool = False) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Play a single self-play game."""
    board = Board(config.board_size)
    mcts = MCTS(model, config)
    record = GameRecord()

    move_count = 0
    max_moves = config.board_size ** 2 * 2  # Reasonable limit

    if verbose:
        print(f"  [Game {game_idx+1}] Starting game...")

    while not board.is_game_over() and move_count < max_moves:
        # Get MCTS policy
        if verbose and move_count % 10 == 0:
            print(f"  [Game {game_idx+1}] Move {move_count}/{max_moves}, running MCTS ({config.mcts_simulations} sims)...")

        policy = mcts.search(board, verbose=(verbose and move_count % 10 == 0))

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


def generate_games(model, config: Config, num_games: int, verbose: bool = True) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Generate multiple self-play games."""
    all_samples = []

    for i in range(num_games):
        samples = play_game(model, config, game_idx=i, verbose=verbose)
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
