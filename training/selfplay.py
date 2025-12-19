"""Self-play game generation with optional tactical enhancement."""
import numpy as np
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from board import Board
from mcts import MCTS
from config import Config
from game_record import GameRecord, MoveStats, compute_move_stats


# Directory for saving games (for dashboard)
GAMES_DIR = Path(__file__).parent / "games"
GAMES_DIR.mkdir(exist_ok=True)


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


## Old GameRecord removed - now imported from game_record.py


def play_game(model, config: Config, game_idx: int = 0, verbose: bool = False,
               batch_size: int = 8, use_hybrid: bool = False,
               tactical_weight: float = 0.3,
               save_for_dashboard: bool = True) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Play a single self-play game with batched MCTS.

    Args:
        model: Neural network model
        config: Configuration
        game_idx: Game index for logging
        verbose: Print progress
        batch_size: Batch size for MCTS
        use_hybrid: If True, use HybridMCTS with tactical enhancements
        tactical_weight: Weight for tactical adjustments (0-1)
        save_for_dashboard: Save game to games/ directory for dashboard

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

    # Create rich game record
    game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    record = GameRecord(
        board_size=config.board_size,
        game_id=game_id,
        timestamp=datetime.now().isoformat(),
    )

    move_count = 0
    max_moves = config.board_size ** 2 * 2  # Reasonable limit
    black_captures = 0
    white_captures = 0

    if verbose:
        print(f"  [Game {game_idx+1}] Starting game...")

    while not board.is_game_over() and move_count < max_moves:
        player = board.current_player

        # Get MCTS policy and value
        policy = mcts.search(board, verbose=False, move_number=move_count)

        # Get value estimate from the root
        mcts_value = 0.5  # Default
        if hasattr(mcts, 'root') and mcts.root is not None:
            mcts_value = (mcts.root.value() + 1) / 2  # Convert from [-1,1] to [0,1]

        # Select move
        temp = config.temperature if move_count < config.temp_threshold else 0.0
        action = mcts.select_action(board, temp)

        # Count captures before move
        stones_before = np.sum(board.board != 0)

        # Play the move
        if action == (-1, -1):
            board.pass_move()
            captures = 0
        else:
            board.play(action[0], action[1])
            stones_after = np.sum(board.board != 0)
            # Captures = stones removed (before + 1 for new stone - after)
            captures = stones_before + 1 - stones_after
            if captures > 0:
                if player == 1:
                    black_captures += captures
                else:
                    white_captures += captures

        # Compute rich stats AFTER the move
        move_stats = compute_move_stats(
            board=board,
            move=action,
            player=player,
            move_num=move_count,
            captures=captures,
            mcts_policy=policy,
            mcts_value=mcts_value,
            mcts_visits=config.mcts_simulations,
            cumulative_captures=(black_captures - (captures if player == 1 else 0),
                                white_captures - (captures if player == -1 else 0))
        )

        # Add to record with training data
        record.add_move(move_stats, board.to_tensor(), policy)

        if verbose and move_count % 10 == 0:
            print(f"  [Game {game_idx+1}] Move {move_count}/{max_moves} | "
                  f"B:{move_stats.black_stones}({move_stats.black_groups}g) "
                  f"W:{move_stats.white_stones}({move_stats.white_groups}g) "
                  f"Score:{move_stats.score_estimate:+.1f} V:{mcts_value:.2f}")

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
        print(f"  [Game {game_idx+1}] Finished after {move_count} moves. "
              f"Result: {record.result_string}")

    # Finalize and get training samples
    samples = record.finalize(winner, score)

    # Save for dashboard (every Nth game to avoid too many files)
    if save_for_dashboard and game_idx % 10 == 0:
        record.save(str(GAMES_DIR / f"{game_id}.json"))

    return samples, record


def generate_games(model, config: Config, num_games: int, verbose: bool = True,
                   use_hybrid: bool = False, tactical_weight: float = 0.3,
                   save_sgf: str = None) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Generate multiple self-play games.

    Args:
        model: Neural network model
        config: Configuration
        num_games: Number of games to generate
        verbose: Print progress
        use_hybrid: If True, use HybridMCTS with tactical enhancements
        tactical_weight: Weight for tactical adjustments (0-1)
        save_sgf: If provided, save games to this directory as SGF files

    Returns:
        List of (state, policy, value) tuples for training
    """
    import os
    all_samples = []
    records = []

    if save_sgf:
        os.makedirs(save_sgf, exist_ok=True)

    for i in range(num_games):
        samples, record = play_game(model, config, game_idx=i, verbose=verbose,
                                    use_hybrid=use_hybrid, tactical_weight=tactical_weight)
        all_samples.extend(samples)
        records.append(record)
        print(f"Game {i+1}/{num_games}: {len(samples)} positions, {record.result_string}")

        if save_sgf:
            sgf_path = os.path.join(save_sgf, f"game_{i+1:04d}.sgf")
            with open(sgf_path, 'w') as f:
                f.write(record.to_sgf())

    return all_samples, records


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
