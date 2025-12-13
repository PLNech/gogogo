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

    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self.states: List[np.ndarray] = []  # Board tensors
        self.policies: List[np.ndarray] = []  # MCTS policies
        self.current_players: List[int] = []  # Who played each move
        self.moves: List[Tuple[int, int]] = []  # Move coordinates
        self.comments: List[str] = []  # Optional comments per move

    def add(self, state: np.ndarray, policy: np.ndarray, player: int,
            move: Tuple[int, int] = None, comment: str = ""):
        self.states.append(state)
        self.policies.append(policy)
        self.current_players.append(player)
        if move is not None:
            self.moves.append(move)
        self.comments.append(comment)

    def to_sgf(self, result: str = "?", black_name: str = "GoGoGo",
               white_name: str = "GoGoGo") -> str:
        """Export game to SGF format."""
        sgf = f"(;GM[1]FF[4]CA[UTF-8]SZ[{self.board_size}]\n"
        sgf += f"PB[{black_name}]PW[{white_name}]RE[{result}]\n"

        for i, (move, player) in enumerate(zip(self.moves, self.current_players)):
            color = "B" if player == 1 else "W"
            if move == (-1, -1):
                sgf += f";{color}[]"  # Pass
            else:
                row, col = move
                # SGF uses a-s for 19x19, lowercase
                sgf_col = chr(ord('a') + col)
                sgf_row = chr(ord('a') + row)
                sgf += f";{color}[{sgf_col}{sgf_row}]"

            # Add comment if present
            if i < len(self.comments) and self.comments[i]:
                sgf += f"C[{self.comments[i]}]"

            # Newline every 10 moves for readability
            if (i + 1) % 10 == 0:
                sgf += "\n"

        sgf += ")\n"
        return sgf

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

    record = GameRecord(board_size=config.board_size)

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

        # Select move first so we can record it
        temp = config.temperature if move_count < config.temp_threshold else 0.0
        action = mcts.select_action(board, temp)

        # Record state before move (with the move that will be played)
        record.add(
            board.to_tensor(),
            policy,
            board.current_player,
            move=action
        )

        # Play the move
        if action == (-1, -1):
            board.pass_move()
        else:
            board.play(action[0], action[1])

        move_count += 1

    # Determine winner
    score = board.score()
    if score > 0:
        winner = 1  # Black wins
        result = f"B+{score:.1f}"
    elif score < 0:
        winner = -1  # White wins
        result = f"W+{-score:.1f}"
    else:
        winner = 0  # Draw
        result = "0"

    if verbose:
        print(f"  [Game {game_idx+1}] Finished after {move_count} moves. "
              f"Result: {result}")

    # Store result and record for potential SGF export
    record.winner = winner
    record.result = result
    record.score = score

    return record.finalize(winner), record


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
        print(f"Game {i+1}/{num_games}: {len(samples)} positions, {record.result}")

        if save_sgf:
            sgf_path = os.path.join(save_sgf, f"game_{i+1:04d}.sgf")
            with open(sgf_path, 'w') as f:
                f.write(record.to_sgf(result=record.result))

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
