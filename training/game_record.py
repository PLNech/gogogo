"""Rich game recording with per-move statistics for analysis and visualization.

Captures everything needed to analyze game quality, model behavior, and training progress.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import json
from pathlib import Path


@dataclass
class MoveStats:
    """Statistics for a single move."""
    move_num: int
    move: Tuple[int, int]  # (-1, -1) for pass
    player: int  # 1=black, -1=white

    # Board state after move
    black_stones: int = 0
    white_stones: int = 0
    black_groups: int = 0
    white_groups: int = 0
    empty_points: int = 0

    # Captures
    captures_this_move: int = 0
    total_black_captures: int = 0  # Cumulative stones black has captured
    total_white_captures: int = 0  # Cumulative stones white has captured

    # Territory/Score estimate
    score_estimate: float = 0.0  # Positive = black ahead
    black_territory: int = 0
    white_territory: int = 0

    # Tactical situations
    ko_point: Optional[Tuple[int, int]] = None
    black_ataris: int = 0  # Black groups in atari
    white_ataris: int = 0  # White groups in atari

    # MCTS insights
    mcts_value: float = 0.0  # Model's P(black wins) from current position
    mcts_visits: int = 0  # Total MCTS simulations
    chosen_move_visits: int = 0  # Visits to the chosen move
    chosen_move_prior: float = 0.0  # Neural net's prior for chosen move
    policy_entropy: float = 0.0  # Entropy of MCTS policy (higher = more uncertain)
    top5_moves: List[Tuple[Tuple[int, int], float]] = field(default_factory=list)

    # Liberty stats
    min_black_liberties: int = 0  # Min liberties among black groups
    min_white_liberties: int = 0  # Min liberties among white groups
    avg_black_liberties: float = 0.0
    avg_white_liberties: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'move_num': self.move_num,
            'move': list(self.move),
            'player': self.player,
            'black_stones': self.black_stones,
            'white_stones': self.white_stones,
            'black_groups': self.black_groups,
            'white_groups': self.white_groups,
            'empty_points': self.empty_points,
            'captures_this_move': self.captures_this_move,
            'total_black_captures': self.total_black_captures,
            'total_white_captures': self.total_white_captures,
            'score_estimate': self.score_estimate,
            'black_territory': self.black_territory,
            'white_territory': self.white_territory,
            'ko_point': list(self.ko_point) if self.ko_point else None,
            'black_ataris': self.black_ataris,
            'white_ataris': self.white_ataris,
            'mcts_value': self.mcts_value,
            'mcts_visits': self.mcts_visits,
            'chosen_move_visits': self.chosen_move_visits,
            'chosen_move_prior': self.chosen_move_prior,
            'policy_entropy': self.policy_entropy,
            'top5_moves': [[list(m), p] for m, p in self.top5_moves],
            'min_black_liberties': self.min_black_liberties,
            'min_white_liberties': self.min_white_liberties,
            'avg_black_liberties': self.avg_black_liberties,
            'avg_white_liberties': self.avg_white_liberties,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MoveStats':
        """Create from dict."""
        return cls(
            move_num=d['move_num'],
            move=tuple(d['move']),
            player=d['player'],
            black_stones=d.get('black_stones', 0),
            white_stones=d.get('white_stones', 0),
            black_groups=d.get('black_groups', 0),
            white_groups=d.get('white_groups', 0),
            empty_points=d.get('empty_points', 0),
            captures_this_move=d.get('captures_this_move', 0),
            total_black_captures=d.get('total_black_captures', 0),
            total_white_captures=d.get('total_white_captures', 0),
            score_estimate=d.get('score_estimate', 0.0),
            black_territory=d.get('black_territory', 0),
            white_territory=d.get('white_territory', 0),
            ko_point=tuple(d['ko_point']) if d.get('ko_point') else None,
            black_ataris=d.get('black_ataris', 0),
            white_ataris=d.get('white_ataris', 0),
            mcts_value=d.get('mcts_value', 0.0),
            mcts_visits=d.get('mcts_visits', 0),
            chosen_move_visits=d.get('chosen_move_visits', 0),
            chosen_move_prior=d.get('chosen_move_prior', 0.0),
            policy_entropy=d.get('policy_entropy', 0.0),
            top5_moves=[(tuple(m), p) for m, p in d.get('top5_moves', [])],
            min_black_liberties=d.get('min_black_liberties', 0),
            min_white_liberties=d.get('min_white_liberties', 0),
            avg_black_liberties=d.get('avg_black_liberties', 0.0),
            avg_white_liberties=d.get('avg_white_liberties', 0.0),
        )


@dataclass
class GameRecord:
    """Complete record of a game with rich statistics."""

    board_size: int = 9
    game_id: str = ""
    timestamp: str = ""

    # Game result
    winner: int = 0  # 1=black, -1=white, 0=draw
    final_score: float = 0.0
    result_string: str = ""  # e.g., "B+5.5", "W+R"
    total_moves: int = 0

    # Per-move data
    move_stats: List[MoveStats] = field(default_factory=list)

    # Training data (for backward compat with old selfplay)
    states: List[np.ndarray] = field(default_factory=list)
    policies: List[np.ndarray] = field(default_factory=list)

    # Metadata
    black_name: str = "GoGoGo"
    white_name: str = "GoGoGo"
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def add_move(self, stats: MoveStats, state: np.ndarray = None, policy: np.ndarray = None):
        """Add a move with its statistics."""
        self.move_stats.append(stats)
        if state is not None:
            self.states.append(state)
        if policy is not None:
            self.policies.append(policy)

    def finalize(self, winner: int, score: float) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Finalize game and return training samples."""
        self.winner = winner
        self.final_score = score
        self.total_moves = len(self.move_stats)

        if score > 0:
            self.result_string = f"B+{score:.1f}"
        elif score < 0:
            self.result_string = f"W+{-score:.1f}"
        else:
            self.result_string = "Draw"

        # Generate training samples
        samples = []
        for i, stats in enumerate(self.move_stats):
            if i < len(self.states) and i < len(self.policies):
                if winner == 0:
                    value = 0.0
                elif stats.player == winner:
                    value = 1.0
                else:
                    value = -1.0
                samples.append((self.states[i], self.policies[i], value))

        return samples

    def to_sgf(self) -> str:
        """Export to SGF format with comments."""
        sgf = f"(;GM[1]FF[4]CA[UTF-8]SZ[{self.board_size}]\n"
        sgf += f"PB[{self.black_name}]PW[{self.white_name}]RE[{self.result_string}]\n"

        for stats in self.move_stats:
            color = "B" if stats.player == 1 else "W"
            if stats.move == (-1, -1):
                sgf += f";{color}[]"
            else:
                row, col = stats.move
                sgf_col = chr(ord('a') + col)
                sgf_row = chr(ord('a') + row)
                sgf += f";{color}[{sgf_col}{sgf_row}]"

            # Add rich comment
            comment = f"V:{stats.mcts_value:.2f} Cap:{stats.captures_this_move}"
            if stats.ko_point:
                comment += " Ko"
            sgf += f"C[{comment}]"

            if (stats.move_num + 1) % 10 == 0:
                sgf += "\n"

        sgf += ")\n"
        return sgf

    def to_json(self) -> str:
        """Export complete record to JSON."""
        return json.dumps({
            'board_size': self.board_size,
            'game_id': self.game_id,
            'timestamp': self.timestamp,
            'winner': self.winner,
            'final_score': self.final_score,
            'result_string': self.result_string,
            'total_moves': self.total_moves,
            'black_name': self.black_name,
            'white_name': self.white_name,
            'config_snapshot': self.config_snapshot,
            'move_stats': [s.to_dict() for s in self.move_stats],
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'GameRecord':
        """Load from JSON."""
        d = json.loads(json_str)
        record = cls(
            board_size=d['board_size'],
            game_id=d.get('game_id', ''),
            timestamp=d.get('timestamp', ''),
            winner=d['winner'],
            final_score=d['final_score'],
            result_string=d['result_string'],
            total_moves=d['total_moves'],
            black_name=d.get('black_name', 'GoGoGo'),
            white_name=d.get('white_name', 'GoGoGo'),
            config_snapshot=d.get('config_snapshot', {}),
        )
        record.move_stats = [MoveStats.from_dict(s) for s in d['move_stats']]
        return record

    def save(self, path: str):
        """Save to JSON file."""
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> 'GameRecord':
        """Load from JSON file."""
        return cls.from_json(Path(path).read_text())

    # Convenience accessors for visualization
    @property
    def black_win_probs(self) -> List[float]:
        """Black's win probability at each move."""
        return [s.mcts_value for s in self.move_stats]

    @property
    def white_win_probs(self) -> List[float]:
        """White's win probability at each move."""
        return [1.0 - s.mcts_value for s in self.move_stats]

    @property
    def black_group_counts(self) -> List[int]:
        return [s.black_groups for s in self.move_stats]

    @property
    def white_group_counts(self) -> List[int]:
        return [s.white_groups for s in self.move_stats]

    @property
    def score_trajectory(self) -> List[float]:
        return [s.score_estimate for s in self.move_stats]

    @property
    def capture_events(self) -> List[Tuple[int, int, int]]:
        """List of (move_num, player, stones_captured)."""
        return [(s.move_num, s.player, s.captures_this_move)
                for s in self.move_stats if s.captures_this_move > 0]


def compute_move_stats(board, move: Tuple[int, int], player: int,
                       move_num: int, captures: int,
                       mcts_policy: np.ndarray = None,
                       mcts_value: float = 0.0,
                       mcts_visits: int = 0,
                       cumulative_captures: Tuple[int, int] = (0, 0)) -> MoveStats:
    """Compute rich statistics for a move.

    Args:
        board: Board state AFTER the move
        move: Move coordinates
        player: Who played (1=black, -1=white)
        move_num: Move number
        captures: Stones captured by this move
        mcts_policy: MCTS visit distribution
        mcts_value: Model's value estimate (P(black wins))
        mcts_visits: Total MCTS simulations
        cumulative_captures: (black_total, white_total) before this move
    """
    stats = MoveStats(move_num=move_num, move=move, player=player)

    # Stone counts
    stats.black_stones = int(np.sum(board.board == 1))
    stats.white_stones = int(np.sum(board.board == -1))
    stats.empty_points = board.size ** 2 - stats.black_stones - stats.white_stones

    # Captures
    stats.captures_this_move = captures
    black_cap, white_cap = cumulative_captures
    if player == 1:
        black_cap += captures
    else:
        white_cap += captures
    stats.total_black_captures = black_cap
    stats.total_white_captures = white_cap

    # Groups and liberties
    visited = set()
    black_groups = []
    white_groups = []

    for r in range(board.size):
        for c in range(board.size):
            if (r, c) in visited or board.board[r, c] == 0:
                continue
            group = board.get_group(r, c)
            for pos in group:
                visited.add(pos)
            liberties = board.count_liberties(group)

            if board.board[r, c] == 1:
                black_groups.append(liberties)
            else:
                white_groups.append(liberties)

    stats.black_groups = len(black_groups)
    stats.white_groups = len(white_groups)

    if black_groups:
        stats.min_black_liberties = min(black_groups)
        stats.avg_black_liberties = sum(black_groups) / len(black_groups)
        stats.black_ataris = sum(1 for lib in black_groups if lib == 1)

    if white_groups:
        stats.min_white_liberties = min(white_groups)
        stats.avg_white_liberties = sum(white_groups) / len(white_groups)
        stats.white_ataris = sum(1 for lib in white_groups if lib == 1)

    # Ko
    stats.ko_point = board.ko_point

    # Score estimate
    stats.score_estimate = board.score()

    # MCTS stats
    stats.mcts_value = mcts_value
    stats.mcts_visits = mcts_visits

    if mcts_policy is not None:
        # Policy entropy
        policy = mcts_policy.flatten()
        policy = policy[policy > 0]  # Only non-zero
        if len(policy) > 0:
            stats.policy_entropy = float(-np.sum(policy * np.log(policy + 1e-10)))

        # Chosen move stats
        if move != (-1, -1):
            move_idx = move[0] * board.size + move[1]
        else:
            move_idx = board.size ** 2  # Pass

        if move_idx < len(mcts_policy):
            stats.chosen_move_visits = int(mcts_policy[move_idx] * mcts_visits) if mcts_visits > 0 else 0
            stats.chosen_move_prior = float(mcts_policy[move_idx])

        # Top 5 moves
        flat_policy = mcts_policy.flatten()
        top_indices = np.argsort(flat_policy)[-5:][::-1]
        stats.top5_moves = []
        for idx in top_indices:
            if idx == board.size ** 2:
                m = (-1, -1)  # Pass
            else:
                m = (idx // board.size, idx % board.size)
            stats.top5_moves.append((m, float(flat_policy[idx])))

    return stats
