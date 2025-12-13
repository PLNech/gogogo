"""Board representation and tensor conversion."""
import numpy as np
from typing import List, Tuple, Optional

# Zobrist hashing for fast board state hashing (NN cache key)
# Pre-computed random 64-bit numbers for each (position, color) combination
# Using fixed seed for reproducibility across sessions
_ZOBRIST_RNG = np.random.RandomState(42)
_MAX_BOARD_SIZE = 19
_ZOBRIST_TABLE = _ZOBRIST_RNG.randint(
    0, 2**63, size=(2, _MAX_BOARD_SIZE, _MAX_BOARD_SIZE), dtype=np.uint64
)
_ZOBRIST_TURN = _ZOBRIST_RNG.randint(0, 2**63, dtype=np.uint64)  # For current player


class Board:
    """Go board with tensor conversion for neural network."""

    def __init__(self, size: int = 9):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)  # 0=empty, 1=black, -1=white
        self.current_player = 1  # 1=black, -1=white
        self.history: List[np.ndarray] = []
        self.ko_point: Optional[Tuple[int, int]] = None
        self.passes = 0
        self.move_count = 0

    def copy(self) -> 'Board':
        b = Board(self.size)
        b.board = self.board.copy()
        b.current_player = self.current_player
        b.history = [h.copy() for h in self.history]
        b.ko_point = self.ko_point
        b.passes = self.passes
        b.move_count = self.move_count
        return b

    def get_group(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all stones in a group."""
        color = self.board[row, col]
        if color == 0:
            return []

        group = []
        visited = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            if r < 0 or r >= self.size or c < 0 or c >= self.size:
                continue
            if self.board[r, c] != color:
                continue

            visited.add((r, c))
            group.append((r, c))
            stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])

        return group

    def count_liberties(self, group: List[Tuple[int, int]]) -> int:
        """Count liberties of a group."""
        liberties = set()
        for r, c in group:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr, nc] == 0:
                        liberties.add((nr, nc))
        return len(liberties)

    def remove_group(self, group: List[Tuple[int, int]]) -> int:
        """Remove a group from the board. Returns number of stones removed."""
        for r, c in group:
            self.board[r, c] = 0
        return len(group)

    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if move is valid (no suicide, no ko)."""
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        if self.board[row, col] != 0:
            return False
        if self.ko_point == (row, col):
            return False

        # Simulate the move to check for suicide
        test_board = self.copy()
        test_board.board[row, col] = self.current_player

        # First, capture any opponent groups with 0 liberties
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if test_board.board[nr, nc] == -self.current_player:
                    group = test_board.get_group(nr, nc)
                    if test_board.count_liberties(group) == 0:
                        # Remove captured stones on test board
                        for gr, gc in group:
                            test_board.board[gr, gc] = 0

        # Now check if our own group has liberties
        our_group = test_board.get_group(row, col)
        if test_board.count_liberties(our_group) == 0:
            return False  # Suicide - illegal

        return True

    def play(self, row: int, col: int) -> int:
        """Play a move. Returns number of captures."""
        if not self.is_valid_move(row, col):
            raise ValueError(f"Invalid move: {row}, {col}")

        # Save history
        self.history.append(self.board.copy())
        if len(self.history) > 8:
            self.history.pop(0)

        # Place stone
        self.board[row, col] = self.current_player
        self.passes = 0
        self.move_count += 1

        # Capture opponent stones
        captured = 0
        captured_group = None
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if self.board[nr, nc] == -self.current_player:
                    group = self.get_group(nr, nc)
                    if self.count_liberties(group) == 0:
                        captured += self.remove_group(group)
                        if len(group) == 1:
                            captured_group = group[0]

        # Ko detection
        if captured == 1 and captured_group:
            own_group = self.get_group(row, col)
            if len(own_group) == 1 and self.count_liberties(own_group) == 1:
                self.ko_point = captured_group
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # Switch player
        self.current_player = -self.current_player

        return captured

    def pass_move(self):
        """Pass."""
        self.passes += 1
        self.current_player = -self.current_player
        self.ko_point = None

    def is_game_over(self) -> bool:
        """Check if game is over (two consecutive passes)."""
        return self.passes >= 2

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Get all legal moves."""
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.is_valid_move(r, c):
                    moves.append((r, c))
        return moves

    def has_eye(self, row: int, col: int, color: int) -> bool:
        """Check if position is a true eye for the given color.

        True eye: empty point where all adjacent positions are same color
        (or edge) and at least 3 of 4 diagonals are controlled.
        """
        if self.board[row, col] != 0:
            return False

        # Check all adjacent positions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if self.board[nr, nc] != color:
                    return False

        # Check diagonals (need 3/4, or all if on edge/corner)
        diagonals = []
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                diagonals.append((nr, nc))

        controlled = 0
        for nr, nc in diagonals:
            if self.board[nr, nc] == color or self.board[nr, nc] == 0:
                controlled += 1

        # Need at least 3/4 (or all if corner/edge with fewer diagonals)
        min_required = 3 if len(diagonals) == 4 else len(diagonals)
        return controlled >= min_required

    def count_eyes(self, group: List[Tuple[int, int]]) -> int:
        """Count true eyes in a group."""
        if not group:
            return 0

        color = self.board[group[0][0], group[0][1]]
        if color == 0:
            return 0

        # Find all empty adjacent positions
        eye_candidates = set()
        for r, c in group:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr, nc] == 0:
                        eye_candidates.add((nr, nc))

        # Count true eyes
        eyes = 0
        for r, c in eye_candidates:
            if self.has_eye(r, c, color):
                eyes += 1

        return eyes

    def is_group_alive(self, group: List[Tuple[int, int]]) -> bool:
        """Determine if a group is unconditionally alive.

        A group is alive if:
        - It has 2+ true eyes, OR
        - It's connected to living territory (simplified: many liberties)

        A group is dead if:
        - It has 0 eyes and is completely surrounded (0 liberties would be captured)
        - It has 0-1 eyes and opponent controls all surrounding space
        """
        if not group:
            return False

        eyes = self.count_eyes(group)
        liberties = self.count_liberties(group)

        # Two eyes = unconditionally alive
        if eyes >= 2:
            return True

        # Large group with many liberties is likely alive
        if len(group) >= 6 and liberties >= 4:
            return True

        # Check if group is surrounded (simplified heuristic)
        # If group has 1 eye and few liberties, check if it can make a second eye
        if eyes == 1 and liberties <= 2:
            return False  # Likely dead

        if eyes == 0:
            # No eyes - check if opponent surrounds all liberties
            color = self.board[group[0][0], group[0][1]]
            opponent = -color

            # Get all liberty positions
            liberty_positions = set()
            for r, c in group:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if self.board[nr, nc] == 0:
                            liberty_positions.add((nr, nc))

            # Check each liberty - if playing there wouldn't help survive
            # This is a heuristic: if opponent controls the area, group is dead
            opponent_surrounding = 0
            for lr, lc in liberty_positions:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = lr + dr, lc + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if self.board[nr, nc] == opponent:
                            opponent_surrounding += 1

            # If heavily surrounded with no eyes, likely dead
            if opponent_surrounding >= len(liberty_positions) * 2 and liberties <= 3:
                return False

        # Default: assume alive (conservative)
        return True

    def remove_dead_stones(self) -> Tuple[int, int]:
        """Remove dead stones from board. Returns (black_removed, white_removed)."""
        black_removed = 0
        white_removed = 0

        visited = set()

        for r in range(self.size):
            for c in range(self.size):
                if (r, c) in visited:
                    continue
                if self.board[r, c] == 0:
                    continue

                group = self.get_group(r, c)
                for pos in group:
                    visited.add(pos)

                if not self.is_group_alive(group):
                    color = self.board[r, c]
                    removed = self.remove_group(group)
                    if color == 1:
                        black_removed += removed
                    else:
                        white_removed += removed

        return black_removed, white_removed

    def score(self) -> float:
        """Simple area scoring. Positive = black wins."""
        black = np.sum(self.board == 1)
        white = np.sum(self.board == -1)

        # Count territory (empty points surrounded by one color)
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 0:
                    # BFS to find owner of empty region
                    region = []
                    visited = set()
                    stack = [(r, c)]
                    touches_black = False
                    touches_white = False

                    while stack:
                        rr, cc = stack.pop()
                        if (rr, cc) in visited:
                            continue
                        if rr < 0 or rr >= self.size or cc < 0 or cc >= self.size:
                            continue

                        if self.board[rr, cc] == 1:
                            touches_black = True
                            continue
                        if self.board[rr, cc] == -1:
                            touches_white = True
                            continue

                        visited.add((rr, cc))
                        region.append((rr, cc))
                        stack.extend([(rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)])

                    if touches_black and not touches_white:
                        black += len(region)
                    elif touches_white and not touches_black:
                        white += len(region)

        komi = 6.5 if self.size >= 9 else 0.5
        return black - white - komi

    def score_with_dead_removal(self) -> float:
        """Score with automatic dead stone removal.

        Creates a copy, removes dead stones, then scores.
        More accurate than simple score() for endgame positions.
        """
        scoring_board = self.copy()
        black_dead, white_dead = scoring_board.remove_dead_stones()

        # Now score the cleaned board
        base_score = scoring_board.score()

        # In territory scoring, captured stones count double (removed + added to opponent)
        # But since we use area scoring, just count the clean board
        return base_score

    def ownership_map(self) -> np.ndarray:
        """
        Compute per-point ownership using Tromp-Taylor (area) scoring.

        Returns: (size, size) array with values:
            +1 = black owns (stone or territory)
            -1 = white owns (stone or territory)
             0 = neutral (dame, seki)

        Used as auxiliary target for neural network training.
        See KataGo paper §4.1 for why ownership prediction helps.
        """
        ownership = np.zeros((self.size, self.size), dtype=np.float32)

        # Stones are owned by their color
        ownership[self.board == 1] = 1.0   # Black stones
        ownership[self.board == -1] = -1.0  # White stones

        # Track visited empty points to avoid recomputing regions
        visited_empty = np.zeros((self.size, self.size), dtype=bool)

        # Determine territory for empty points
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 0 and not visited_empty[r, c]:
                    # BFS to find the entire empty region and what it touches
                    region = []
                    stack = [(r, c)]
                    touches_black = False
                    touches_white = False

                    while stack:
                        rr, cc = stack.pop()
                        if rr < 0 or rr >= self.size or cc < 0 or cc >= self.size:
                            continue
                        if visited_empty[rr, cc]:
                            continue

                        if self.board[rr, cc] == 1:
                            touches_black = True
                            continue
                        if self.board[rr, cc] == -1:
                            touches_white = True
                            continue

                        # Empty point
                        visited_empty[rr, cc] = True
                        region.append((rr, cc))
                        stack.extend([(rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)])

                    # Assign ownership to the region
                    if touches_black and not touches_white:
                        for (rr, cc) in region:
                            ownership[rr, cc] = 1.0  # Black territory
                    elif touches_white and not touches_black:
                        for (rr, cc) in region:
                            ownership[rr, cc] = -1.0  # White territory
                    # else: neutral (dame) - stays 0

        return ownership

    def to_tensor(self, use_tactical_features: bool = False) -> np.ndarray:
        """Convert board to neural network input tensor.

        Returns: (N, size, size) tensor with planes:
        Basic (17 planes):
        - 0: current player's stones
        - 1: opponent's stones
        - 2-9: current player's history (last 8 moves)
        - 10-16: opponent's history (last 7 moves)

        Tactical features (+10 planes = 27 total):
        - 17: current player groups with 1 liberty (atari!)
        - 18: current player groups with 2 liberties (danger)
        - 19: current player groups with 3+ liberties (safe)
        - 20: opponent groups with 1 liberty (can capture!)
        - 21: opponent groups with 2 liberties (can atari)
        - 22: opponent groups with 3+ liberties
        - 23: capture moves (playing here captures opponent)
        - 24: self-atari moves (playing here puts self in atari)
        - 25: eye-like points for current player
        - 26: edge distance (normalized)
        """
        n_planes = 27 if use_tactical_features else 17
        planes = np.zeros((n_planes, self.size, self.size), dtype=np.float32)

        # Current position
        my_color = self.current_player
        opp_color = -self.current_player
        if my_color == 1:
            planes[0] = (self.board == 1).astype(np.float32)
            planes[1] = (self.board == -1).astype(np.float32)
        else:
            planes[0] = (self.board == -1).astype(np.float32)
            planes[1] = (self.board == 1).astype(np.float32)

        # History planes (8 for current player: 2-9, 7 for opponent: 10-16)
        history = list(reversed(self.history[-8:]))
        for i, hist in enumerate(history):
            if i < 8:  # planes 2-9 for current player
                if my_color == 1:
                    planes[2 + i] = (hist == 1).astype(np.float32)
                else:
                    planes[2 + i] = (hist == -1).astype(np.float32)
            if i < 7:  # planes 10-16 for opponent
                if my_color == 1:
                    planes[10 + i] = (hist == -1).astype(np.float32)
                else:
                    planes[10 + i] = (hist == 1).astype(np.float32)

        if not use_tactical_features:
            return planes

        # === TACTICAL FEATURE PLANES ===
        # Track which groups we've already processed
        processed = set()

        # Liberty planes for current player's groups
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == my_color and (r, c) not in processed:
                    group = self.get_group(r, c)
                    libs = self.count_liberties(group)
                    processed.update(group)

                    # Mark all stones in group with liberty count
                    for gr, gc in group:
                        if libs == 1:
                            planes[17, gr, gc] = 1.0  # In atari!
                        elif libs == 2:
                            planes[18, gr, gc] = 1.0  # Danger
                        else:
                            planes[19, gr, gc] = 1.0  # Safe

        # Liberty planes for opponent's groups
        processed.clear()
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == opp_color and (r, c) not in processed:
                    group = self.get_group(r, c)
                    libs = self.count_liberties(group)
                    processed.update(group)

                    for gr, gc in group:
                        if libs == 1:
                            planes[20, gr, gc] = 1.0  # Can capture!
                        elif libs == 2:
                            planes[21, gr, gc] = 1.0  # Can atari
                        else:
                            planes[22, gr, gc] = 1.0  # Stable

        # FAST capture detection: mark liberties of opponent groups in atari
        # Instead of checking each empty position, find atari groups and mark their liberties
        for r in range(self.size):
            for c in range(self.size):
                # If opponent stone in atari (from planes[20]), mark its liberties
                if planes[20, r, c] == 1.0:  # Opponent group in atari
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            if self.board[nr, nc] == 0:
                                planes[23, nr, nc] = 1.0  # Capture move!

        # FAST self-atari detection: mark positions adjacent to own groups with 2 liberties
        # If we play next to a 2-liberty group and fill one liberty, it becomes atari
        # Skip full simulation - just mark risky positions
        for r in range(self.size):
            for c in range(self.size):
                if planes[18, r, c] == 1.0:  # Own group with 2 liberties
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            if self.board[nr, nc] == 0:
                                planes[24, nr, nc] = 1.0  # Potential self-atari

        # Eye-like points (empty surrounded by friendly stones)
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    continue

                # Count adjacent friendly/opponent/edge
                friendly = 0
                opponent = 0
                edge = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
                        edge += 1
                    elif self.board[nr, nc] == my_color:
                        friendly += 1
                    elif self.board[nr, nc] == opp_color:
                        opponent += 1

                # Eye-like: surrounded by friendly (allow edge)
                if friendly + edge >= 3 and opponent == 0:
                    planes[25, r, c] = 1.0

        # Edge distance (encourages spreading out, corners > edges > center)
        # Vectorized for speed
        rows = np.arange(self.size).reshape(-1, 1)
        cols = np.arange(self.size).reshape(1, -1)
        dist_r = np.minimum(rows, self.size - 1 - rows)
        dist_c = np.minimum(cols, self.size - 1 - cols)
        planes[26] = np.minimum(dist_r, dist_c) / (self.size / 2)

        return planes

    def __str__(self) -> str:
        symbols = {0: '.', 1: 'X', -1: 'O'}
        rows = []
        for r in range(self.size):
            row = ' '.join(symbols[self.board[r, c]] for c in range(self.size))
            rows.append(f"{r:2d} {row}")
        header = '   ' + ' '.join(str(c) for c in range(self.size))
        return header + '\n' + '\n'.join(rows)

    def zobrist_hash(self) -> int:
        """Compute Zobrist hash for current position.

        Used as cache key for NN evaluation cache (MCTS speedup).
        Includes: stone positions + current player to move.
        O(board_size²) but very fast in practice.

        Returns:
            64-bit hash uniquely identifying this position
        """
        h = np.uint64(0)

        # Hash stone positions
        for r in range(self.size):
            for c in range(self.size):
                stone = self.board[r, c]
                if stone == 1:  # Black
                    h ^= _ZOBRIST_TABLE[0, r, c]
                elif stone == -1:  # White
                    h ^= _ZOBRIST_TABLE[1, r, c]

        # Hash current player (important: same position, different turn = different hash)
        if self.current_player == -1:  # White to play
            h ^= _ZOBRIST_TURN

        return int(h)
