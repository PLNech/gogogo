"""Board representation and tensor conversion."""
import numpy as np
from typing import List, Tuple, Optional

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
        """Check if move is valid."""
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        if self.board[row, col] != 0:
            return False
        if self.ko_point == (row, col):
            return False

        # Check for suicide
        test_board = self.copy()
        test_board.board[row, col] = self.current_player

        # Check if we capture anything
        captured = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if test_board.board[nr, nc] == -self.current_player:
                    group = test_board.get_group(nr, nc)
                    if test_board.count_liberties(group) == 0:
                        captured = True
                        break

        if not captured:
            # Check if our own group has liberties
            group = test_board.get_group(row, col)
            if test_board.count_liberties(group) == 0:
                return False

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

    def to_tensor(self) -> np.ndarray:
        """Convert board to neural network input tensor.

        Returns: (17, size, size) tensor with planes:
        - 0: current player's stones
        - 1: opponent's stones
        - 2-9: current player's history (last 8 moves)
        - 10-16: opponent's history (last 7 moves)
        - Or simplified: just current position + whose turn
        """
        planes = np.zeros((17, self.size, self.size), dtype=np.float32)

        # Current position
        if self.current_player == 1:
            planes[0] = (self.board == 1).astype(np.float32)
            planes[1] = (self.board == -1).astype(np.float32)
        else:
            planes[0] = (self.board == -1).astype(np.float32)
            planes[1] = (self.board == 1).astype(np.float32)

        # History planes (8 for current player: 2-9, 7 for opponent: 10-16)
        history = list(reversed(self.history[-8:]))
        for i, hist in enumerate(history):
            if i < 8:  # planes 2-9 for current player
                if self.current_player == 1:
                    planes[2 + i] = (hist == 1).astype(np.float32)
                else:
                    planes[2 + i] = (hist == -1).astype(np.float32)
            if i < 7:  # planes 10-16 for opponent
                if self.current_player == 1:
                    planes[10 + i] = (hist == -1).astype(np.float32)
                else:
                    planes[10 + i] = (hist == 1).astype(np.float32)

        return planes

    def __str__(self) -> str:
        symbols = {0: '.', 1: 'X', -1: 'O'}
        rows = []
        for r in range(self.size):
            row = ' '.join(symbols[self.board[r, c]] for c in range(self.size))
            rows.append(f"{r:2d} {row}")
        header = '   ' + ' '.join(str(c) for c in range(self.size))
        return header + '\n' + '\n'.join(rows)
