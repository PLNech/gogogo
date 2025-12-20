#!/usr/bin/env python3
"""Sensei's 8 Basic Instincts - Pattern Detectors.

The 8 Basic Instincts (https://senseis.xmp.net/?BasicInstinct):
1. Extend from Atari (アタリから伸びよ) - Stone in atari → extend
2. Hane vs Tsuke (ツケにはハネ) - Opponent attaches → hane
3. Hane at Head of Two (二子の頭にハネ) - Two stones in row → play above
4. Stretch from Kosumi (コスミから伸びよ) - Diagonal attach → stretch away
5. Block the Angle (カケにはオサエ) - Angle attack → block
6. Connect vs Peep (ノゾキにはツギ) - Opponent peeps → connect
7. Block the Thrust (ツキアタリには) - Opponent thrusts → block
8. Stretch from Bump (ブツカリから伸びよ) - Supported attach → stretch

Usage:
    from sensei_instincts import SenseiInstinctDetector

    detector = SenseiInstinctDetector()
    result = detector.detect_extend_from_atari(board)
    if result:
        print(f"Extend moves: {result.moves}")
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
from board import Board


@dataclass
class InstinctResult:
    """Result of an instinct detection."""
    instinct: str  # Name of the instinct
    moves: List[Tuple[int, int]]  # Correct response moves
    priority: float  # Higher = more urgent (survival > development)


class SenseiInstinctDetector:
    """Detects Sensei's 8 Basic Instincts in Go positions."""

    # Priority order - survival instincts highest
    PRIORITIES = {
        'extend_from_atari': 3.0,    # Survival - highest
        'connect_vs_peep': 2.5,      # Shape integrity
        'block_the_thrust': 2.0,     # Prevent cut
        'hane_vs_tsuke': 1.5,        # Development
        'hane_at_head_of_two': 1.5,  # Attack
        'stretch_from_kosumi': 1.2,  # Shape
        'block_the_angle': 1.2,      # Defense
        'stretch_from_bump': 1.0,    # Shape
    }

    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(self):
        pass

    def _in_bounds(self, board: Board, r: int, c: int) -> bool:
        """Check if position is within board bounds."""
        return 0 <= r < board.size and 0 <= c < board.size

    def _get_neighbors(self, board: Board, r: int, c: int) -> List[Tuple[int, int]]:
        """Get orthogonal neighbors within bounds."""
        return [(r + dr, c + dc) for dr, dc in self.DIRECTIONS
                if self._in_bounds(board, r + dr, c + dc)]

    def _get_diagonal_neighbors(self, board: Board, r: int, c: int) -> List[Tuple[int, int]]:
        """Get diagonal neighbors within bounds."""
        return [(r + dr, c + dc) for dr, dc in self.DIAGONALS
                if self._in_bounds(board, r + dr, c + dc)]

    # =========================================================================
    # 1. EXTEND FROM ATARI (アタリから伸びよ)
    # =========================================================================

    def detect_extend_from_atari(self, board: Board) -> Optional[InstinctResult]:
        """Detect if current player has stones in atari that should extend.

        When your stone faces capture (one liberty), extend to gain liberties.
        This is the most fundamental survival instinct.
        """
        player = board.current_player
        extend_moves = []

        # Find our groups in atari (1 liberty)
        visited: Set[Tuple[int, int]] = set()

        for r in range(board.size):
            for c in range(board.size):
                if (r, c) in visited or board.board[r, c] != player:
                    continue

                group = board.get_group(r, c)
                visited.update(group)

                liberties = board.count_liberties(group)
                if liberties == 1:
                    # Find the liberty - this is where we extend
                    for gr, gc in group:
                        for nr, nc in self._get_neighbors(board, gr, gc):
                            if board.board[nr, nc] == 0:
                                extend_moves.append((nr, nc))

        if extend_moves:
            return InstinctResult(
                instinct='extend_from_atari',
                moves=list(set(extend_moves)),
                priority=self.PRIORITIES['extend_from_atari']
            )
        return None

    # =========================================================================
    # 2. HANE VS TSUKE (ツケにはハネ)
    # =========================================================================

    def detect_hane_vs_tsuke(self, board: Board) -> Optional[InstinctResult]:
        """Detect if opponent played a tsuke (unsupported attachment).

        When opponent plays an unsupported adjacent stone, respond with hane.
        Tsuke = attachment with NO support stones nearby.
        """
        player = board.current_player
        opponent = -player
        hane_moves = []

        # Find our stones
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != player:
                    continue

                # Check for opponent tsuke (adjacent unsupported attachment)
                for nr, nc in self._get_neighbors(board, r, c):
                    if board.board[nr, nc] != opponent:
                        continue

                    # Check if this opponent stone is unsupported (tsuke)
                    # Tsuke = no other opponent stones adjacent to it
                    has_support = False
                    for sr, sc in self._get_neighbors(board, nr, nc):
                        if (sr, sc) != (r, c) and board.board[sr, sc] == opponent:
                            has_support = True
                            break

                    if not has_support:
                        # Found a tsuke! Hane moves are diagonal from the tsuke
                        # Hane goes around the opponent stone
                        dr, dc = nr - r, nc - c

                        # Hane positions: diagonal to the tsuke stone
                        hane_candidates = []
                        if dr != 0:  # Vertical attachment
                            hane_candidates = [(nr, nc - 1), (nr, nc + 1)]
                        else:  # Horizontal attachment
                            hane_candidates = [(nr - 1, nc), (nr + 1, nc)]

                        for hr, hc in hane_candidates:
                            if self._in_bounds(board, hr, hc) and board.board[hr, hc] == 0:
                                hane_moves.append((hr, hc))

        if hane_moves:
            return InstinctResult(
                instinct='hane_vs_tsuke',
                moves=list(set(hane_moves)),
                priority=self.PRIORITIES['hane_vs_tsuke']
            )
        return None

    # =========================================================================
    # 3. HANE AT HEAD OF TWO (二子の頭にハネ)
    # =========================================================================

    def detect_hane_at_head_of_two(self, board: Board) -> Optional[InstinctResult]:
        """Detect 2v2 confrontation where we can play at head of opponent's two.

        This is a 2v2 pattern: our two stones facing their two stones in parallel.
        Play at the HEAD of opponent's two to wrap around and gain influence.

        Pattern (vertical example):
            . . * . .    ← Play at head of opponent's two
            . 0 1 . .    ← Our stone (0) faces their stone (1)
            . 2 3 . .    ← Our stone (2) faces their stone (3)
        """
        player = board.current_player
        opponent = -player
        head_moves = []

        # Find pairs of OUR stones first
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != player:
                    continue

                # Check for adjacent friendly stone (our pair)
                for dr, dc in [(0, 1), (1, 0)]:  # Right and down to avoid duplicates
                    nr, nc = r + dr, c + dc
                    if not self._in_bounds(board, nr, nc):
                        continue
                    if board.board[nr, nc] != player:
                        continue

                    # We have a pair of our stones at (r,c) and (nr,nc)
                    # Check for PARALLEL opponent pair
                    # The opponent pair should be adjacent perpendicular to our pair direction

                    # Perpendicular directions
                    if dr == 0:  # Our pair is horizontal
                        perp_dirs = [(-1, 0), (1, 0)]  # Check above and below
                    else:  # Our pair is vertical
                        perp_dirs = [(0, -1), (0, 1)]  # Check left and right

                    for pdr, pdc in perp_dirs:
                        # Check if opponent has parallel pair
                        opp1_r, opp1_c = r + pdr, c + pdc
                        opp2_r, opp2_c = nr + pdr, nc + pdc

                        if not (self._in_bounds(board, opp1_r, opp1_c) and
                                self._in_bounds(board, opp2_r, opp2_c)):
                            continue

                        if (board.board[opp1_r, opp1_c] == opponent and
                            board.board[opp2_r, opp2_c] == opponent):
                            # Found 2v2! Now find the head of opponent's two
                            # Head is at the extension of their pair

                            # Check both ends of opponent's pair
                            # "prev" end (before first opponent stone in pair direction)
                            prev_r, prev_c = opp1_r - dr, opp1_c - dc
                            # "next" end (after second opponent stone in pair direction)
                            next_r, next_c = opp2_r + dr, opp2_c + dc

                            # Add valid head moves
                            if (self._in_bounds(board, prev_r, prev_c) and
                                board.board[prev_r, prev_c] == 0):
                                head_moves.append((prev_r, prev_c))

                            if (self._in_bounds(board, next_r, next_c) and
                                board.board[next_r, next_c] == 0):
                                head_moves.append((next_r, next_c))

        if head_moves:
            return InstinctResult(
                instinct='hane_at_head_of_two',
                moves=list(set(head_moves)),
                priority=self.PRIORITIES['hane_at_head_of_two']
            )
        return None

    # =========================================================================
    # 4. STRETCH FROM KOSUMI (コスミから伸びよ)
    # =========================================================================

    def detect_stretch_from_kosumi(self, board: Board) -> Optional[InstinctResult]:
        """Detect opponent diagonal attachment (kosumi-tsuke).

        When opponent plays a diagonal attachment, stretch away.
        """
        player = board.current_player
        opponent = -player
        stretch_moves = []

        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != player:
                    continue

                # Check for diagonal opponent stones (kosumi-tsuke)
                for dr, dc in self.DIAGONALS:
                    nr, nc = r + dr, c + dc
                    if not self._in_bounds(board, nr, nc):
                        continue
                    if board.board[nr, nc] != opponent:
                        continue

                    # Found kosumi-tsuke! Stretch away from it
                    # Stretch = extend in opposite direction
                    stretch_r, stretch_c = r - dr, c - dc
                    if self._in_bounds(board, stretch_r, stretch_c) and board.board[stretch_r, stretch_c] == 0:
                        stretch_moves.append((stretch_r, stretch_c))

                    # Also consider orthogonal extensions away
                    for odr, odc in self.DIRECTIONS:
                        # Only extend away from the kosumi direction
                        if (odr * dr >= 0 and odc * dc >= 0):
                            continue  # This goes toward the kosumi
                        or_, oc = r + odr, c + odc
                        if self._in_bounds(board, or_, oc) and board.board[or_, oc] == 0:
                            stretch_moves.append((or_, oc))

        if stretch_moves:
            return InstinctResult(
                instinct='stretch_from_kosumi',
                moves=list(set(stretch_moves)),
                priority=self.PRIORITIES['stretch_from_kosumi']
            )
        return None

    # =========================================================================
    # 5. BLOCK THE ANGLE (カケにはオサエ)
    # =========================================================================

    # Knight's move (keima) offsets - NOT diagonals!
    KEIMA_OFFSETS = [
        (-1, -2), (-1, 2), (1, -2), (1, 2),   # 1 vertical, 2 horizontal
        (-2, -1), (-2, 1), (2, -1), (2, 1),   # 2 vertical, 1 horizontal
    ]

    def detect_block_the_angle(self, board: Board) -> Optional[InstinctResult]:
        """Detect opponent knight's move (keima) approach and find block moves.

        When opponent plays a knight's move approach to your stone, block diagonally.

        Pattern:
            . . . . .
            . . . B .    ← Your stone
            . . * . .    ← Block diagonally
            . W . . .    ← Opponent's knight's move approach
            . . . . .

        The block is played diagonally between your stone and opponent's keima.
        """
        player = board.current_player
        opponent = -player
        block_moves = []

        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != player:
                    continue

                # Check for opponent knight's move (keima) approach
                for dr, dc in self.KEIMA_OFFSETS:
                    nr, nc = r + dr, c + dc
                    if not self._in_bounds(board, nr, nc):
                        continue
                    if board.board[nr, nc] != opponent:
                        continue

                    # Found keima approach! Block is diagonal between us and opponent
                    # The block goes in the direction that narrows the gap
                    # For keima at (r+dr, c+dc), block is at the "waist" of the knight's move

                    # Determine block position - it's the diagonal step toward the keima
                    if abs(dr) == 1:  # Keima is 1 row, 2 cols away
                        # Block is 1 step toward opponent in both directions
                        block_r = r + dr
                        block_c = c + (1 if dc > 0 else -1)
                    else:  # Keima is 2 rows, 1 col away
                        block_r = r + (1 if dr > 0 else -1)
                        block_c = c + dc

                    if self._in_bounds(board, block_r, block_c) and board.board[block_r, block_c] == 0:
                        block_moves.append((block_r, block_c))

        if block_moves:
            return InstinctResult(
                instinct='block_the_angle',
                moves=list(set(block_moves)),
                priority=self.PRIORITIES['block_the_angle']
            )
        return None

    # =========================================================================
    # 6. CONNECT VS PEEP (ノゾキにはツギ)
    # =========================================================================

    def detect_connect_vs_peep(self, board: Board) -> Optional[InstinctResult]:
        """Detect opponent peep and find connection point.

        'Even a moron connects against a peep.'
        When opponent peeps between your stones, connect immediately.
        """
        player = board.current_player
        opponent = -player
        connect_moves = []

        # Find cutting points between our stones
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != 0:
                    continue

                # Check if this empty point is between two of our stones
                player_neighbors = []
                for nr, nc in self._get_neighbors(board, r, c):
                    if board.board[nr, nc] == player:
                        player_neighbors.append((nr, nc))

                if len(player_neighbors) < 2:
                    continue

                # This is a cutting point! Check if opponent is peeping
                for nr, nc in self._get_neighbors(board, r, c):
                    if board.board[nr, nc] == opponent:
                        # Opponent is adjacent to our cutting point = peep!
                        connect_moves.append((r, c))
                        break

        if connect_moves:
            return InstinctResult(
                instinct='connect_vs_peep',
                moves=list(set(connect_moves)),
                priority=self.PRIORITIES['connect_vs_peep']
            )
        return None

    # =========================================================================
    # 7. BLOCK THE THRUST (ツキアタリには)
    # =========================================================================

    def detect_block_the_thrust(self, board: Board) -> Optional[InstinctResult]:
        """Detect opponent thrust into our stone formation.

        When opponent thrusts into your wall (adjacent to a stone in your line),
        block by extending the wall.

        Pattern (vertical wall):
            . . . . .
            . . * . .    ← Black BLOCKS by extending wall
            . B W . .    ← White THRUSTS adjacent to our stone
            . B . . .    ← Our stones in column (wall)
            . . . . .

        The thrust is perpendicular to our wall. Block extends the wall.
        """
        player = board.current_player
        opponent = -player
        block_moves = []

        # Find our stones that are part of a wall (2+ in a line)
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != player:
                    continue

                # Check if this stone is part of a wall in each direction
                for wall_dr, wall_dc in [(0, 1), (1, 0)]:  # horizontal and vertical walls
                    # Check for adjacent friendly stone forming a wall
                    wall_r, wall_c = r + wall_dr, c + wall_dc
                    if not self._in_bounds(board, wall_r, wall_c):
                        continue
                    if board.board[wall_r, wall_c] != player:
                        continue

                    # Found a wall segment! (r,c) and (wall_r, wall_c)
                    # Check for thrust perpendicular to the wall
                    if wall_dr == 0:  # Horizontal wall (stones side by side)
                        perp_dirs = [(-1, 0), (1, 0)]  # thrust from above/below
                    else:  # Vertical wall (stones stacked)
                        perp_dirs = [(0, -1), (0, 1)]  # thrust from left/right

                    # Check both stones in the wall for adjacent opponent thrust
                    for stone_r, stone_c in [(r, c), (wall_r, wall_c)]:
                        for perp_dr, perp_dc in perp_dirs:
                            thrust_r = stone_r + perp_dr
                            thrust_c = stone_c + perp_dc
                            if not self._in_bounds(board, thrust_r, thrust_c):
                                continue
                            if board.board[thrust_r, thrust_c] != opponent:
                                continue

                            # Found thrust! Block by extending the wall in thrust direction
                            # The block position is adjacent to the opponent stone,
                            # continuing our wall
                            block_r = thrust_r + perp_dr
                            block_c = thrust_c + perp_dc

                            # Also consider blocking by extending parallel to wall
                            # at the head of the thrust (same row/col as thrust but extending wall)
                            if wall_dr == 0:  # Horizontal wall
                                # Block above/below the thrust stone in wall direction
                                block_along_r = thrust_r
                                block_along_c = thrust_c + wall_dc if (thrust_c + wall_dc) not in [c, wall_c] else thrust_c - wall_dc
                            else:  # Vertical wall
                                block_along_r = thrust_r + wall_dr if (thrust_r + wall_dr) not in [r, wall_r] else thrust_r - wall_dr
                                block_along_c = thrust_c

                            # Primary block: extending in direction of thrust (blocking their path)
                            if self._in_bounds(board, block_r, block_c) and board.board[block_r, block_c] == 0:
                                block_moves.append((block_r, block_c))

        if block_moves:
            return InstinctResult(
                instinct='block_the_thrust',
                moves=list(set(block_moves)),
                priority=self.PRIORITIES['block_the_thrust']
            )
        return None

    # =========================================================================
    # 8. STRETCH FROM BUMP (ブツカリから伸びよ)
    # =========================================================================

    def detect_stretch_from_bump(self, board: Board) -> Optional[InstinctResult]:
        """Detect opponent bump (supported attachment).

        When opponent bumps (attachment with support), stretch rather than hane.
        Bump = attachment where the attaching stone has a support stone behind it.
        """
        player = board.current_player
        opponent = -player
        stretch_moves = []

        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != player:
                    continue

                # Check for opponent attachment
                for dr, dc in self.DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if not self._in_bounds(board, nr, nc):
                        continue
                    if board.board[nr, nc] != opponent:
                        continue

                    # Check if this opponent stone has support (bump, not tsuke)
                    support_r, support_c = nr + dr, nc + dc
                    if (self._in_bounds(board, support_r, support_c) and
                        board.board[support_r, support_c] == opponent):
                        # This is a bump! Stretch away
                        stretch_r, stretch_c = r - dr, c - dc
                        if self._in_bounds(board, stretch_r, stretch_c) and board.board[stretch_r, stretch_c] == 0:
                            stretch_moves.append((stretch_r, stretch_c))

        if stretch_moves:
            return InstinctResult(
                instinct='stretch_from_bump',
                moves=list(set(stretch_moves)),
                priority=self.PRIORITIES['stretch_from_bump']
            )
        return None

    # =========================================================================
    # DETECT ALL
    # =========================================================================

    def detect_all(self, board: Board) -> List[InstinctResult]:
        """Detect all instinct opportunities in a position.

        Returns list sorted by priority (highest first).
        """
        results = []

        # Check each instinct
        detectors = [
            self.detect_extend_from_atari,
            self.detect_connect_vs_peep,
            self.detect_block_the_thrust,
            self.detect_hane_vs_tsuke,
            self.detect_hane_at_head_of_two,
            self.detect_stretch_from_kosumi,
            self.detect_block_the_angle,
            self.detect_stretch_from_bump,
        ]

        for detector in detectors:
            result = detector(board)
            if result:
                results.append(result)

        # Sort by priority
        results.sort(key=lambda x: x.priority, reverse=True)

        return results


if __name__ == '__main__':
    # Quick test
    board = Board(9)
    c = 4

    # Create atari position
    board.board[c, c] = 1      # Black
    board.board[c-1, c] = -1   # White surrounding
    board.board[c+1, c] = -1
    board.board[c, c-1] = -1
    # Liberty at (c, c+1)
    board.current_player = 1

    detector = SenseiInstinctDetector()

    print("Board:")
    print(board)
    print()

    result = detector.detect_extend_from_atari(board)
    if result:
        print(f"Extend from atari: {result.moves}")

    all_results = detector.detect_all(board)
    print(f"\nAll instincts detected: {len(all_results)}")
    for r in all_results:
        print(f"  {r.instinct}: {r.moves} (priority={r.priority})")
