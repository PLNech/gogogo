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
        """Detect two consecutive opponent stones where we can play at the head.

        Play above two consecutive opponent stones to create weakness.
        Must be exactly TWO stones (not three or more).
        """
        player = board.current_player
        opponent = -player
        head_moves = []

        # Find pairs of opponent stones
        visited_pairs: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != opponent:
                    continue

                # Check for adjacent opponent stone (horizontal or vertical)
                for dr, dc in [(0, 1), (1, 0)]:  # Only check right and down to avoid duplicates
                    nr, nc = r + dr, c + dc
                    if not self._in_bounds(board, nr, nc):
                        continue
                    if board.board[nr, nc] != opponent:
                        continue

                    # Found a pair! Check it's exactly two (no third stone continuing)
                    # Check both ends
                    prev_r, prev_c = r - dr, c - dc
                    next_r, next_c = nr + dr, nc + dc

                    has_prev = (self._in_bounds(board, prev_r, prev_c) and
                               board.board[prev_r, prev_c] == opponent)
                    has_next = (self._in_bounds(board, next_r, next_c) and
                               board.board[next_r, next_c] == opponent)

                    if has_prev or has_next:
                        # Part of a longer chain, not "head of two"
                        continue

                    # Valid pair! Head positions are at both ends
                    pair = ((r, c), (nr, nc))
                    if pair in visited_pairs:
                        continue
                    visited_pairs.add(pair)

                    # Head at the "next" end
                    if self._in_bounds(board, next_r, next_c) and board.board[next_r, next_c] == 0:
                        head_moves.append((next_r, next_c))

                    # Head at the "prev" end
                    if self._in_bounds(board, prev_r, prev_c) and board.board[prev_r, prev_c] == 0:
                        head_moves.append((prev_r, prev_c))

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

    def detect_block_the_angle(self, board: Board) -> Optional[InstinctResult]:
        """Detect opponent angle attack (kake) and find block moves.

        Respond to diagonal threats by blocking.
        """
        player = board.current_player
        opponent = -player
        block_moves = []

        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != player:
                    continue

                # Check for diagonal opponent (angle attack)
                for dr, dc in self.DIAGONALS:
                    nr, nc = r + dr, c + dc
                    if not self._in_bounds(board, nr, nc):
                        continue
                    if board.board[nr, nc] != opponent:
                        continue

                    # Found angle attack! Block positions are orthogonally adjacent
                    # to our stone, toward the attacker
                    block_candidates = [(r + dr, c), (r, c + dc)]
                    for br, bc in block_candidates:
                        if self._in_bounds(board, br, bc) and board.board[br, bc] == 0:
                            block_moves.append((br, bc))

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
        """Detect opponent thrust between our stones.

        When opponent thrusts between your stones, block it.
        """
        player = board.current_player
        opponent = -player
        block_moves = []

        # Find empty points between our stones where opponent threatens
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != 0:
                    continue

                # Check if this empty point has our stones on opposite sides
                # and opponent threatening from another direction

                # Horizontal check: our stones at (r, c-1) and (r, c+1)
                if (self._in_bounds(board, r, c-1) and self._in_bounds(board, r, c+1) and
                    board.board[r, c-1] == player and board.board[r, c+1] == player):
                    # Check for opponent thrust from above or below
                    for thrust_r in [r-1, r+1]:
                        if self._in_bounds(board, thrust_r, c) and board.board[thrust_r, c] == opponent:
                            block_moves.append((r, c))
                            break

                # Vertical check: our stones at (r-1, c) and (r+1, c)
                if (self._in_bounds(board, r-1, c) and self._in_bounds(board, r+1, c) and
                    board.board[r-1, c] == player and board.board[r+1, c] == player):
                    # Check for opponent thrust from left or right
                    for thrust_c in [c-1, c+1]:
                        if self._in_bounds(board, r, thrust_c) and board.board[r, thrust_c] == opponent:
                            block_moves.append((r, c))
                            break

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
