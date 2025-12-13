"""The 8 Basic Instincts from Sensei's Library.

These are fundamental Go responses that strong players execute instinctively.
Encoding them symbolically provides the neural network with tactical guidance
that it struggles to learn from pattern matching alone.

Reference: https://senseis.xmp.net/?BasicInstinct

The 8 Instincts:
1. From an atari, extend
2. Answer the tsuke with a hane
3. Hane at the head of two stones
4. Stretch from a kosumi-tsuke
5. Block the angle play
6. Connect against a peep
7. Block the thrust
8. Stretch from a bump
"""

import numpy as np
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass
from board import Board


@dataclass
class Instinct:
    """Result of instinct detection."""
    name: str
    move: Tuple[int, int]
    boost: float
    reason: str


class InstinctAnalyzer:
    """Detects the 8 basic Go instincts in positions.

    Each instinct provides a boost multiplier for moves that follow
    these fundamental Go principles.
    """

    # Boost values for each instinct
    BOOSTS = {
        'extend_from_atari': 3.0,      # Critical - save stones
        'hane_response': 1.5,          # Good shape response
        'hane_at_head_of_two': 2.0,    # Strong attacking move
        'stretch_from_kosumi': 1.4,    # Solid response
        'block_angle': 1.5,            # Defensive necessity
        'connect_against_peep': 2.5,   # "Even a moron connects"
        'block_thrust': 1.8,           # Maintain structure
        'stretch_from_bump': 1.3,      # Correct response
    }

    def __init__(self):
        # Direction vectors for adjacency
        self.ORTHOGONAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.DIAGONAL = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        self.ALL_DIRS = self.ORTHOGONAL + self.DIAGONAL

    def _in_bounds(self, board: Board, r: int, c: int) -> bool:
        return 0 <= r < board.size and 0 <= c < board.size

    def _get_adjacent(self, board: Board, r: int, c: int,
                      dirs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get adjacent positions in given directions."""
        return [(r + dr, c + dc) for dr, dc in dirs
                if self._in_bounds(board, r + dr, c + dc)]

    def _get_color_at(self, board: Board, pos: Tuple[int, int]) -> int:
        """Get stone color at position (0=empty, 1=black, -1=white)."""
        return board.board[pos[0], pos[1]]

    def _find_groups_in_atari(self, board: Board, color: int) -> List[Set[Tuple[int, int]]]:
        """Find all groups of given color that are in atari (1 liberty)."""
        groups = []
        visited = set()

        for r in range(board.size):
            for c in range(board.size):
                if (r, c) in visited or board.board[r, c] != color:
                    continue

                group = board.get_group(r, c)
                visited.update(group)

                if board.count_liberties(group) == 1:
                    groups.append(group)

        return groups

    def _get_liberties(self, board: Board, group: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Get all liberty positions for a group."""
        liberties = set()
        for r, c in group:
            for nr, nc in self._get_adjacent(board, r, c, self.ORTHOGONAL):
                if board.board[nr, nc] == 0:
                    liberties.add((nr, nc))
        return liberties

    # =========================================================================
    # 1. FROM AN ATARI, EXTEND
    # =========================================================================
    def detect_extend_from_atari(self, board: Board, move: Tuple[int, int]) -> bool:
        """Detect if move extends a group in atari."""
        player = board.current_player

        # Find player's groups in atari
        atari_groups = self._find_groups_in_atari(board, player)

        for group in atari_groups:
            liberties = self._get_liberties(board, group)
            if move in liberties:
                # This move extends the atari'd group
                return True

        return False

    # =========================================================================
    # 2. ANSWER THE TSUKE WITH A HANE
    # =========================================================================
    def detect_hane_response(self, board: Board, move: Tuple[int, int]) -> bool:
        """Detect if move is a hane response to opponent's tsuke.

        Tsuke: opponent plays contact (adjacent) to our stone.
        Hane: we wrap around (orthogonal to opponent, diagonal to our stone).

        Pattern:
            . H .      H = hane move (orthogonal to W, diagonal to B)
            B W .      B = our stone, W = opponent's tsuke
            . . .
        """
        player = board.current_player
        opponent = -player
        mr, mc = move

        if board.board[mr, mc] != 0:
            return False

        # Hane: orthogonally adjacent to opponent, diagonal to our stone
        for dr, dc in self.ORTHOGONAL:
            opp_r, opp_c = mr + dr, mc + dc
            if not self._in_bounds(board, opp_r, opp_c):
                continue
            if board.board[opp_r, opp_c] != opponent:
                continue

            # Found opponent stone orthogonally adjacent to move
            # Check if our stone is diagonal to move (creating hane shape)
            for ddr, ddc in self.DIAGONAL:
                our_r, our_c = mr + ddr, mc + ddc
                if not self._in_bounds(board, our_r, our_c):
                    continue
                if board.board[our_r, our_c] == player:
                    # Additionally verify opponent is adjacent to our stone (tsuke)
                    if abs(opp_r - our_r) + abs(opp_c - our_c) == 1:
                        return True

        return False

    # =========================================================================
    # 3. HANE AT THE HEAD OF TWO STONES
    # =========================================================================
    def detect_hane_at_head_of_two(self, board: Board, move: Tuple[int, int]) -> bool:
        """Detect if move is at the head of two consecutive opponent stones."""
        opponent = -board.current_player
        mr, mc = move

        if board.board[mr, mc] != 0:
            return False

        # Check vertical two stones
        # Move at (mr, mc), stones at (mr+1, mc) and (mr+2, mc)
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            s1_r, s1_c = mr + dr, mc + dc
            s2_r, s2_c = mr + 2*dr, mc + 2*dc

            if not self._in_bounds(board, s1_r, s1_c):
                continue
            if not self._in_bounds(board, s2_r, s2_c):
                continue

            if (board.board[s1_r, s1_c] == opponent and
                board.board[s2_r, s2_c] == opponent):
                return True

        return False

    # =========================================================================
    # 4. STRETCH FROM A KOSUMI-TSUKE
    # =========================================================================
    def detect_stretch_from_kosumi(self, board: Board, move: Tuple[int, int]) -> bool:
        """Detect if move stretches away from opponent's diagonal contact."""
        player = board.current_player
        opponent = -player
        mr, mc = move

        if board.board[mr, mc] != 0:
            return False

        # Look for our stone with opponent's diagonal contact
        for dr, dc in self.ORTHOGONAL:
            our_r, our_c = mr + dr, mc + dc
            if not self._in_bounds(board, our_r, our_c):
                continue
            if board.board[our_r, our_c] != player:
                continue

            # Found our stone - check for opponent's kosumi (diagonal)
            for ddr, ddc in self.DIAGONAL:
                diag_r, diag_c = our_r + ddr, our_c + ddc
                if not self._in_bounds(board, diag_r, diag_c):
                    continue
                if board.board[diag_r, diag_c] == opponent:
                    # Kosumi-tsuke exists - is our move stretching away?
                    # Move should be away from the diagonal threat
                    if self._is_stretch_away(move, (our_r, our_c), (diag_r, diag_c)):
                        return True

        return False

    def _is_stretch_away(self, move: Tuple[int, int], our_pos: Tuple[int, int],
                         threat_pos: Tuple[int, int]) -> bool:
        """Check if move stretches away from threat."""
        mr, mc = move
        our_r, our_c = our_pos
        threat_r, threat_c = threat_pos

        # Direction from our stone to threat
        threat_dr = 1 if threat_r > our_r else (-1 if threat_r < our_r else 0)
        threat_dc = 1 if threat_c > our_c else (-1 if threat_c < our_c else 0)

        # Direction from our stone to move
        move_dr = mr - our_r
        move_dc = mc - our_c

        # Move should be in opposite direction (away from threat)
        return (move_dr * threat_dr <= 0 and move_dc * threat_dc <= 0 and
                (move_dr != 0 or move_dc != 0))

    # =========================================================================
    # 5. BLOCK THE ANGLE PLAY
    # =========================================================================
    def detect_block_angle(self, board: Board, move: Tuple[int, int]) -> bool:
        """Detect if move blocks opponent's angle play (diagonal attack)."""
        player = board.current_player
        opponent = -player
        mr, mc = move

        if board.board[mr, mc] != 0:
            return False

        # Check if move is between our stone and opponent's angle stone
        for dr, dc in self.DIAGONAL:
            # Our stone at one diagonal
            our_r, our_c = mr - dr, mc - dc
            # Opponent at further diagonal
            opp_r, opp_c = mr + dr, mc + dc

            if not self._in_bounds(board, our_r, our_c):
                continue
            if not self._in_bounds(board, opp_r, opp_c):
                continue

            if (board.board[our_r, our_c] == player and
                board.board[opp_r, opp_c] == opponent):
                return True

        return False

    # =========================================================================
    # 6. CONNECT AGAINST A PEEP
    # =========================================================================
    def find_peeps(self, board: Board) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Find all peep positions (opponent stone threatening to cut).

        Returns list of (peep_stone, cutting_point) tuples.
        """
        player = board.current_player
        opponent = -player
        peeps = []

        # Find gaps in player's formation that opponent threatens
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != player:
                    continue

                # Check for one-point jump pattern (stones with gap)
                for dr, dc in [(0, 2), (2, 0)]:  # Horizontal and vertical jumps
                    far_r, far_c = r + dr, c + dc
                    mid_r, mid_c = r + dr//2, c + dc//2

                    if not self._in_bounds(board, far_r, far_c):
                        continue

                    if board.board[far_r, far_c] != player:
                        continue

                    if board.board[mid_r, mid_c] != 0:
                        continue

                    # Found gap - check for peep (opponent adjacent to cutting point)
                    for pdr, pdc in self.ORTHOGONAL:
                        peep_r, peep_c = mid_r + pdr, mid_c + pdc
                        if not self._in_bounds(board, peep_r, peep_c):
                            continue
                        if board.board[peep_r, peep_c] == opponent:
                            peeps.append(((peep_r, peep_c), (mid_r, mid_c)))

        return peeps

    def detect_connect_against_peep(self, board: Board, move: Tuple[int, int]) -> bool:
        """Detect if move connects against a peep."""
        peeps = self.find_peeps(board)

        for peep_stone, cutting_point in peeps:
            if move == cutting_point:
                return True

        return False

    # =========================================================================
    # 7. BLOCK THE THRUST
    # =========================================================================
    def detect_block_thrust(self, board: Board, move: Tuple[int, int]) -> bool:
        """Detect if move blocks opponent's thrust into our formation."""
        player = board.current_player
        opponent = -player
        mr, mc = move

        if board.board[mr, mc] != 0:
            return False

        # Look for opponent stone pushing into our structure
        for dr, dc in self.ORTHOGONAL:
            opp_r, opp_c = mr + dr, mc + dc
            if not self._in_bounds(board, opp_r, opp_c):
                continue
            if board.board[opp_r, opp_c] != opponent:
                continue

            # Found opponent - check if we have stones on both sides
            perp_dirs = [(dc, dr), (-dc, -dr)] if dr == 0 else [(-dr, dc), (dr, -dc)]
            our_stones = 0
            for pdr, pdc in perp_dirs:
                check_r, check_c = mr + pdr, mc + pdc
                if self._in_bounds(board, check_r, check_c):
                    if board.board[check_r, check_c] == player:
                        our_stones += 1

            if our_stones >= 1:
                # We have structure - this blocks the thrust
                return True

        return False

    # =========================================================================
    # 8. STRETCH FROM A BUMP
    # =========================================================================
    def detect_stretch_from_bump(self, board: Board, move: Tuple[int, int]) -> bool:
        """Detect if move stretches from a bump (when opponent is reinforced)."""
        player = board.current_player
        opponent = -player
        mr, mc = move

        if board.board[mr, mc] != 0:
            return False

        # Look for our stone that was bumped by reinforced opponent
        for dr, dc in self.ORTHOGONAL:
            our_r, our_c = mr + dr, mc + dc
            if not self._in_bounds(board, our_r, our_c):
                continue
            if board.board[our_r, our_c] != player:
                continue

            # Found our stone - check for opponent bump with backup
            for bdr, bdc in self.ORTHOGONAL:
                if (bdr, bdc) == (-dr, -dc):
                    continue  # Skip opposite direction

                bump_r, bump_c = our_r + bdr, our_c + bdc
                if not self._in_bounds(board, bump_r, bump_c):
                    continue
                if board.board[bump_r, bump_c] != opponent:
                    continue

                # Found bump - check if opponent has backup (reinforced)
                for rdr, rdc in self.ORTHOGONAL:
                    reinf_r, reinf_c = bump_r + rdr, bump_c + rdc
                    if (reinf_r, reinf_c) == (our_r, our_c):
                        continue
                    if not self._in_bounds(board, reinf_r, reinf_c):
                        continue
                    if board.board[reinf_r, reinf_c] == opponent:
                        # Opponent is reinforced - stretch is correct
                        return True

        return False

    # =========================================================================
    # UNIFIED BOOST CALCULATION
    # =========================================================================
    def get_instinct_boost(self, board: Board, move: Tuple[int, int]) -> float:
        """Get combined boost for a move based on all applicable instincts.

        Returns multiplicative boost (1.0 = no boost).
        """
        if not self._in_bounds(board, move[0], move[1]):
            return 1.0

        boost = 1.0
        detected = []

        # Check each instinct
        if self.detect_extend_from_atari(board, move):
            boost *= self.BOOSTS['extend_from_atari']
            detected.append('extend_from_atari')

        if self.detect_hane_response(board, move):
            boost *= self.BOOSTS['hane_response']
            detected.append('hane_response')

        if self.detect_hane_at_head_of_two(board, move):
            boost *= self.BOOSTS['hane_at_head_of_two']
            detected.append('hane_at_head_of_two')

        if self.detect_stretch_from_kosumi(board, move):
            boost *= self.BOOSTS['stretch_from_kosumi']
            detected.append('stretch_from_kosumi')

        if self.detect_block_angle(board, move):
            boost *= self.BOOSTS['block_angle']
            detected.append('block_angle')

        if self.detect_connect_against_peep(board, move):
            boost *= self.BOOSTS['connect_against_peep']
            detected.append('connect_against_peep')

        if self.detect_block_thrust(board, move):
            boost *= self.BOOSTS['block_thrust']
            detected.append('block_thrust')

        if self.detect_stretch_from_bump(board, move):
            boost *= self.BOOSTS['stretch_from_bump']
            detected.append('stretch_from_bump')

        return boost

    def analyze_position(self, board: Board) -> List[Instinct]:
        """Analyze position and return all detected instincts with moves."""
        instincts = []

        for r in range(board.size):
            for c in range(board.size):
                move = (r, c)
                if board.board[r, c] != 0:
                    continue

                boost = self.get_instinct_boost(board, move)
                if boost > 1.0:
                    # Determine which instinct(s) triggered
                    reasons = []
                    if self.detect_extend_from_atari(board, move):
                        reasons.append("extend from atari")
                    if self.detect_connect_against_peep(board, move):
                        reasons.append("connect against peep")
                    if self.detect_hane_at_head_of_two(board, move):
                        reasons.append("hane at head of two")
                    if self.detect_hane_response(board, move):
                        reasons.append("hane response to tsuke")
                    if self.detect_block_thrust(board, move):
                        reasons.append("block thrust")
                    if self.detect_block_angle(board, move):
                        reasons.append("block angle")
                    if self.detect_stretch_from_kosumi(board, move):
                        reasons.append("stretch from kosumi")
                    if self.detect_stretch_from_bump(board, move):
                        reasons.append("stretch from bump")

                    instincts.append(Instinct(
                        name=reasons[0] if reasons else "instinct",
                        move=move,
                        boost=boost,
                        reason=", ".join(reasons)
                    ))

        return instincts
