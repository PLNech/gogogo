"""Symbolic tactical analysis for Go positions.

This module provides deterministic tactical calculations that complement
neural network pattern recognition. While NNs excel at strategic patterns,
tactical sequences (ladders, snapbacks, capture races) require concrete
calculation to verify outcomes.

Usage:
    from tactics import TacticalAnalyzer
    tactics = TacticalAnalyzer()

    # Check if move creates working ladder
    is_ladder = tactics.trace_ladder(board, group)

    # Find snapback opportunities
    captures = tactics.detect_snapback(board, move)

    # Verify capture sequence depth-first
    result = tactics.verify_capture(board, move, depth=4)
"""

import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass
from board import Board


@dataclass
class TacticalResult:
    """Result of tactical analysis."""
    move: Tuple[int, int]
    score: float  # Positive = good for current player
    captures: int  # Stones captured
    sequence: List[Tuple[int, int]]  # Principal variation
    confidence: float  # 0-1, how certain is the result


class TacticalAnalyzer:
    """Symbolic tactical analyzer for Go.

    Provides deterministic answers for tactical questions:
    - Is this group in a working ladder?
    - Can I capture this group in N moves?
    - Is this a snapback?
    - Is this group alive/dead?
    """

    def __init__(self, max_ladder_length: int = 50):
        """Initialize analyzer.

        Args:
            max_ladder_length: Maximum moves to trace ladder (default 50)
        """
        self.max_ladder_length = max_ladder_length
        # Cache for ladder results (position_hash -> result)
        self._ladder_cache: Dict[int, Optional[bool]] = {}

    def clear_cache(self):
        """Clear all caches."""
        self._ladder_cache.clear()

    # =========================================================================
    # LADDER DETECTION
    # =========================================================================

    def trace_ladder(
        self,
        board: Board,
        group: List[Tuple[int, int]],
        attacker: int = None
    ) -> Optional[bool]:
        """Trace a ladder to determine if group is captured.

        Uses DIAGONAL SCAN algorithm (from SNAP.md):
        1. Determine ladder escape direction
        2. Scan diagonal for defender stones (breakers)
        3. If no breaker before edge → ladder works

        Falls back to recursive simulation for complex cases.

        Args:
            board: Current board state
            group: Group to check (list of positions)
            attacker: Color attacking (default: opponent of group)

        Returns:
            True if group is captured (ladder works)
            False if group escapes (ladder broken)
            None if unclear (not a simple ladder)
        """
        if not group:
            return None

        # Get colors
        defender = board.board[group[0][0], group[0][1]]
        if defender == 0:
            return None
        attacker = attacker or -defender

        # Must start in atari
        liberties = board.count_liberties(group)
        if liberties != 1:
            return None  # Not in atari, not a ladder situation

        # Check cache
        cache_key = (board.zobrist_hash(), tuple(sorted(group)))
        if cache_key in self._ladder_cache:
            return self._ladder_cache[cache_key]

        # Try fast diagonal scan first
        result = self._diagonal_ladder_check(board, group, attacker, defender)

        # Fall back to recursive if unclear
        if result is None:
            result = self._trace_ladder_recursive(
                board.copy(), group, attacker, defender, depth=0
            )

        self._ladder_cache[cache_key] = result
        return result

    def _diagonal_ladder_check(
        self,
        board: Board,
        group: List[Tuple[int, int]],
        attacker: int,
        defender: int
    ) -> Optional[bool]:
        """Fast diagonal scan for ladder breakers.

        Ladders run diagonally. If defender has a stone on the escape diagonal,
        the ladder is broken. If the diagonal reaches the edge with no breaker,
        the ladder works.

        Returns:
            True = captured (no breaker, reaches edge)
            False = escapes (breaker found)
            None = unclear (complex position, need simulation)
        """
        # Find the single liberty (escape direction)
        liberty = self._get_single_liberty(board, group)
        if liberty is None:
            return None

        lib_r, lib_c = liberty

        # Determine ladder direction from attacker stone positions
        # Ladder runs perpendicular to the line of attackers
        attacker_positions = []
        for r, c in group:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board.size and 0 <= nc < board.size:
                    if board.board[nr, nc] == attacker:
                        attacker_positions.append((nr, nc))

        if len(attacker_positions) < 2:
            return None  # Need at least 2 attackers for standard ladder

        # Calculate center of group
        group_r = sum(r for r, c in group) / len(group)
        group_c = sum(c for r, c in group) / len(group)

        # Determine escape direction based on liberty position relative to group
        # Liberty direction gives primary escape direction
        lib_dr = lib_r - group_r  # Positive = down, negative = up
        lib_dc = lib_c - group_c  # Positive = right, negative = left

        # Diagonal escape is perpendicular to attackers
        # Find which diagonal quadrant is away from most attackers
        atk_above = sum(1 for r, c in attacker_positions if r < group_r)
        atk_below = sum(1 for r, c in attacker_positions if r > group_r)
        atk_left = sum(1 for r, c in attacker_positions if c < group_c)
        atk_right = sum(1 for r, c in attacker_positions if c > group_c)

        # Escape AWAY from attackers
        dr = -1 if atk_below > atk_above else (1 if atk_above > atk_below else int(lib_dr) if lib_dr != 0 else -1)
        dc = -1 if atk_right > atk_left else (1 if atk_left > atk_right else int(lib_dc) if lib_dc != 0 else -1)

        # Scan diagonal for breakers from the liberty position
        r, c = lib_r, lib_c
        steps = 0
        max_steps = board.size * 2  # Safety limit

        while steps < max_steps:
            # Move diagonally
            r += dr
            c += dc
            steps += 1

            # Check bounds
            if r < 0 or r >= board.size or c < 0 or c >= board.size:
                # Reached edge - ladder works!
                return True

            # Check for breaker stone (defender's stone on escape path)
            if board.board[r, c] == defender:
                return False  # Ladder broken!

            # Check for attacker stone blocking escape
            if board.board[r, c] == attacker:
                # Attacker already there - need simulation
                continue

        return None  # Unclear, use simulation

    def _trace_ladder_recursive(
        self,
        board: Board,
        group: List[Tuple[int, int]],
        attacker: int,
        defender: int,
        depth: int
    ) -> Optional[bool]:
        """Recursive ladder tracing.

        Returns:
            True = captured, False = escapes, None = unclear
        """
        if depth > self.max_ladder_length:
            return None  # Too long, unclear

        # Find the single liberty
        liberty = self._get_single_liberty(board, group)
        if liberty is None:
            return None  # Should not happen if in atari

        # Defender extends (plays on liberty)
        test_board = board.copy()
        try:
            test_board.board[liberty[0], liberty[1]] = defender
            # Don't actually call play() to avoid turn switching
        except:
            return True  # Can't extend = captured

        # Update group with new stone
        new_group = test_board.get_group(liberty[0], liberty[1])
        new_libs = test_board.count_liberties(new_group)

        # Escaped! More than 2 liberties means ladder broken
        if new_libs >= 3:
            return False

        # Exactly 2 liberties - attacker must continue chase
        if new_libs == 2:
            # Find the two liberties
            lib_positions = self._get_liberties(test_board, new_group)
            if len(lib_positions) != 2:
                return None

            # Attacker plays on one liberty, defender on the other
            # Try both orders to see if ladder works
            for atk_lib in lib_positions:
                chase_board = test_board.copy()
                chase_board.board[atk_lib[0], atk_lib[1]] = attacker

                # Now defender's group should be in atari again
                chase_group = chase_board.get_group(liberty[0], liberty[1])
                chase_libs = chase_board.count_liberties(chase_group)

                if chase_libs == 0:
                    # Captured!
                    return True
                elif chase_libs == 1:
                    # Continue chase
                    result = self._trace_ladder_recursive(
                        chase_board, chase_group, attacker, defender, depth + 1
                    )
                    if result is True:
                        return True  # Found working ladder
                    # If this attack doesn't work, try other liberty
                else:
                    # Defender has too many liberties
                    continue

            # Neither attack worked
            return False

        # 1 liberty after extending - still in atari, continue
        elif new_libs == 1:
            # Attacker continues
            return self._trace_ladder_recursive(
                test_board, new_group, attacker, defender, depth + 1
            )

        # 0 liberties - captured!
        return True

    def _get_single_liberty(
        self,
        board: Board,
        group: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """Get the single liberty of a group in atari."""
        for r, c in group:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board.size and 0 <= nc < board.size:
                    if board.board[nr, nc] == 0:
                        return (nr, nc)
        return None

    def _get_liberties(
        self,
        board: Board,
        group: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Get all liberties of a group."""
        liberties = set()
        for r, c in group:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board.size and 0 <= nc < board.size:
                    if board.board[nr, nc] == 0:
                        liberties.add((nr, nc))
        return list(liberties)

    # =========================================================================
    # SNAPBACK DETECTION
    # =========================================================================

    def detect_snapback(
        self,
        board: Board,
        move: Tuple[int, int]
    ) -> int:
        """Detect if move creates a snapback opportunity.

        Snapback: throw-in move that gets captured, but the capture
        leaves the capturing group with only 1 liberty, allowing
        immediate recapture of a larger group.

        ATARI-FIRST ALGORITHM (from SNAP.md):
        1. Check if move is the ONLY liberty of an opponent group (group in atari)
        2. Simulate: we play there, get captured
        3. Check if opponent group now has exactly 1 liberty (at our throw-in point)
        4. If yes -> SNAPBACK

        Args:
            board: Current board state
            move: Move to analyze

        Returns:
            Number of stones captured by snapback (0 if not snapback)
        """
        if board.board[move[0], move[1]] != 0:
            return 0

        player = board.current_player
        opponent = -player

        # ATARI-FIRST: Find opponent groups in ATARI where move is their only liberty
        atari_groups = []
        seen_groups = set()

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = move[0] + dr, move[1] + dc
            if 0 <= nr < board.size and 0 <= nc < board.size:
                if board.board[nr, nc] == opponent and (nr, nc) not in seen_groups:
                    opp_group = board.get_group(nr, nc)
                    seen_groups.update(opp_group)
                    opp_libs = board.count_liberties(opp_group)

                    # Group in atari - move is their only liberty
                    if opp_libs == 1:
                        atari_groups.append((opp_group, nr, nc))

        # Also check for groups with 2 libs (we reduce to 1, they capture, we recapture)
        # This is the "delayed snapback" pattern
        seen_groups.clear()
        two_lib_groups = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = move[0] + dr, move[1] + dc
            if 0 <= nr < board.size and 0 <= nc < board.size:
                if board.board[nr, nc] == opponent and (nr, nc) not in seen_groups:
                    opp_group = board.get_group(nr, nc)
                    seen_groups.update(opp_group)
                    opp_libs = board.count_liberties(opp_group)

                    if opp_libs == 2:
                        two_lib_groups.append((opp_group, nr, nc))

        if not atari_groups and not two_lib_groups:
            return 0

        # Simulate the move
        test = board.copy()
        test.board[move[0], move[1]] = player

        # Check if our stone will be captured (0 liberties = suicide/throw-in)
        my_group = test.get_group(move[0], move[1])
        my_libs = test.count_liberties(my_group)

        # Case 1: Atari groups - we're filling their last liberty
        # After we play, they capture us, but then have only 1 lib at the throw-in
        for opp_group, nr, nc in atari_groups:
            # Our stone fills their liberty, we get captured
            # After capture, the throw-in point is now opponent's only liberty
            # Simulate capture
            after_capture = board.copy()  # Start fresh
            after_capture.board[move[0], move[1]] = 0  # We got captured

            # Check opponent's liberties after they captured us
            opp_group_after = after_capture.get_group(nr, nc)
            opp_libs_after = after_capture.count_liberties(opp_group_after)

            # The throw-in point should be their only liberty
            if opp_libs_after == 1:
                only_lib = self._get_single_liberty(after_capture, opp_group_after)
                if only_lib == move:
                    # Perfect snapback! We can recapture immediately
                    return len(opp_group_after)

        # Case 2: Two-lib groups - we reduce to 1, get captured, they have 1 lib
        if my_libs == 0 or my_libs == 1:  # We'll be captured
            for opp_group, nr, nc in two_lib_groups:
                # Simulate: we get captured
                after_capture = test.copy()
                for gr, gc in my_group:
                    after_capture.board[gr, gc] = 0

                # Check opponent's liberties after capture
                opp_group_after = after_capture.get_group(nr, nc)
                opp_libs_after = after_capture.count_liberties(opp_group_after)

                if opp_libs_after == 1:
                    # Snapback! We can recapture
                    return len(opp_group_after)

        return 0

    # =========================================================================
    # CAPTURE SEQUENCE VERIFICATION
    # =========================================================================

    def verify_capture(
        self,
        board: Board,
        move: Tuple[int, int],
        depth: int = 4
    ) -> TacticalResult:
        """Verify capture sequence with alpha-beta search.

        Does a minimax search to verify if a capture sequence works.

        Args:
            board: Current board state
            move: Initial move to analyze
            depth: Search depth

        Returns:
            TacticalResult with score and principal variation
        """
        if board.board[move[0], move[1]] != 0:
            return TacticalResult(move, 0, 0, [], 0)

        player = board.current_player
        test = board.copy()

        try:
            captures = test.play(move[0], move[1])
        except:
            return TacticalResult(move, -100, 0, [], 1.0)

        if captures > 0:
            # Immediate capture - verify it's not recapturable (ko check)
            if test.ko_point is not None:
                # Ko - opponent can recapture
                return TacticalResult(move, captures * 0.5, captures, [move], 0.5)
            return TacticalResult(move, captures * 2, captures, [move], 1.0)

        # No immediate capture - search deeper
        result = self._capture_search(test, depth - 1, -1000, 1000, [move])

        return TacticalResult(
            move=move,
            score=result[0],
            captures=result[1],
            sequence=result[2],
            confidence=0.8 if depth >= 4 else 0.5
        )

    def _capture_search(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        pv: List[Tuple[int, int]]
    ) -> Tuple[float, int, List[Tuple[int, int]]]:
        """Alpha-beta search for captures.

        Returns (score, total_captures, principal_variation)
        """
        if depth <= 0 or board.is_game_over():
            return (0, 0, pv)

        player = board.current_player
        best_score = -1000
        best_captures = 0
        best_pv = pv

        # Generate tactical moves (captures and atari escapes)
        moves = self._get_tactical_moves(board)

        for move in moves:
            test = board.copy()
            try:
                captures = test.play(move[0], move[1])
            except:
                continue

            # Score this move
            move_score = captures * 2  # Captures are good

            # Recurse
            child_result = self._capture_search(
                test, depth - 1, -beta, -alpha, pv + [move]
            )
            child_score = -child_result[0]
            total_captures = captures + child_result[1]

            score = move_score + child_score * 0.9  # Discount future

            if score > best_score:
                best_score = score
                best_captures = total_captures
                best_pv = child_result[2]

            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Pruning

        return (best_score, best_captures, best_pv)

    def _get_tactical_moves(self, board: Board) -> List[Tuple[int, int]]:
        """Get high-priority tactical moves.

        Returns moves that:
        - Capture opponent stones
        - Save own groups in atari
        - Put opponent in atari
        """
        moves = []
        player = board.current_player
        opponent = -player

        # Find capture moves and atari escapes
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != 0:
                    continue

                score = 0

                # Check adjacent groups
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board.size and 0 <= nc < board.size:
                        adj_color = board.board[nr, nc]
                        if adj_color == opponent:
                            group = board.get_group(nr, nc)
                            libs = board.count_liberties(group)
                            if libs == 1:
                                score += 10 * len(group)  # Capture!
                            elif libs == 2:
                                score += 3  # Atari threat
                        elif adj_color == player:
                            group = board.get_group(nr, nc)
                            libs = board.count_liberties(group)
                            if libs == 1:
                                score += 8 * len(group)  # Save group

                if score > 0:
                    moves.append((r, c, score))

        # Sort by score descending
        moves.sort(key=lambda x: -x[2])

        # Return top N moves (limit branching)
        return [(m[0], m[1]) for m in moves[:10]]

    # =========================================================================
    # LIFE/DEATH ANALYSIS
    # =========================================================================

    def evaluate_life_death(
        self,
        board: Board,
        group: List[Tuple[int, int]],
        depth: int = 6
    ) -> float:
        """Evaluate if a group is alive, dead, or unsettled.

        Uses minimax search to determine life/death status.

        Args:
            board: Current board state
            group: Group to evaluate
            depth: Search depth

        Returns:
            +1.0 = alive (two eyes or escape)
            -1.0 = dead (no way to live)
            0.0 = unsettled (depends on who plays first)
        """
        if not group:
            return 0.0

        color = board.board[group[0][0], group[0][1]]
        if color == 0:
            return 0.0

        # Quick checks
        eyes = board.count_eyes(group)
        if eyes >= 2:
            return 1.0  # Unconditionally alive

        liberties = board.count_liberties(group)
        if liberties == 0:
            return -1.0  # Already captured

        # Large groups with many liberties are usually alive
        if len(group) >= 8 and liberties >= 6:
            return 0.9

        # Do life/death search
        # Attacker (opponent) tries to kill, defender tries to live
        attacker = -color

        # Attacker moves first (worst case for group)
        if board.current_player == attacker:
            score = self._life_death_search(board, group, depth, True)
        else:
            # Defender moves first (best case)
            score = self._life_death_search(board, group, depth, False)

        return score

    def _life_death_search(
        self,
        board: Board,
        group: List[Tuple[int, int]],
        depth: int,
        attacker_to_move: bool
    ) -> float:
        """Minimax search for life/death.

        Returns:
            +1.0 = alive, -1.0 = dead, intermediate = unsettled
        """
        if depth <= 0:
            # Evaluate current position
            return self._static_life_eval(board, group)

        color = board.board[group[0][0], group[0][1]]

        # Check if group still exists
        if not self._group_exists(board, group, color):
            return -1.0  # Captured

        # Check for two eyes
        eyes = board.count_eyes(group)
        if eyes >= 2:
            return 1.0  # Alive

        # Get vital points (liberties and eye-making points)
        vital_points = self._get_vital_points(board, group)

        if attacker_to_move:
            # Attacker tries to minimize (kill)
            best_score = 1.0  # Start optimistic for defender
            for move in vital_points:
                test = board.copy()
                test.current_player = -color  # Attacker
                try:
                    test.play(move[0], move[1])
                except:
                    continue

                # Check if group captured
                if not self._group_exists(test, group, color):
                    return -1.0  # Killed

                new_group = self._find_group_after_move(test, group, color)
                if new_group:
                    score = self._life_death_search(test, new_group, depth - 1, False)
                    best_score = min(best_score, score)

            return best_score
        else:
            # Defender tries to maximize (live)
            best_score = -1.0  # Start pessimistic
            for move in vital_points:
                test = board.copy()
                test.current_player = color  # Defender
                try:
                    test.play(move[0], move[1])
                except:
                    continue

                new_group = self._find_group_after_move(test, group, color)
                if new_group:
                    eyes = test.count_eyes(new_group)
                    if eyes >= 2:
                        return 1.0  # Made two eyes

                    score = self._life_death_search(test, new_group, depth - 1, True)
                    best_score = max(best_score, score)

            return best_score

    def _group_exists(
        self,
        board: Board,
        original_group: List[Tuple[int, int]],
        color: int
    ) -> bool:
        """Check if at least one stone from original group still exists."""
        for r, c in original_group:
            if board.board[r, c] == color:
                return True
        return False

    def _find_group_after_move(
        self,
        board: Board,
        original_group: List[Tuple[int, int]],
        color: int
    ) -> Optional[List[Tuple[int, int]]]:
        """Find the group containing stones from original group after a move."""
        for r, c in original_group:
            if board.board[r, c] == color:
                return board.get_group(r, c)
        return None

    def _get_vital_points(
        self,
        board: Board,
        group: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Get vital points for life/death of a group.

        Includes liberties and eye-making points.
        """
        vital = set()

        # All liberties
        for r, c in group:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board.size and 0 <= nc < board.size:
                    if board.board[nr, nc] == 0:
                        vital.add((nr, nc))

        # Eye-making points (diagonals that could form eyes)
        for r, c in group:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board.size and 0 <= nc < board.size:
                    if board.board[nr, nc] == 0:
                        vital.add((nr, nc))

        return list(vital)

    def _static_life_eval(
        self,
        board: Board,
        group: List[Tuple[int, int]]
    ) -> float:
        """Static evaluation of group life status."""
        color = board.board[group[0][0], group[0][1]]

        if not self._group_exists(board, group, color):
            return -1.0

        # Find current group
        current_group = self._find_group_after_move(board, group, color)
        if not current_group:
            return -1.0

        eyes = board.count_eyes(current_group)
        libs = board.count_liberties(current_group)
        size = len(current_group)

        if eyes >= 2:
            return 1.0
        if libs == 0:
            return -1.0

        # Heuristic scoring
        score = 0.0
        score += eyes * 0.4  # One eye is good
        score += min(libs, 4) * 0.1  # Liberties help
        score += min(size, 10) * 0.02  # Bigger groups are harder to kill

        return max(-1.0, min(1.0, score))

    # =========================================================================
    # POSITION CLASSIFICATION
    # =========================================================================

    def is_tactical_position(self, board: Board) -> bool:
        """Check if position has tactical activity.

        Used to decide whether to apply tactical search vs trust NN.

        Returns True if:
        - Any group is in atari or near-atari
        - Recent capture occurred
        - Ko in progress
        - Cutting points exist between groups
        - Connection points exist between friendly groups
        """
        player = board.current_player

        # Check for atari groups
        visited = set()
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != 0 and (r, c) not in visited:
                    group = board.get_group(r, c)
                    visited.update(group)

                    libs = board.count_liberties(group)
                    if libs <= 2:  # Atari or near-atari
                        return True

        # Ko in progress
        if board.ko_point is not None:
            return True

        # Check for cutting/connecting points
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] == 0:
                    # Count adjacent groups by color
                    adj_player_groups = set()
                    adj_opp_groups = set()

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < board.size and 0 <= nc < board.size:
                            stone = board.board[nr, nc]
                            if stone == player:
                                # Find group id (first stone as proxy)
                                group = board.get_group(nr, nc)
                                adj_player_groups.add(group[0])
                            elif stone == -player:
                                group = board.get_group(nr, nc)
                                adj_opp_groups.add(group[0])

                    # If 2+ friendly groups adjacent = connection point
                    if len(adj_player_groups) >= 2:
                        return True
                    # If 2+ opponent groups adjacent = cutting point
                    if len(adj_opp_groups) >= 2:
                        return True

        return False

    def get_tactical_boost(
        self,
        board: Board,
        move: Tuple[int, int]
    ) -> float:
        """Get tactical priority boost for a move.

        Returns multiplier for neural network prior:
        - >1.0: boost this move
        - 1.0: neutral
        - <1.0: penalize this move
        """
        if board.board[move[0], move[1]] != 0:
            return 0.01  # Illegal

        player = board.current_player
        boost = 1.0

        # Check for immediate capture
        test = board.copy()
        try:
            captures = test.play(move[0], move[1])
        except:
            return 0.01  # Illegal

        if captures > 0:
            boost *= (1 + captures * 0.5)

        # Check for snapback (stronger boost) - track for later
        snapback = self.detect_snapback(board, move)
        is_snapback = snapback > 0
        if is_snapback:
            boost *= (2.0 + snapback * 0.5)  # Much stronger snapback boost

        # Check if move connects two friendly groups
        connect_boost = self._check_connect_boost(board, move, player)
        if connect_boost > 1.0:
            boost *= connect_boost

        # Check if move cuts opponent groups
        cut_boost = self._check_cut_boost(board, move, player)
        if cut_boost > 1.0:
            boost *= cut_boost

        # Check if move puts opponent in atari (ladder continuation)
        test = board.copy()
        test.board[move[0], move[1]] = player
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = move[0] + dr, move[1] + dc
            if 0 <= nr < board.size and 0 <= nc < board.size:
                if test.board[nr, nc] == -player:
                    group = test.get_group(nr, nc)
                    libs = test.count_liberties(group)
                    if libs == 1:
                        # Atari! Check if it's a working ladder
                        ladder_result = self.trace_ladder(test, group, player)
                        if ladder_result is True:
                            boost *= (2.0 + len(group) * 0.3)  # Strong ladder chase
                        else:
                            boost *= (1 + len(group) * 0.2)  # Atari threat

        # Check if move saves own group
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = move[0] + dr, move[1] + dc
            if 0 <= nr < board.size and 0 <= nc < board.size:
                if board.board[nr, nc] == player:
                    group = board.get_group(nr, nc)
                    libs = board.count_liberties(group)
                    if libs == 1:
                        # Saving group in atari - but verify it actually helps
                        test = board.copy()
                        test.board[move[0], move[1]] = player
                        new_group = test.get_group(move[0], move[1])
                        new_libs = test.count_liberties(new_group)
                        if new_libs >= 2:
                            boost *= (1 + len(group) * 0.4)

        # Check if move creates dead group (self-atari into ladder)
        # BUT skip this check for snapback moves (snapback IS intentional self-atari)
        if not is_snapback:
            test = board.copy()
            test.board[move[0], move[1]] = player
            new_group = test.get_group(move[0], move[1])
            new_libs = test.count_liberties(new_group)

            if new_libs == 1:
                # In atari - check if it's a ladder
                ladder_result = self.trace_ladder(test, new_group, -player)
                if ladder_result is True:
                    boost *= 0.01  # Massive penalty - creates dead group

        # Apply 8 Basic Instincts from Sensei's Library
        instinct_boost = self._get_instinct_boost(board, move)
        if instinct_boost > 1.0:
            boost *= instinct_boost

        return boost

    def _get_instinct_boost(self, board: Board, move: Tuple[int, int]) -> float:
        """Get boost from the 8 basic instincts.

        Lazy loads InstinctAnalyzer to avoid circular imports.
        """
        if not hasattr(self, '_instinct_analyzer'):
            from instincts import InstinctAnalyzer
            self._instinct_analyzer = InstinctAnalyzer()

        return self._instinct_analyzer.get_instinct_boost(board, move)

    def _check_connect_boost(
        self,
        board: Board,
        move: Tuple[int, int],
        player: int
    ) -> float:
        """Check if move connects two friendly groups.

        Returns boost multiplier (>1.0 if connecting).
        """
        r, c = move

        # Find adjacent friendly groups (before move)
        adjacent_groups = []
        seen = set()

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board.size and 0 <= nc < board.size:
                if board.board[nr, nc] == player and (nr, nc) not in seen:
                    group = board.get_group(nr, nc)
                    seen.update(group)
                    adjacent_groups.append(group)

        # Connecting 2+ groups is valuable
        if len(adjacent_groups) >= 2:
            total_stones = sum(len(g) for g in adjacent_groups)
            # Stronger boost for connecting larger groups
            return 1.5 + total_stones * 0.1

        return 1.0

    def _check_cut_boost(
        self,
        board: Board,
        move: Tuple[int, int],
        player: int
    ) -> float:
        """Check if move cuts opponent groups.

        Returns boost multiplier (>1.0 if cutting).
        """
        r, c = move
        opponent = -player

        # Find adjacent opponent groups (before move)
        adjacent_opp_groups = []
        seen = set()

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board.size and 0 <= nc < board.size:
                if board.board[nr, nc] == opponent and (nr, nc) not in seen:
                    group = board.get_group(nr, nc)
                    seen.update(group)
                    adjacent_opp_groups.append(group)

        # Cutting between 2+ opponent groups is valuable
        if len(adjacent_opp_groups) >= 2:
            total_stones = sum(len(g) for g in adjacent_opp_groups)
            # Stronger boost for cutting larger groups
            return 1.5 + total_stones * 0.1

        return 1.0


# =============================================================================
# TACTICAL FEATURE GENERATION
# =============================================================================

def compute_tactical_planes(
    board: Board,
    analyzer: TacticalAnalyzer
) -> np.ndarray:
    """Compute additional tactical feature planes for neural network.

    Returns 6 planes:
    - 0: Ladder-threatened groups (own groups in working ladders)
    - 1: Ladder-breaker moves (playing here breaks a ladder)
    - 2: Snapback opportunities
    - 3: Capture-in-1 moves
    - 4: Capture-in-2 moves
    - 5: Dead groups (own groups that are tactically dead)

    Args:
        board: Current board state
        analyzer: TacticalAnalyzer instance

    Returns:
        (6, size, size) numpy array
    """
    size = board.size
    planes = np.zeros((6, size, size), dtype=np.float32)
    player = board.current_player

    # Process groups
    visited = set()
    for r in range(size):
        for c in range(size):
            if board.board[r, c] == player and (r, c) not in visited:
                group = board.get_group(r, c)
                visited.update(group)

                libs = board.count_liberties(group)
                if libs <= 2:
                    # Check for ladder
                    ladder_result = analyzer.trace_ladder(board, group, -player)
                    if ladder_result is True:
                        # Mark group as ladder-threatened
                        for gr, gc in group:
                            planes[0, gr, gc] = 1.0

                        # Find ladder-breaker moves
                        for lr, lc in analyzer._get_liberties(board, group):
                            # Would playing here break the ladder?
                            test = board.copy()
                            test.board[lr, lc] = player
                            new_group = test.get_group(lr, lc)
                            new_result = analyzer.trace_ladder(test, new_group, -player)
                            if new_result is False:
                                planes[1, lr, lc] = 1.0

    # Process opponent groups for captures
    visited.clear()
    for r in range(size):
        for c in range(size):
            if board.board[r, c] == -player and (r, c) not in visited:
                group = board.get_group(r, c)
                visited.update(group)

                libs = board.count_liberties(group)
                if libs == 1:
                    # Capture in 1
                    lib = analyzer._get_single_liberty(board, group)
                    if lib:
                        planes[3, lib[0], lib[1]] = 1.0

    # Snapback opportunities
    for r in range(size):
        for c in range(size):
            if board.board[r, c] == 0:
                snapback = analyzer.detect_snapback(board, (r, c))
                if snapback > 0:
                    planes[2, r, c] = snapback / 5.0  # Normalize

    # Capture-in-2 (atari moves that lead to capture)
    for r in range(size):
        for c in range(size):
            if board.board[r, c] == 0:
                test = board.copy()
                try:
                    test.board[r, c] = player
                except:
                    continue

                # Check if this creates atari on opponent
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        if test.board[nr, nc] == -player:
                            opp_group = test.get_group(nr, nc)
                            opp_libs = test.count_liberties(opp_group)
                            if opp_libs == 1:
                                planes[4, r, c] = 1.0
                                break

    # Dead groups (own groups that fail life/death search)
    visited.clear()
    for r in range(size):
        for c in range(size):
            if board.board[r, c] == player and (r, c) not in visited:
                group = board.get_group(r, c)
                visited.update(group)

                libs = board.count_liberties(group)
                if libs <= 3 and len(group) >= 3:
                    # Small group in danger - evaluate
                    status = analyzer.evaluate_life_death(board, group, depth=4)
                    if status < -0.5:
                        for gr, gc in group:
                            planes[5, gr, gc] = 1.0

    return planes
