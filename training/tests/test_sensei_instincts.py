#!/usr/bin/env python3
"""Tests for Sensei's 8 Basic Instincts detectors.

TDD approach: Tests define expected behavior.

The 8 Basic Instincts (https://senseis.xmp.net/?BasicInstinct):
1. Extend from Atari (アタリから伸びよ)
2. Hane vs Tsuke (ツケにはハネ)
3. Hane at Head of Two (二子の頭にハネ)
4. Stretch from Kosumi (コスミから伸びよ)
5. Block the Angle (カケにはオサエ)
6. Connect vs Peep (ノゾキにはツギ)
7. Block the Thrust (ツキアタリには)
8. Stretch from Bump (ブツカリから伸びよ)
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from board import Board
from sensei_instincts import SenseiInstinctDetector


@pytest.fixture
def detector():
    return SenseiInstinctDetector()


class TestExtendFromAtari:
    """1. Extend from Atari (アタリから伸びよ)"""

    def test_single_stone_in_atari_should_extend(self, detector):
        """Black stone with one liberty should extend."""
        board = Board(9)
        c = 4

        board.board[c, c] = 1      # Black stone
        board.board[c-1, c] = -1   # White above
        board.board[c+1, c] = -1   # White below
        board.board[c, c-1] = -1   # White left
        board.current_player = 1

        result = detector.detect_extend_from_atari(board)

        assert result is not None
        assert (c, c+1) in result.moves

    def test_group_in_atari_should_extend(self, detector):
        """Black group with one liberty should extend."""
        board = Board(9)

        board.board[4, 4] = 1
        board.board[4, 5] = 1
        board.board[3, 4] = -1
        board.board[3, 5] = -1
        board.board[5, 4] = -1
        board.board[5, 5] = -1
        board.board[4, 3] = -1
        board.current_player = 1

        result = detector.detect_extend_from_atari(board)

        assert result is not None
        assert (4, 6) in result.moves

    def test_no_atari_no_extend(self, detector):
        """If no stones in atari, should not detect extend."""
        board = Board(9)
        board.board[4, 4] = 1
        board.current_player = 1

        result = detector.detect_extend_from_atari(board)

        assert result is None


class TestHaneVsTsuke:
    """2. Hane vs Tsuke (ツケにはハネ)"""

    def test_tsuke_should_trigger_hane(self, detector):
        """White tsuke on black stone should suggest hane."""
        board = Board(9)
        c = 4

        board.board[c, c] = 1
        board.board[c+1, c] = -1  # White tsuke (unsupported)
        board.current_player = 1

        result = detector.detect_hane_vs_tsuke(board)

        assert result is not None
        # Hane at (c+1, c+1) or (c+1, c-1)
        assert any(m in result.moves for m in [(c+1, c+1), (c+1, c-1)])

    def test_supported_attachment_not_tsuke(self, detector):
        """Supported attachment is bump, not tsuke."""
        board = Board(9)
        c = 4

        board.board[c, c] = 1
        board.board[c+1, c] = -1   # Attachment
        board.board[c+2, c] = -1   # Support
        board.current_player = 1

        result = detector.detect_hane_vs_tsuke(board)

        # Should NOT detect tsuke (this is a bump)
        assert result is None


class TestHaneAtHeadOfTwo:
    """3. Hane at Head of Two (二子の頭にハネ)

    CORRECT PATTERN: This is a 2v2 confrontation!
    Our two stones parallel to their two stones.

    Pattern (vertical):
        . . * . .    ← Play at head of opponent's two
        . B W . .    ← Our stone faces their stone
        . B W . .    ← Our stone faces their stone
        . . . . .
    """

    def test_2v2_vertical_confrontation(self, detector):
        """2v2 vertical: our two face their two, play at head."""
        board = Board(9)

        # Our pair (Black) at column 3
        board.board[4, 3] = 1
        board.board[5, 3] = 1
        # Their pair (White) at column 4
        board.board[4, 4] = -1
        board.board[5, 4] = -1
        board.current_player = 1

        result = detector.detect_hane_at_head_of_two(board)

        assert result is not None
        # Head of their two: (3, 4) or (6, 4)
        assert (3, 4) in result.moves or (6, 4) in result.moves

    def test_2v2_horizontal_confrontation(self, detector):
        """2v2 horizontal: our two face their two, play at head."""
        board = Board(9)

        # Our pair (Black) at row 3
        board.board[3, 4] = 1
        board.board[3, 5] = 1
        # Their pair (White) at row 4
        board.board[4, 4] = -1
        board.board[4, 5] = -1
        board.current_player = 1

        result = detector.detect_hane_at_head_of_two(board)

        assert result is not None
        # Head of their two: (4, 3) or (4, 6)
        assert (4, 3) in result.moves or (4, 6) in result.moves

    def test_isolated_two_stones_not_detected(self, detector):
        """Only opponent stones (no 2v2) should NOT trigger."""
        board = Board(9)

        # Only opponent stones - no confrontation
        board.board[4, 4] = -1
        board.board[4, 5] = -1
        board.current_player = 1

        result = detector.detect_hane_at_head_of_two(board)

        # Should NOT detect - this is NOT a 2v2 confrontation
        assert result is None


class TestStretchFromKosumi:
    """4. Stretch from Kosumi (コスミから伸びよ)"""

    def test_kosumi_tsuke_should_stretch(self, detector):
        """White diagonal attachment - black stretches away."""
        board = Board(9)
        c = 4

        board.board[c, c] = 1
        board.board[c+1, c+1] = -1  # Diagonal
        board.current_player = 1

        result = detector.detect_stretch_from_kosumi(board)

        assert result is not None
        # Stretch away: (c-1, c-1) or orthogonal away
        assert len(result.moves) > 0


class TestBlockTheAngle:
    """5. Block the Angle (カケにはオサエ)

    CORRECT PATTERN: Knight's move (keima) approach, NOT diagonal!

    Pattern:
        . . . . .
        . . . B .    ← Your stone
        . . * . .    ← Block diagonally between
        . W . . .    ← Opponent's knight's move (2 down, 1 left)
        . . . . .
    """

    def test_keima_approach_should_block(self, detector):
        """White knight's move approach - black blocks diagonally."""
        board = Board(9)
        c = 4

        board.board[c, c] = 1           # Our stone
        board.board[c+2, c-1] = -1      # Opponent's keima (2 down, 1 left)
        board.current_player = 1

        result = detector.detect_block_the_angle(board)

        assert result is not None
        # Block is diagonal between: (c+1, c-1) - the "waist" of the keima
        assert (c+1, c-1) in result.moves

    def test_keima_other_direction(self, detector):
        """Knight's move in another direction."""
        board = Board(9)

        board.board[4, 4] = 1           # Our stone
        board.board[5, 6] = -1          # Opponent's keima (1 down, 2 right)
        board.current_player = 1

        result = detector.detect_block_the_angle(board)

        assert result is not None
        # Block diagonal: (5, 5)
        assert (5, 5) in result.moves

    def test_diagonal_not_keima(self, detector):
        """Diagonal (1,1) is NOT keima - should not trigger."""
        board = Board(9)

        board.board[4, 4] = 1
        board.board[5, 5] = -1   # Diagonal, not keima
        board.current_player = 1

        result = detector.detect_block_the_angle(board)

        # Diagonal is kosumi-tsuke, not keima - should NOT trigger this instinct
        assert result is None


class TestConnectVsPeep:
    """6. Connect vs Peep (ノゾキにはツギ)"""

    def test_peep_should_connect(self, detector):
        """White peeps between two black stones - black must connect."""
        board = Board(9)

        board.board[4, 3] = 1
        board.board[4, 5] = 1
        board.board[3, 4] = -1  # Peep
        board.current_player = 1

        result = detector.detect_connect_vs_peep(board)

        assert result is not None
        assert (4, 4) in result.moves


class TestBlockTheThrust:
    """7. Block the Thrust (ツキアタリには)

    CORRECT PATTERN: Opponent thrusts INTO our wall formation.

    Pattern (vertical wall):
        . . . . .
        . . * . .    ← Black BLOCKS by extending wall
        . B W . .    ← White THRUSTS adjacent to our stone
        . B . . .    ← Our stones in column (wall)
        . . . . .

    The thrust is perpendicular to our wall. Block extends the wall.
    """

    def test_thrust_into_vertical_wall(self, detector):
        """White thrusts into vertical black wall - block extends wall."""
        board = Board(9)

        # Vertical wall of black stones
        board.board[5, 3] = 1   # Bottom of wall
        board.board[4, 3] = 1   # Top of wall
        # White thrusts adjacent to top stone
        board.board[4, 4] = -1  # Thrust perpendicular to wall
        board.current_player = 1

        result = detector.detect_block_the_thrust(board)

        assert result is not None
        # Block extends wall past the thrust: (4, 5) - continuing perpendicular
        assert (4, 5) in result.moves

    def test_thrust_into_horizontal_wall(self, detector):
        """White thrusts into horizontal black wall."""
        board = Board(9)

        # Horizontal wall of black stones
        board.board[4, 4] = 1
        board.board[4, 5] = 1
        # White thrusts from below
        board.board[5, 4] = -1
        board.current_player = 1

        result = detector.detect_block_the_thrust(board)

        assert result is not None
        # Block extends in thrust direction: (6, 4)
        assert (6, 4) in result.moves

    def test_no_wall_no_thrust(self, detector):
        """Single stone with adjacent opponent is not a thrust."""
        board = Board(9)

        board.board[4, 4] = 1    # Single stone (not a wall)
        board.board[4, 5] = -1   # Adjacent opponent
        board.current_player = 1

        result = detector.detect_block_the_thrust(board)

        # Single stone is not a wall - should not trigger thrust
        assert result is None


class TestStretchFromBump:
    """8. Stretch from Bump (ブツカリから伸びよ)"""

    def test_bump_should_stretch(self, detector):
        """White bump (supported) - black stretches."""
        board = Board(9)
        c = 4

        board.board[c, c] = 1
        board.board[c+1, c] = -1   # Attachment
        board.board[c+2, c] = -1   # Support (makes it bump)
        board.current_player = 1

        result = detector.detect_stretch_from_bump(board)

        assert result is not None
        assert (c-1, c) in result.moves


class TestDetectAll:
    """Test detect_all returns prioritized results."""

    def test_atari_highest_priority(self, detector):
        """Extend from atari should be highest priority."""
        board = Board(9)

        # Stone in atari
        board.board[4, 4] = 1
        board.board[3, 4] = -1
        board.board[5, 4] = -1
        board.board[4, 3] = -1
        board.current_player = 1

        results = detector.detect_all(board)

        assert len(results) > 0
        assert results[0].instinct == 'extend_from_atari'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
