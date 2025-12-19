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
    """3. Hane at Head of Two (二子の頭にハネ)"""

    def test_two_horizontal_stones_hane_at_head(self, detector):
        """Two horizontal white stones - black plays at the head."""
        board = Board(9)

        board.board[4, 4] = -1
        board.board[4, 5] = -1
        board.current_player = 1

        result = detector.detect_hane_at_head_of_two(board)

        assert result is not None
        assert (4, 6) in result.moves or (4, 3) in result.moves

    def test_two_vertical_stones_hane_at_head(self, detector):
        """Two vertical white stones - black plays at the head."""
        board = Board(9)

        board.board[4, 4] = -1
        board.board[5, 4] = -1
        board.current_player = 1

        result = detector.detect_hane_at_head_of_two(board)

        assert result is not None
        assert (6, 4) in result.moves or (3, 4) in result.moves

    def test_three_stones_not_two(self, detector):
        """Three stones - not 'head of two'."""
        board = Board(9)

        board.board[4, 4] = -1
        board.board[4, 5] = -1
        board.board[4, 6] = -1
        board.current_player = 1

        result = detector.detect_hane_at_head_of_two(board)

        # Should not detect (chain is longer than 2)
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
    """5. Block the Angle (カケにはオサエ)"""

    def test_angle_attack_should_block(self, detector):
        """White angle attack - black blocks."""
        board = Board(9)
        c = 4

        board.board[c, c] = 1
        board.board[c+1, c+1] = -1
        board.current_player = 1

        result = detector.detect_block_the_angle(board)

        assert result is not None
        # Block at (c+1, c) or (c, c+1)
        assert (c+1, c) in result.moves or (c, c+1) in result.moves


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
    """7. Block the Thrust (ツキアタリには)"""

    def test_thrust_should_block(self, detector):
        """White thrusts between black stones - black blocks."""
        board = Board(9)

        board.board[4, 3] = 1
        board.board[4, 5] = 1
        board.board[5, 4] = -1  # Thrust
        board.current_player = 1

        result = detector.detect_block_the_thrust(board)

        assert result is not None
        assert (4, 4) in result.moves


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
