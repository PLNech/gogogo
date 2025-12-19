#!/usr/bin/env python3
"""Tests for Sensei's 8 Basic Instincts detectors.

TDD approach: Write tests FIRST, then implement detectors.

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
import numpy as np
import sys
from pathlib import Path

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from board import Board


# We'll import this once implemented
# from sensei_instincts import SenseiInstinctDetector


class TestExtendFromAtari:
    """1. Extend from Atari (アタリから伸びよ)

    When your stone faces capture (one liberty), extend to gain liberties.
    """

    def test_single_stone_in_atari_should_extend(self):
        """Black stone with one liberty should extend."""
        board = Board(9)
        c = 4  # center

        # Black stone surrounded on 3 sides
        board.board[c, c] = 1      # Black stone
        board.board[c-1, c] = -1   # White above
        board.board[c+1, c] = -1   # White below
        board.board[c, c-1] = -1   # White left
        # Liberty at (c, c+1) - right side

        board.current_player = 1  # Black to play

        # detector = SenseiInstinctDetector()
        # result = detector.detect_extend_from_atari(board)
        #
        # assert result is not None
        # assert (c, c+1) in result.moves  # Extend to the liberty
        pytest.skip("Detector not yet implemented")

    def test_group_in_atari_should_extend(self):
        """Black group with one liberty should extend."""
        board = Board(9)

        # Two black stones in a row, surrounded
        board.board[4, 4] = 1
        board.board[4, 5] = 1
        board.board[3, 4] = -1
        board.board[3, 5] = -1
        board.board[5, 4] = -1
        board.board[5, 5] = -1
        board.board[4, 3] = -1
        # Liberty at (4, 6)

        board.current_player = 1

        pytest.skip("Detector not yet implemented")

    def test_no_atari_no_extend(self):
        """If no stones in atari, should not detect extend."""
        board = Board(9)
        board.board[4, 4] = 1  # Black stone with all liberties
        board.current_player = 1

        pytest.skip("Detector not yet implemented")


class TestHaneVsTsuke:
    """2. Hane vs Tsuke (ツケにはハネ)

    When opponent plays an unsupported adjacent stone (tsuke),
    respond with a hane to block their development.
    """

    def test_tsuke_should_trigger_hane(self):
        """White tsuke on black stone should suggest hane."""
        board = Board(9)
        c = 4

        # Black stone
        board.board[c, c] = 1
        # White tsuke (unsupported attachment)
        board.board[c+1, c] = -1

        board.current_player = 1  # Black to respond

        # Hane moves would be diagonal: (c+1, c+1) or (c+1, c-1)
        pytest.skip("Detector not yet implemented")

    def test_supported_attachment_not_tsuke(self):
        """Supported attachment is NOT a tsuke - different response."""
        board = Board(9)
        c = 4

        board.board[c, c] = 1      # Black
        board.board[c+1, c] = -1   # White attachment
        board.board[c+2, c] = -1   # Support stone - this is a bump, not tsuke

        board.current_player = 1

        pytest.skip("Detector not yet implemented")


class TestHaneAtHeadOfTwo:
    """3. Hane at Head of Two (二子の頭にハネ)

    Play above two consecutive opponent stones to create weakness.
    """

    def test_two_horizontal_stones_hane_at_head(self):
        """Two horizontal white stones - black plays at the head."""
        board = Board(9)

        # Two white stones in a row
        board.board[4, 4] = -1
        board.board[4, 5] = -1

        board.current_player = 1  # Black to play

        # Head is at (4, 6) or (4, 3) depending on direction
        pytest.skip("Detector not yet implemented")

    def test_two_vertical_stones_hane_at_head(self):
        """Two vertical white stones - black plays at the head."""
        board = Board(9)

        board.board[4, 4] = -1
        board.board[5, 4] = -1

        board.current_player = 1

        # Head is at (6, 4) or (3, 4)
        pytest.skip("Detector not yet implemented")

    def test_three_stones_not_two(self):
        """Three stones - different situation, not 'head of two'."""
        board = Board(9)

        board.board[4, 4] = -1
        board.board[4, 5] = -1
        board.board[4, 6] = -1

        board.current_player = 1

        pytest.skip("Detector not yet implemented")


class TestStretchFromKosumi:
    """4. Stretch from Kosumi (コスミから伸びよ)

    When opponent plays a diagonal attachment (kosumi-tsuke),
    stretch away rather than hane.
    """

    def test_kosumi_tsuke_should_stretch(self):
        """White diagonal attachment - black stretches away."""
        board = Board(9)
        c = 4

        board.board[c, c] = 1      # Black stone
        board.board[c+1, c+1] = -1  # White kosumi-tsuke (diagonal)

        board.current_player = 1

        # Stretch move: extend away from the kosumi
        # e.g., (c-1, c) or (c, c-1)
        pytest.skip("Detector not yet implemented")


class TestBlockTheAngle:
    """5. Block the Angle (カケにはオサエ)

    Respond to diagonal threats by blocking diagonally.
    """

    def test_angle_attack_should_block(self):
        """White angle attack - black blocks."""
        board = Board(9)
        c = 4

        board.board[c, c] = 1      # Black stone
        board.board[c+1, c+1] = -1  # White angle play (kake)

        board.current_player = 1

        # Block move at (c+1, c) or (c, c+1)
        pytest.skip("Detector not yet implemented")


class TestConnectVsPeep:
    """6. Connect vs Peep (ノゾキにはツギ)

    'Even a moron connects against a peep.'
    When opponent peeps between your stones, connect immediately.
    """

    def test_peep_should_connect(self):
        """White peeps between two black stones - black must connect."""
        board = Board(9)

        # Two black stones with gap
        board.board[4, 3] = 1
        board.board[4, 5] = 1
        # White peeps at the cutting point
        board.board[3, 4] = -1  # Peep threatens (4, 4)

        board.current_player = 1

        # Must connect at (4, 4)
        pytest.skip("Detector not yet implemented")

    def test_diagonal_connection_peep(self):
        """Peep threatening diagonal connection."""
        board = Board(9)

        # Diagonal black stones
        board.board[4, 4] = 1
        board.board[5, 5] = 1
        # White peeps
        board.board[4, 5] = -1  # or (5, 4)

        board.current_player = 1

        pytest.skip("Detector not yet implemented")


class TestBlockTheThrust:
    """7. Block the Thrust (ツキアタリには)

    When opponent thrusts between your stones, block it.
    """

    def test_thrust_should_block(self):
        """White thrusts between black stones - black blocks."""
        board = Board(9)

        # Two black stones
        board.board[4, 3] = 1
        board.board[4, 5] = 1
        # White thrusts from below
        board.board[5, 4] = -1

        board.current_player = 1

        # Block at (4, 4)
        pytest.skip("Detector not yet implemented")


class TestStretchFromBump:
    """8. Stretch from Bump (ブツカリから伸びよ)

    When opponent bumps (supported attachment), stretch rather than hane.
    """

    def test_bump_should_stretch(self):
        """White bump (supported) - black stretches, not hane."""
        board = Board(9)
        c = 4

        board.board[c, c] = 1      # Black stone
        board.board[c+1, c] = -1   # White attachment
        board.board[c+2, c] = -1   # Support stone (makes it a bump)

        board.current_player = 1

        # Stretch: extend away, e.g., (c-1, c)
        # NOT hane at (c+1, c+1)
        pytest.skip("Detector not yet implemented")

    def test_bump_vs_tsuke_distinction(self):
        """Bump has support, tsuke doesn't - different responses."""
        board = Board(9)
        c = 4

        # Bump scenario (supported)
        board.board[c, c] = 1
        board.board[c+1, c] = -1
        board.board[c+2, c] = -1  # Support

        board.current_player = 1

        # Should suggest stretch, NOT hane
        pytest.skip("Detector not yet implemented")


class TestInstinctPriorities:
    """Test that instincts are prioritized correctly.

    Some situations may trigger multiple instincts.
    Priority order matters for training signal.
    """

    def test_extend_from_atari_highest_priority(self):
        """Survival (extend from atari) should be highest priority."""
        board = Board(9)

        # Stone in atari AND peep situation
        board.board[4, 4] = 1
        board.board[3, 4] = -1
        board.board[5, 4] = -1
        board.board[4, 3] = -1
        # Also create a peep situation elsewhere
        board.board[6, 6] = 1
        board.board[6, 8] = 1
        board.board[5, 7] = -1

        board.current_player = 1

        # Extend from atari should be detected first
        pytest.skip("Detector not yet implemented")


class TestEdgeCases:
    """Edge cases and corner positions."""

    def test_corner_extend_from_atari(self):
        """Atari in corner - fewer escape routes."""
        board = Board(9)

        # Black stone in corner, in atari
        board.board[0, 0] = 1
        board.board[0, 1] = -1
        # Liberty at (1, 0)

        board.current_player = 1

        pytest.skip("Detector not yet implemented")

    def test_edge_hane_at_head_of_two(self):
        """Head of two near edge."""
        board = Board(9)

        board.board[0, 4] = -1
        board.board[0, 5] = -1

        board.current_player = 1

        # Only one valid head position (0, 6) since (0, 3) or beyond edge
        pytest.skip("Detector not yet implemented")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
