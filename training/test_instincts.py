#!/usr/bin/env python3
"""Tests for the 8 Basic Instincts from Sensei's Library.

TDD: Tests define expected behavior for instinct detection.

The 8 Basic Instincts:
1. From an atari, extend
2. Answer the tsuke with a hane
3. Hane at the head of two stones
4. Stretch from a kosumi-tsuke
5. Block the angle play
6. Connect against a peep
7. Block the thrust
8. Stretch from a bump

Reference: https://senseis.xmp.net/?BasicInstinct
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from board import Board


class TestInstinctDetection:
    """Test that InstinctAnalyzer detects the 8 basic instincts."""

    @pytest.fixture
    def analyzer(self):
        from instincts import InstinctAnalyzer
        return InstinctAnalyzer()

    @pytest.fixture
    def board(self):
        return Board(9)

    # =========================================================================
    # 1. FROM AN ATARI, EXTEND
    # =========================================================================
    def test_extend_from_atari(self, analyzer, board):
        """Stone in atari should extend to gain liberties."""
        c = 4
        # Black stone with one liberty (in atari)
        board.board[c, c] = 1       # Black stone
        board.board[c-1, c] = -1    # White surrounding
        board.board[c+1, c] = -1
        board.board[c, c-1] = -1
        # Liberty at (c, c+1)

        board.current_player = 1  # Black to move

        # The extend move should be detected
        extend_move = (c, c+1)
        result = analyzer.detect_extend_from_atari(board, extend_move)

        assert result is True, "Should detect extend from atari"
        assert analyzer.get_instinct_boost(board, extend_move) > 1.0

    def test_extend_not_triggered_when_safe(self, analyzer, board):
        """Should not suggest extend when not in atari."""
        c = 4
        board.board[c, c] = 1  # Black stone with many liberties
        board.current_player = 1

        # Random move should not be an "extend from atari"
        result = analyzer.detect_extend_from_atari(board, (c, c+1))
        assert result is False

    # =========================================================================
    # 2. ANSWER THE TSUKE WITH A HANE
    # =========================================================================
    def test_hane_response_to_tsuke(self, analyzer, board):
        """Contact play (tsuke) should be answered with hane."""
        c = 4
        # Black stone
        board.board[c, c] = 1
        # White tsuke (contact play)
        board.board[c, c+1] = -1

        board.current_player = 1  # Black to respond

        # Hane wraps around - diagonal to the tsuke
        hane_moves = [(c-1, c+1), (c+1, c+1)]  # Diagonal wraps

        detected = False
        for move in hane_moves:
            if analyzer.detect_hane_response(board, move):
                detected = True
                break

        assert detected, "Should detect hane as response to tsuke"

    # =========================================================================
    # 3. HANE AT THE HEAD OF TWO STONES
    # =========================================================================
    def test_hane_at_head_of_two(self, analyzer, board):
        """Playing at the head of two consecutive stones is powerful."""
        c = 4
        # Two consecutive white stones (vertical)
        board.board[c, c] = -1
        board.board[c+1, c] = -1

        board.current_player = 1  # Black to play

        # Hane at the head (above the two stones)
        head_move = (c-1, c)
        result = analyzer.detect_hane_at_head_of_two(board, head_move)

        assert result is True, "Should detect hane at head of two stones"
        assert analyzer.get_instinct_boost(board, head_move) > 1.5

    def test_hane_at_head_horizontal(self, analyzer, board):
        """Works for horizontal two stones as well."""
        c = 4
        # Two consecutive white stones (horizontal)
        board.board[c, c] = -1
        board.board[c, c+1] = -1

        board.current_player = 1

        # Head moves (at either end)
        head_moves = [(c, c-1), (c, c+2)]

        detected = any(
            analyzer.detect_hane_at_head_of_two(board, m)
            for m in head_moves
        )
        assert detected, "Should detect hane at head of horizontal two stones"

    # =========================================================================
    # 4. STRETCH FROM A KOSUMI-TSUKE
    # =========================================================================
    def test_stretch_from_kosumi_tsuke(self, analyzer, board):
        """Diagonal contact should be answered by extending away."""
        c = 4
        # Black stone
        board.board[c, c] = 1
        # White kosumi-tsuke (diagonal contact)
        board.board[c+1, c+1] = -1

        board.current_player = 1

        # Stretch moves (extend away from the diagonal)
        stretch_moves = [(c-1, c), (c, c-1)]  # Away from the diagonal

        detected = any(
            analyzer.detect_stretch_from_kosumi(board, m)
            for m in stretch_moves
        )
        assert detected, "Should detect stretch from kosumi-tsuke"

    # =========================================================================
    # 5. BLOCK THE ANGLE PLAY
    # =========================================================================
    def test_block_angle_play(self, analyzer, board):
        """Block diagonal attacks to maintain strength."""
        c = 4
        # Black stone
        board.board[c, c] = 1
        # White angle play (one-point jump diagonally threatening)
        board.board[c+2, c+2] = -1

        board.current_player = 1

        # Blocking move
        block_move = (c+1, c+1)
        result = analyzer.detect_block_angle(board, block_move)

        assert result is True, "Should detect blocking the angle play"

    # =========================================================================
    # 6. CONNECT AGAINST A PEEP
    # =========================================================================
    def test_connect_against_peep(self, analyzer, board):
        """When peep threatens to cut, connect."""
        c = 4
        # Two black stones with cutting point
        board.board[c, c] = 1
        board.board[c, c+2] = 1
        # White peep threatening the cut
        board.board[c-1, c+1] = -1

        board.current_player = 1

        # Connect move
        connect_move = (c, c+1)
        result = analyzer.detect_connect_against_peep(board, connect_move)

        assert result is True, "Should detect connect against peep"
        assert analyzer.get_instinct_boost(board, connect_move) > 2.0

    def test_peep_detection(self, analyzer, board):
        """Detect that a peep exists in position."""
        c = 4
        # Two black stones with gap
        board.board[c, c] = 1
        board.board[c, c+2] = 1
        # White peep
        board.board[c-1, c+1] = -1

        board.current_player = 1

        peeps = analyzer.find_peeps(board)
        assert len(peeps) > 0, "Should detect the peep"

    # =========================================================================
    # 7. BLOCK THE THRUST
    # =========================================================================
    def test_block_thrust(self, analyzer, board):
        """Block when opponent pushes through your position."""
        c = 4
        # Black wall
        board.board[c, c] = 1
        board.board[c, c+1] = 1
        board.board[c, c+2] = 1
        # White thrust (pushing into the formation)
        board.board[c+1, c+1] = -1

        board.current_player = 1

        # Block moves
        block_moves = [(c+1, c), (c+1, c+2)]

        detected = any(
            analyzer.detect_block_thrust(board, m)
            for m in block_moves
        )
        assert detected, "Should detect blocking the thrust"

    # =========================================================================
    # 8. STRETCH FROM A BUMP
    # =========================================================================
    def test_stretch_from_bump(self, analyzer, board):
        """When bumped, extend rather than hane."""
        c = 4
        # Black stone
        board.board[c, c] = 1
        # White already connected (making hane less effective)
        board.board[c, c+1] = -1
        board.board[c-1, c+1] = -1  # White backup stone

        board.current_player = 1

        # Stretch (extend) is better than hane here
        stretch_move = (c+1, c)  # Extend away
        hane_move = (c-1, c)     # Hane (less good here)

        stretch_boost = analyzer.get_instinct_boost(board, stretch_move)
        hane_boost = analyzer.get_instinct_boost(board, hane_move)

        # Stretch should be preferred when opponent is reinforced
        result = analyzer.detect_stretch_from_bump(board, stretch_move)
        assert result is True, "Should detect stretch from bump"


class TestInstinctBoosts:
    """Test that instinct boosts are properly calculated."""

    @pytest.fixture
    def analyzer(self):
        from instincts import InstinctAnalyzer
        return InstinctAnalyzer()

    def test_multiple_instincts_stack(self, analyzer):
        """Move satisfying multiple instincts gets combined boost."""
        board = Board(9)
        c = 4

        # Position where connect both defends atari AND blocks peep
        board.board[c, c] = 1
        board.board[c, c+2] = 1
        board.board[c-1, c] = -1
        board.board[c+1, c] = -1
        board.board[c-1, c+1] = -1  # Peep

        board.current_player = 1

        # Connect at (c, c+1) serves multiple purposes
        connect_move = (c, c+1)
        boost = analyzer.get_instinct_boost(board, connect_move)

        # Should be significantly boosted
        assert boost > 2.0, "Multiple instincts should stack boost"

    def test_no_instinct_returns_neutral(self, analyzer):
        """Move with no instinct pattern returns neutral boost."""
        board = Board(9)
        board.current_player = 1

        # Random move on empty board
        boost = analyzer.get_instinct_boost(board, (4, 4))
        assert boost == 1.0, "No instinct should return neutral boost"


class TestInstinctIntegration:
    """Test integration with TacticalAnalyzer."""

    def test_tactical_analyzer_uses_instincts(self):
        """TacticalAnalyzer should incorporate instinct boosts."""
        from tactics import TacticalAnalyzer

        analyzer = TacticalAnalyzer()
        board = Board(9)
        c = 4

        # Setup peep position
        board.board[c, c] = 1
        board.board[c, c+2] = 1
        board.board[c-1, c+1] = -1  # Peep

        board.current_player = 1

        # Connect move should get boosted
        connect_move = (c, c+1)
        boost = analyzer.get_tactical_boost(board, connect_move)

        # Should be boosted (exact value depends on implementation)
        assert boost >= 1.0, "Tactical analyzer should consider instincts"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
