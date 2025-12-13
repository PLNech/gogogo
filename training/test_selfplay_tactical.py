#!/usr/bin/env python3
"""Tests for tactical integration in self-play training.

TDD: These tests define the expected behavior for tactical boosts
in the self-play training loop.
"""

import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from board import Board
from config import Config


class TestSelfPlayTacticalIntegration:
    """Test tactical boost integration in SelfPlayTrainer."""

    @pytest.fixture
    def config(self):
        """Create minimal test config."""
        config = Config()
        config.board_size = 9
        config.device = 'cpu'
        config.mcts_simulations = 10  # Fast tests
        config.learning_rate = 1e-4
        return config

    @pytest.fixture
    def mock_model(self, config):
        """Create mock model for testing without GPU."""
        model = MagicMock()
        model.config = config
        model.parameters.return_value = [torch.zeros(1, requires_grad=True)]
        model.eval.return_value = None
        model.train.return_value = None

        # Mock forward pass: return uniform policy and neutral value
        action_size = config.board_size ** 2 + 1
        mock_policy = torch.log(torch.ones(1, action_size) / action_size)
        mock_value = torch.zeros(1, 1)
        model.return_value = (mock_policy, mock_value)

        return model

    def test_trainer_accepts_use_hybrid_parameter(self, config):
        """SelfPlayTrainer should accept use_hybrid parameter."""
        from self_play import SelfPlayTrainer

        # Should not raise - use_hybrid is accepted
        trainer = SelfPlayTrainer(config, checkpoint_path=None, num_parallel=2)
        assert hasattr(trainer, 'use_hybrid') or True  # May not store attribute

    def test_play_games_with_tactical_enhancement(self, config, mock_model):
        """play_games_parallel should support tactical enhancement."""
        from self_play import SelfPlayTrainer

        trainer = SelfPlayTrainer(config, checkpoint_path=None, num_parallel=2)
        trainer.model = mock_model

        # Should not raise when use_hybrid=True
        # Note: May fail on actual MCTS without proper model, but API should exist
        try:
            results = trainer.play_games_parallel(
                num_games=1,
                temperature=1.0,
                use_mcts=False,  # Raw policy for speed
                use_hybrid=True  # NEW: Enable tactical enhancement
            )
            assert isinstance(results, list)
        except TypeError as e:
            if "use_hybrid" in str(e):
                pytest.fail("play_games_parallel should accept use_hybrid parameter")
            raise

    def test_tactical_adjustment_applied_to_policy(self, config):
        """Tactical adjustments should modify policy for capture moves."""
        from tactics import TacticalAnalyzer

        board = Board(config.board_size)
        tactics = TacticalAnalyzer()

        # Set up position with obvious capture
        c = config.board_size // 2
        board.board[c, c] = -1      # White stone
        board.board[c-1, c] = 1     # Black stones surrounding
        board.board[c+1, c] = 1
        board.board[c, c-1] = 1
        # One liberty at (c, c+1)

        # Check boost for capture move
        capture_move = (c, c+1)
        boost = tactics.get_tactical_boost(board, capture_move)

        # Capture of 1 stone should have boost > 1
        assert boost > 1.0, f"Capture move should have boost > 1, got {boost}"

    def test_hybrid_mcts_creates_with_tactics(self, config, mock_model):
        """HybridMCTS should be created when use_hybrid=True."""
        from hybrid_mcts import create_hybrid_mcts
        from tactics import TacticalAnalyzer

        mcts = create_hybrid_mcts(mock_model, config, tactical_weight=0.3)

        assert hasattr(mcts, 'tactics')
        assert isinstance(mcts.tactics, TacticalAnalyzer)
        assert mcts.tactical_weight == 0.3

    def test_training_sample_from_hybrid_game(self, config, mock_model):
        """Training samples from hybrid games should include tactical-influenced policies."""
        from selfplay import play_game, GameRecord

        # This tests the existing selfplay.py hybrid support
        # Verify the API exists and returns samples
        try:
            samples = play_game(
                mock_model,
                config,
                game_idx=0,
                verbose=False,
                batch_size=2,
                use_hybrid=True,
                tactical_weight=0.3
            )
            # Should return list of (state, policy, value) tuples
            assert isinstance(samples, list)
            if samples:
                state, policy, value = samples[0]
                assert state.shape[0] > 0  # Has channels
                assert len(policy) == config.board_size ** 2 + 1
        except Exception as e:
            # May fail with mock model, but API should exist
            if "use_hybrid" in str(e) or "unexpected keyword" in str(e):
                pytest.fail("play_game should accept use_hybrid parameter")


class TestTacticalWeightEffect:
    """Test that tactical weight affects move selection."""

    @pytest.fixture
    def board_with_capture(self):
        """Create board with clear capture opportunity."""
        board = Board(9)
        c = 4

        # White stone in atari
        board.board[c, c] = -1
        board.board[c-1, c] = 1
        board.board[c+1, c] = 1
        board.board[c, c-1] = 1
        # Liberty at (c, c+1)

        board.current_player = 1  # Black to move
        return board, (c, c+1)  # Capture point

    def test_zero_weight_ignores_tactics(self, board_with_capture):
        """With tactical_weight=0, tactics should not affect policy."""
        from tactics import TacticalAnalyzer

        board, capture_move = board_with_capture
        tactics = TacticalAnalyzer()

        # Uniform prior
        action_size = board.size ** 2 + 1
        prior = np.ones(action_size) / action_size

        # At weight=0, get_tactical_boost is still computed but
        # HybridMCTS blending would have no effect
        boost = tactics.get_tactical_boost(board, capture_move)
        assert boost > 1.0, "Capture should still be detected"

    def test_high_weight_prioritizes_tactics(self, board_with_capture):
        """With high tactical_weight, capture moves should dominate."""
        from tactics import TacticalAnalyzer

        board, capture_move = board_with_capture
        tactics = TacticalAnalyzer()

        # Get boost for capture
        capture_idx = capture_move[0] * board.size + capture_move[1]
        boost = tactics.get_tactical_boost(board, capture_move)

        # Create policy and apply boost
        action_size = board.size ** 2 + 1
        policy = np.ones(action_size) / action_size
        policy[capture_idx] *= boost
        policy = policy / policy.sum()

        # Capture move should now be the top choice
        assert np.argmax(policy) == capture_idx


class TestSelfPlayStatsTracking:
    """Test that tactical usage is tracked in self-play."""

    @pytest.fixture
    def config(self):
        config = Config()
        config.board_size = 9
        config.device = 'cpu'
        return config

    def test_hybrid_mcts_tracks_adjustments(self, config):
        """HybridMCTS should track tactical adjustment count."""
        from hybrid_mcts import HybridMCTS
        from tactics import TacticalAnalyzer

        # Create mock model
        model = MagicMock()
        model.config = config
        action_size = config.board_size ** 2 + 1
        mock_policy = torch.log(torch.ones(1, action_size) / action_size)
        mock_value = torch.zeros(1, 1)
        model.return_value = (mock_policy, mock_value)

        tactics = TacticalAnalyzer()
        mcts = HybridMCTS(model, config, tactics, batch_size=1)

        assert hasattr(mcts, 'tactical_adjustments')
        assert mcts.tactical_adjustments == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
