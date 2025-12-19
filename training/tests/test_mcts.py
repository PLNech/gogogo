"""Tests for MCTS KataGo improvements (A.4, A.5).

Testing the techniques we shipped without tests. Paying our debt.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add training directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from mcts import MCTS


class TestShapedDirichletNoise:
    """Test KataGo A.4: Shaped Dirichlet noise at root."""

    @pytest.fixture
    def mcts(self):
        """Create MCTS instance with mock model."""
        config = Config(board_size=9, shaped_dirichlet=True)
        mock_model = MagicMock()
        return MCTS(mock_model, config, use_cache=False)

    @pytest.fixture
    def mcts_uniform(self):
        """MCTS with uniform (non-shaped) Dirichlet."""
        config = Config(board_size=9, shaped_dirichlet=False)
        mock_model = MagicMock()
        return MCTS(mock_model, config, use_cache=False)

    def test_output_is_normalized(self, mcts):
        """Noisy policy should sum to 1."""
        action_size = 9 * 9 + 1  # 82
        policy = np.ones(action_size) / action_size
        legal_mask = np.ones(action_size)

        noisy = mcts._add_dirichlet_noise(policy, legal_mask)

        assert np.isclose(noisy.sum(), 1.0, atol=1e-6)

    def test_output_is_nonnegative(self, mcts):
        """All policy values should be non-negative."""
        action_size = 9 * 9 + 1
        policy = np.ones(action_size) / action_size
        legal_mask = np.ones(action_size)

        noisy = mcts._add_dirichlet_noise(policy, legal_mask)

        assert (noisy >= 0).all()

    def test_legal_mask_respected(self, mcts):
        """Illegal moves should have zero probability."""
        action_size = 9 * 9 + 1
        policy = np.ones(action_size) / action_size

        # Only corners and pass are legal
        legal_mask = np.zeros(action_size)
        legal_mask[0] = 1  # (0,0)
        legal_mask[8] = 1  # (0,8)
        legal_mask[72] = 1  # (8,0)
        legal_mask[80] = 1  # (8,8)
        legal_mask[81] = 1  # pass

        noisy = mcts._add_dirichlet_noise(policy, legal_mask)

        # Illegal moves should be zero
        illegal_indices = np.where(legal_mask == 0)[0]
        assert (noisy[illegal_indices] == 0).all()

        # Legal moves should have non-zero probability
        legal_indices = np.where(legal_mask == 1)[0]
        assert (noisy[legal_indices] > 0).all()

    def test_shaped_concentrates_on_high_policy(self, mcts):
        """Shaped noise should favor moves with higher policy values.

        With shaped Dirichlet, high-policy moves get more noise weight,
        so they remain relatively higher after noise is added.
        """
        action_size = 9 * 9 + 1
        legal_mask = np.ones(action_size)

        # Create peaked policy: one move has 90% probability
        peaked_policy = np.full(action_size, 0.1 / (action_size - 1))
        peaked_policy[0] = 0.9

        # Run many times and average (noise is stochastic)
        np.random.seed(42)
        samples = [mcts._add_dirichlet_noise(peaked_policy.copy(), legal_mask) for _ in range(100)]
        avg_noisy = np.mean(samples, axis=0)

        # The originally high-probability move should still be highest on average
        assert np.argmax(avg_noisy) == 0
        # And should retain significant probability
        assert avg_noisy[0] > 0.5

    def test_fraction_zero_is_identity(self, mcts):
        """With exploration_fraction=0, policy should be unchanged."""
        mcts.config.root_exploration_fraction = 0.0

        action_size = 9 * 9 + 1
        policy = np.random.dirichlet(np.ones(action_size))
        legal_mask = np.ones(action_size)

        noisy = mcts._add_dirichlet_noise(policy.copy(), legal_mask)

        np.testing.assert_array_almost_equal(noisy, policy)

    def test_uniform_vs_shaped_differ(self, mcts, mcts_uniform):
        """Shaped and uniform noise should produce different distributions."""
        action_size = 9 * 9 + 1
        legal_mask = np.ones(action_size)

        # Peaked policy
        policy = np.full(action_size, 0.01)
        policy[0] = 0.5
        policy[1] = 0.3
        policy = policy / policy.sum()

        # Same seed for both
        np.random.seed(42)
        shaped = mcts._add_dirichlet_noise(policy.copy(), legal_mask)

        np.random.seed(42)
        uniform = mcts_uniform._add_dirichlet_noise(policy.copy(), legal_mask)

        # They should differ (shaped concentrates noise differently)
        assert not np.allclose(shaped, uniform)


class TestRootPolicyTemperature:
    """Test KataGo A.5: Root policy softmax temperature."""

    @pytest.fixture
    def mcts(self):
        """Create MCTS with temperature config."""
        config = Config(
            board_size=9,
            root_policy_temp=1.1,
            root_policy_temp_early=1.25,
            root_policy_temp_early_moves=30
        )
        mock_model = MagicMock()
        return MCTS(mock_model, config, use_cache=False)

    def test_temp_one_is_identity(self, mcts):
        """Temperature 1.0 should return input unchanged."""
        mcts.config.root_policy_temp = 1.0
        mcts.config.root_policy_temp_early = 1.0

        action_size = 9 * 9 + 1
        policy = np.random.dirichlet(np.ones(action_size))

        result = mcts._apply_root_policy_temp(policy.copy(), move_number=50)

        np.testing.assert_array_almost_equal(result, policy, decimal=5)

    def test_higher_temp_flattens_distribution(self, mcts):
        """Temperature > 1 should make distribution more uniform."""
        mcts.config.root_policy_temp = 2.0
        mcts.config.root_policy_temp_early = 2.0

        action_size = 9 * 9 + 1
        # Peaked distribution
        policy = np.full(action_size, 0.001)
        policy[0] = 0.9
        policy = policy / policy.sum()

        original_entropy = -np.sum(policy * np.log(policy + 1e-10))

        result = mcts._apply_root_policy_temp(policy, move_number=50)
        result_entropy = -np.sum(result * np.log(result + 1e-10))

        # Higher temperature → higher entropy (more uniform)
        assert result_entropy > original_entropy
        # Peak should be lower
        assert result[0] < policy[0]

    def test_lower_temp_sharpens_distribution(self, mcts):
        """Temperature < 1 should make distribution more peaked."""
        mcts.config.root_policy_temp = 0.5
        mcts.config.root_policy_temp_early = 0.5

        action_size = 9 * 9 + 1
        # Slightly peaked distribution
        policy = np.full(action_size, 0.01)
        policy[0] = 0.2
        policy = policy / policy.sum()

        original_entropy = -np.sum(policy * np.log(policy + 1e-10))

        result = mcts._apply_root_policy_temp(policy, move_number=50)
        result_entropy = -np.sum(result * np.log(result + 1e-10))

        # Lower temperature → lower entropy (more peaked)
        assert result_entropy < original_entropy
        # Peak should be higher
        assert result[0] > policy[0]

    def test_early_vs_late_game_temp(self, mcts):
        """Early game should use higher temperature than late game."""
        mcts.config.root_policy_temp = 1.1
        mcts.config.root_policy_temp_early = 1.5
        mcts.config.root_policy_temp_early_moves = 30

        action_size = 9 * 9 + 1
        policy = np.full(action_size, 0.01)
        policy[0] = 0.5
        policy = policy / policy.sum()

        early_result = mcts._apply_root_policy_temp(policy.copy(), move_number=10)
        late_result = mcts._apply_root_policy_temp(policy.copy(), move_number=50)

        # Early game (higher temp) should be more uniform
        early_entropy = -np.sum(early_result * np.log(early_result + 1e-10))
        late_entropy = -np.sum(late_result * np.log(late_result + 1e-10))

        assert early_entropy > late_entropy

    def test_output_is_normalized(self, mcts):
        """Temperature-adjusted policy should sum to 1."""
        action_size = 9 * 9 + 1
        policy = np.random.dirichlet(np.ones(action_size))

        for temp in [0.5, 1.0, 1.5, 2.0]:
            mcts.config.root_policy_temp = temp
            result = mcts._apply_root_policy_temp(policy.copy(), move_number=50)
            assert np.isclose(result.sum(), 1.0, atol=1e-6)

    def test_output_is_nonnegative(self, mcts):
        """All values should be non-negative."""
        action_size = 9 * 9 + 1
        policy = np.random.dirichlet(np.ones(action_size))

        result = mcts._apply_root_policy_temp(policy, move_number=10)

        assert (result >= 0).all()


class TestDirichletEdgeCases:
    """Edge cases for Dirichlet noise."""

    @pytest.fixture
    def mcts(self):
        config = Config(board_size=9, shaped_dirichlet=True)
        mock_model = MagicMock()
        return MCTS(mock_model, config, use_cache=False)

    def test_single_legal_move(self, mcts):
        """With one legal move, it should get all probability."""
        action_size = 9 * 9 + 1
        policy = np.ones(action_size) / action_size

        legal_mask = np.zeros(action_size)
        legal_mask[42] = 1  # Only one legal move

        noisy = mcts._add_dirichlet_noise(policy, legal_mask)

        assert noisy[42] == 1.0
        assert noisy.sum() == 1.0

    def test_zero_policy_legal_moves_uniform(self, mcts):
        """With uniform noise, legal moves with zero policy should get probability.

        Note: Shaped Dirichlet concentrates noise on high-policy moves,
        so zero-policy moves may get near-zero noise. Use uniform for this test.
        """
        # Switch to uniform noise for this test
        mcts.config.shaped_dirichlet = False

        action_size = 9 * 9 + 1
        policy = np.zeros(action_size)
        policy[0] = 1.0  # All mass on one move

        legal_mask = np.ones(action_size)

        noisy = mcts._add_dirichlet_noise(policy, legal_mask)

        # With uniform noise, other legal moves should have some probability
        assert noisy[1] > 0
        assert noisy.sum() == pytest.approx(1.0)

    def test_no_legal_moves(self, mcts):
        """With no legal moves, should return original policy."""
        action_size = 9 * 9 + 1
        policy = np.ones(action_size) / action_size
        legal_mask = np.zeros(action_size)

        noisy = mcts._add_dirichlet_noise(policy.copy(), legal_mask)

        np.testing.assert_array_equal(noisy, policy)
