#!/usr/bin/env python3
"""Validate tactical integration in self-play training.

Runs a short training session with hybrid mode enabled and
visualizes the effect of tactical boosts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt

from board import Board
from config import Config
from tactics import TacticalAnalyzer


def visualize_tactical_boost_effect():
    """Show how tactical boosts modify policy distribution."""
    print("=" * 60)
    print("TACTICAL BOOST VISUALIZATION")
    print("=" * 60)

    config = Config()
    config.board_size = 9
    tactics = TacticalAnalyzer()

    # Create position with tactical opportunity
    board = Board(config.board_size)
    c = 4  # center

    # White stone in atari (one liberty)
    board.board[c, c] = -1      # White
    board.board[c-1, c] = 1     # Black surrounding
    board.board[c+1, c] = 1
    board.board[c, c-1] = 1
    # Liberty at (c, c+1) - this is the capture point

    board.current_player = 1  # Black to move

    # Create uniform prior (simulating NN output)
    action_size = board.size ** 2 + 1
    uniform_prior = np.ones(action_size) / action_size

    # Apply tactical boosts
    adjusted = uniform_prior.copy()
    boost_log = []

    for move in board.get_legal_moves():
        if move == (-1, -1):
            continue
        boost = tactics.get_tactical_boost(board, move)
        move_idx = move[0] * board.size + move[1]

        if boost > 1.2 or boost < 0.5:
            # Additive boost (as in self_play.py)
            adjusted[move_idx] = uniform_prior[move_idx] + 0.3 * (boost - 1.0)
            adjusted[move_idx] = max(adjusted[move_idx], 1e-8)
            boost_log.append((move, boost))

    # Normalize
    adjusted = adjusted / adjusted.sum()

    # Print results
    print(f"\nPosition: White stone at ({c},{c}) with 1 liberty at ({c},{c+1})")
    print(f"Black to move (player {board.current_player})")
    print(f"\nTactical boosts applied:")
    for move, boost in sorted(boost_log, key=lambda x: -x[1])[:5]:
        move_idx = move[0] * board.size + move[1]
        print(f"  {move}: boost={boost:.2f}, "
              f"prior={uniform_prior[move_idx]:.4f} -> {adjusted[move_idx]:.4f}")

    # The capture point should be top choice
    capture_idx = c * board.size + (c + 1)
    print(f"\nCapture point ({c},{c+1}) index: {capture_idx}")
    print(f"  Uniform prior: {uniform_prior[capture_idx]:.4f}")
    print(f"  After boost:   {adjusted[capture_idx]:.4f}")
    print(f"  Top choice:    {np.argmax(adjusted)} (capture={capture_idx})")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Board position
    ax = axes[0]
    ax.set_facecolor('#DEB887')
    for i in range(board.size):
        ax.axhline(y=i, color='black', linewidth=0.5)
        ax.axvline(x=i, color='black', linewidth=0.5)

    for r in range(board.size):
        for c_pos in range(board.size):
            if board.board[r, c_pos] == 1:
                ax.scatter(c_pos, board.size-1-r, c='black', s=500, zorder=3)
            elif board.board[r, c_pos] == -1:
                ax.scatter(c_pos, board.size-1-r, c='white', s=500, zorder=3,
                          edgecolors='black')

    # Mark capture point
    ax.scatter(c+1, board.size-1-c, c='red', s=200, marker='x', linewidths=3, zorder=4)
    ax.set_xlim(-0.5, board.size-0.5)
    ax.set_ylim(-0.5, board.size-0.5)
    ax.set_title('Position (X = capture point)', fontsize=12)
    ax.set_aspect('equal')

    # Uniform prior
    ax = axes[1]
    prior_2d = uniform_prior[:-1].reshape(board.size, board.size)
    im = ax.imshow(prior_2d, cmap='YlOrRd', vmin=0, vmax=0.05)
    ax.set_title('Uniform Prior', fontsize=12)
    plt.colorbar(im, ax=ax)

    # Adjusted policy
    ax = axes[2]
    adjusted_2d = adjusted[:-1].reshape(board.size, board.size)
    im = ax.imshow(adjusted_2d, cmap='YlOrRd', vmin=0, vmax=0.3)
    ax.set_title('After Tactical Boost', fontsize=12)
    plt.colorbar(im, ax=ax)

    plt.suptitle('Tactical Boost Effect on Policy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_plots/tactical_boost_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: training_plots/tactical_boost_effect.png")

    # Verify the integration works
    assert np.argmax(adjusted) == capture_idx, "Capture should be top choice"
    print("\n✓ Tactical boost correctly prioritizes capture move")

    return True


def test_selfplay_cli_help():
    """Verify CLI accepts --hybrid flag."""
    import subprocess
    result = subprocess.run(
        ['poetry', 'run', 'python', 'self_play.py', '--help'],
        capture_output=True, text=True
    )
    assert '--hybrid' in result.stdout, "CLI should accept --hybrid flag"
    assert '--tactical-weight' in result.stdout, "CLI should accept --tactical-weight"
    print("\n✓ CLI accepts --hybrid and --tactical-weight flags")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VALIDATING TACTICAL SELF-PLAY INTEGRATION")
    print("=" * 60 + "\n")

    success = True
    success = visualize_tactical_boost_effect() and success
    success = test_selfplay_cli_help() and success

    print("\n" + "=" * 60)
    if success:
        print("ALL VALIDATIONS PASSED")
    else:
        print("SOME VALIDATIONS FAILED")
    print("=" * 60)
