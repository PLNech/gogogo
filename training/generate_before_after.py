#!/usr/bin/env python3
"""Generate before/after self-play game visualizations.

Shows how games look with untrained vs curriculum-trained model.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

from board import Board
from model import GoNet
from config import Config


def play_game(model, config, max_moves=50, temperature=1.0):
    """Play a self-play game and return final board."""
    board = Board(config.board_size)
    use_tactical = getattr(config, 'tactical_features', False) or config.input_planes == 27

    model.eval()
    moves_played = 0
    consecutive_passes = 0

    with torch.no_grad():
        while moves_played < max_moves and consecutive_passes < 2:
            tensor = board.to_tensor(use_tactical_features=use_tactical)
            x = torch.FloatTensor(tensor).unsqueeze(0).to(config.device)

            log_policy, _ = model(x)
            policy = torch.exp(log_policy).cpu().numpy()[0]

            # Mask illegal moves
            legal = board.get_legal_moves()
            legal_mask = np.zeros(config.board_size ** 2 + 1)
            for move in legal:
                if move == (-1, -1):
                    legal_mask[-1] = 1  # Pass
                else:
                    idx = move[0] * config.board_size + move[1]
                    legal_mask[idx] = 1

            policy = policy * legal_mask
            if policy.sum() > 0:
                policy = policy / policy.sum()
            else:
                # All illegal - pass
                policy[-1] = 1.0

            # Sample move with temperature
            if temperature > 0:
                policy = policy ** (1 / temperature)
                policy = policy / policy.sum()
                move_idx = np.random.choice(len(policy), p=policy)
            else:
                move_idx = np.argmax(policy)

            if move_idx == config.board_size ** 2:
                # Pass
                consecutive_passes += 1
                board.current_player = -board.current_player
            else:
                r, c = move_idx // config.board_size, move_idx % config.board_size
                if board.is_valid_move(r, c):
                    board.play(r, c)
                    consecutive_passes = 0
                else:
                    consecutive_passes += 1
                    board.current_player = -board.current_player

            moves_played += 1

    return board, moves_played


def draw_board(ax, board, title=""):
    """Draw a Go board on matplotlib axes."""
    size = board.size

    # Board background
    ax.set_facecolor('#DEB887')

    # Grid
    for i in range(size):
        ax.axhline(i, color='#4A4A4A', linewidth=0.8)
        ax.axvline(i, color='#4A4A4A', linewidth=0.8)

    # Star points for 9x9
    if size == 9:
        stars = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        for r, c in stars:
            ax.plot(c, r, 'o', color='#4A4A4A', markersize=4)

    # Stones
    for r in range(size):
        for c in range(size):
            if board.board[r, c] == 1:  # Black
                circle = Circle((c, r), 0.4, color='#1a1a1a', zorder=10)
                ax.add_patch(circle)
            elif board.board[r, c] == -1:  # White
                circle = Circle((c, r), 0.4, color='#F5F5DC', ec='#888', linewidth=1, zorder=10)
                ax.add_patch(circle)

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)  # Flip y for Go convention
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, fontweight='bold')


def generate_comparison(num_games=5, output_path='training_plots/before_after_games.png'):
    """Generate before/after comparison image."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create untrained model
    config_before = Config(board_size=9)
    config_before.device = device
    model_before = GoNet(config_before).to(device)
    print("Created untrained model")

    # Create/load trained model
    config_after = Config(board_size=9)
    config_after.device = device
    model_after = GoNet(config_after).to(device)

    # Try to load curriculum-trained model
    checkpoint_path = Path('checkpoints/curriculum_trained.pt')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model_after.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained model from {checkpoint_path}")
    else:
        # Train quickly on instincts
        print("Training model on instincts...")
        from train_curriculum import train_with_curriculum
        train_with_curriculum(model_after, config_after, epochs=20, verbose=False)

    # Generate games
    print(f"\nGenerating {num_games} games each...")

    fig, axes = plt.subplots(num_games, 2, figsize=(8, 4 * num_games))
    fig.suptitle('Self-Play Games: Before vs After Curriculum Training',
                 fontsize=14, fontweight='bold', y=0.98)

    np.random.seed(42)  # Reproducible

    for i in range(num_games):
        # Before (untrained)
        torch.manual_seed(i * 100)
        np.random.seed(i * 100)
        board_before, moves_before = play_game(model_before, config_before, max_moves=40, temperature=1.0)

        # After (trained)
        torch.manual_seed(i * 100)
        np.random.seed(i * 100)
        board_after, moves_after = play_game(model_after, config_after, max_moves=40, temperature=1.0)

        # Count stones
        black_before = np.sum(board_before.board == 1)
        white_before = np.sum(board_before.board == -1)
        black_after = np.sum(board_after.board == 1)
        white_after = np.sum(board_after.board == -1)

        ax_before = axes[i, 0] if num_games > 1 else axes[0]
        ax_after = axes[i, 1] if num_games > 1 else axes[1]

        draw_board(ax_before, board_before,
                   f"BEFORE (Game {i+1})\n{moves_before} moves | B:{black_before} W:{white_before}")
        draw_board(ax_after, board_after,
                   f"AFTER (Game {i+1})\n{moves_after} moves | B:{black_after} W:{white_after}")

    # Add labels
    fig.text(0.25, 0.02, 'Untrained Model\n(Random patterns, no captures)',
             ha='center', fontsize=10, style='italic', color='#666')
    fig.text(0.75, 0.02, 'Curriculum-Trained Model\n(Captures, connections, structure)',
             ha='center', fontsize=10, style='italic', color='#666')

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    generate_comparison(num_games=5)
