#!/usr/bin/env python3
"""Visualize network policy as heatmap on Go board."""
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from pathlib import Path

from board import Board
from model import load_checkpoint
from config import Config


def draw_go_board(ax, size: int):
    """Draw empty Go board grid."""
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # Draw grid lines
    for i in range(size):
        ax.axhline(y=i, color='black', linewidth=0.5)
        ax.axvline(x=i, color='black', linewidth=0.5)

    # Star points (for 19x19)
    if size == 19:
        star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        for r, c in star_points:
            ax.add_patch(Circle((c, r), 0.1, color='black'))
    elif size == 9:
        star_points = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        for r, c in star_points:
            ax.add_patch(Circle((c, r), 0.08, color='black'))

    # Labels
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels([str(i) for i in range(size)])
    ax.set_yticklabels([str(i) for i in range(size)])

    ax.set_facecolor('#DEB887')  # Wood color


def draw_stones(ax, board: Board):
    """Draw stones on board."""
    size = board.size
    for r in range(size):
        for c in range(size):
            if board.board[r, c] == 1:  # Black
                ax.add_patch(Circle((c, r), 0.4, color='black', zorder=10))
            elif board.board[r, c] == -1:  # White
                ax.add_patch(Circle((c, r), 0.4, color='white', edgecolor='black', linewidth=1, zorder=10))


def draw_policy_heatmap(ax, policy: np.ndarray, board: Board, alpha: float = 0.6):
    """Draw policy as heatmap overlay."""
    size = board.size
    board_policy = policy[:-1].reshape(size, size)

    # Mask out occupied positions
    mask = board.board != 0
    board_policy_masked = np.ma.array(board_policy, mask=mask)

    # Draw heatmap
    im = ax.imshow(board_policy_masked, cmap='YlOrRd', alpha=alpha,
                   extent=[-0.5, size-0.5, size-0.5, -0.5], zorder=5)
    return im


def draw_top_moves(ax, policy: np.ndarray, board: Board, top_k: int = 5):
    """Draw markers for top-k moves."""
    size = board.size
    board_policy = policy[:-1].reshape(size, size)

    # Get top moves
    flat_indices = np.argsort(board_policy.flatten())[::-1]

    count = 0
    for idx in flat_indices:
        if count >= top_k:
            break
        r, c = idx // size, idx % size
        if board.board[r, c] == 0:  # Only empty points
            prob = board_policy[r, c]
            # Draw number
            ax.text(c, r, str(count + 1), ha='center', va='center',
                   fontsize=12, fontweight='bold', color='blue', zorder=20)
            count += 1


def get_policy(model, board: Board, use_tactical: bool = True):
    """Get policy and value from model."""
    model.eval()
    with torch.no_grad():
        tensor = board.to_tensor(use_tactical_features=use_tactical)
        x = torch.FloatTensor(tensor).unsqueeze(0).to(next(model.parameters()).device)
        log_policy, value = model(x)
        policy = torch.exp(log_policy).cpu().numpy()[0]
        value = value.cpu().numpy()[0, 0]
    return policy, value


def visualize_position(model, board: Board, use_tactical: bool = True,
                       output_path: str = None, title: str = None):
    """Create visualization of policy on position."""
    policy, value = get_policy(model, board, use_tactical)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw board
    draw_go_board(ax, board.size)
    draw_stones(ax, board)

    # Draw policy heatmap
    im = draw_policy_heatmap(ax, policy, board)

    # Draw top moves
    draw_top_moves(ax, policy, board, top_k=5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Move Probability', fontsize=12)

    # Title
    if title:
        ax.set_title(title, fontsize=14)
    else:
        to_play = 'Black' if board.current_player == 1 else 'White'
        ax.set_title(f'{to_play} to play | Value: {value:+.3f}', fontsize=14)

    # Pass probability
    pass_prob = policy[-1]
    ax.text(0.02, 0.02, f'Pass: {pass_prob:.1%}', transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()
    return policy, value


def create_test_position(position_name: str) -> Board:
    """Create a test position by name."""
    if position_name == 'empty':
        return Board(19)

    elif position_name == 'opening':
        board = Board(19)
        # Standard opening moves
        moves = [(3, 3), (15, 15), (3, 15), (15, 3)]
        for r, c in moves:
            board.play(r, c)
        return board

    elif position_name == 'capture':
        board = Board(9)
        # Black stones surrounding white
        board.board[4, 3] = 1
        board.board[4, 5] = 1
        board.board[3, 4] = 1
        board.board[4, 4] = -1  # White to capture
        board.current_player = 1
        return board

    elif position_name == 'joseki':
        board = Board(19)
        board.board[3, 3] = 1  # Black 4-4
        board.board[2, 5] = -1  # White approach
        board.current_player = 1
        return board

    elif position_name == 'corner':
        board = Board(19)
        # Corner position
        board.board[2, 2] = 1
        board.board[2, 3] = -1
        board.board[3, 2] = -1
        board.current_player = 1
        return board

    else:
        print(f"Unknown position: {position_name}")
        return Board(19)


def main():
    parser = argparse.ArgumentParser(description='Visualize network policy')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--position', type=str, default='empty',
                        choices=['empty', 'opening', 'capture', 'joseki', 'corner'],
                        help='Test position to visualize')
    parser.add_argument('--output', type=str, default='policy_viz.png',
                        help='Output image path')
    parser.add_argument('--no-tactical', action='store_true',
                        help='Disable tactical features')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    config = Config()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, step = load_checkpoint(args.checkpoint, config)
    print(f"Loaded model (step {step}, board_size={model.config.board_size})")

    use_tactical = not args.no_tactical and model.config.input_planes == 27

    # Create test position
    board = create_test_position(args.position)

    # Resize if needed
    if board.size != model.config.board_size:
        print(f"Creating position on {model.config.board_size}x{model.config.board_size} board")
        board = create_test_position(args.position)
        if board.size != model.config.board_size:
            board = Board(model.config.board_size)

    # Visualize
    visualize_position(model, board, use_tactical, args.output,
                      title=f'Policy: {args.position} position')


if __name__ == '__main__':
    main()
