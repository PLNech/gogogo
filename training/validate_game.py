#!/usr/bin/env python3
"""Validate improved HybridAgent with a short game.

Plays a short game and analyzes tactical decisions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from board import Board
from config import DEFAULT
from model import load_checkpoint
from mcts import MCTS
from tactics import TacticalAnalyzer


def draw_board_state(ax, board: Board, title: str = "", last_move=None):
    """Draw board state on matplotlib axes."""
    size = board.size
    ax.set_facecolor('#DEB887')

    # Draw grid
    for i in range(size):
        ax.axhline(y=i, color='black', linewidth=0.5)
        ax.axvline(x=i, color='black', linewidth=0.5)

    # Draw star points
    if size == 19:
        stars = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        for r, c in stars:
            ax.plot(c, size - 1 - r, 'ko', markersize=3)

    # Draw stones
    for r in range(size):
        for c in range(size):
            stone = board.board[r, c]
            if stone == 1:
                color = 'black' if last_move != (r, c) else 'darkgreen'
                circle = Circle((c, size - 1 - r), 0.4, facecolor=color, edgecolor='black')
                ax.add_patch(circle)
            elif stone == -1:
                color = 'white' if last_move != (r, c) else 'lightcoral'
                circle = Circle((c, size - 1 - r), 0.4, facecolor=color, edgecolor='black')
                ax.add_patch(circle)

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def format_move(move, size):
    if move == (-1, -1):
        return "PASS"
    cols = "ABCDEFGHJKLMNOPQRST"[:size]
    return f"{cols[move[1]]}{size - move[0]}"


def validate_tactical_game():
    """Play a game and validate tactical decisions."""
    print("Loading model...")
    config = DEFAULT
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, step = load_checkpoint('checkpoints/supervised_best.pt', config)
    config = model.config
    config.mcts_simulations = 100
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Model step: {step}, Board: {config.board_size}x{config.board_size}")

    # Create analyzer and MCTS
    tactics = TacticalAnalyzer()
    mcts = MCTS(model, config, batch_size=8)

    board = Board(config.board_size)
    move_history = []
    tactical_moments = []

    print("\nPlaying game with tactical analysis...")
    print("-" * 50)

    max_moves = 60  # Play first 60 moves to see opening/midgame
    for move_num in range(max_moves):
        # Check if tactical position
        is_tactical = tactics.is_tactical_position(board)

        # Get move from MCTS
        policy = mcts.search(board, verbose=False)
        action_idx = np.argmax(policy)
        if action_idx == board.size ** 2:
            action = (-1, -1)
        else:
            action = (action_idx // board.size, action_idx % board.size)

        # Get tactical boost for this move
        if action != (-1, -1):
            boost = tactics.get_tactical_boost(board, action)
        else:
            boost = 1.0

        player = "B" if board.current_player == 1 else "W"
        move_str = format_move(action, board.size)

        # Record tactical moments
        if is_tactical or boost > 1.5:
            tactical_moments.append({
                'move_num': move_num + 1,
                'player': player,
                'move': move_str,
                'boost': boost,
                'board_copy': board.copy()
            })

        # Play move
        if action == (-1, -1):
            board.pass_move()
            print(f"{move_num+1:3d}. {player}: PASS")
        else:
            captures = board.play(action[0], action[1])
            cap_str = f" (cap {captures})" if captures > 0 else ""
            tac_str = f" [tactical x{boost:.1f}]" if boost > 1.5 else ""
            print(f"{move_num+1:3d}. {player}: {move_str}{cap_str}{tac_str}")

        move_history.append(action)

        # Check for game end
        if board.is_game_over():
            break

    # Create visualization of key moments
    print("\n" + "=" * 50)
    print("TACTICAL MOMENTS")
    print("=" * 50)

    if tactical_moments:
        n_moments = min(6, len(tactical_moments))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, moment in enumerate(tactical_moments[:n_moments]):
            draw_board_state(
                axes[i],
                moment['board_copy'],
                f"Move {moment['move_num']}: {moment['player']} {moment['move']}\n"
                f"Tactical boost: {moment['boost']:.2f}"
            )
            print(f"  Move {moment['move_num']}: {moment['player']} {moment['move']} (boost {moment['boost']:.2f})")

        # Hide unused subplots
        for i in range(n_moments, 6):
            axes[i].axis('off')

        plt.suptitle('Tactical Moments in Game', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('training_plots/game_tactical_moments.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: training_plots/game_tactical_moments.png")
    else:
        print("  No significant tactical moments detected")

    # Final position
    print("\n" + "=" * 50)
    print("GAME SUMMARY")
    print("=" * 50)
    print(f"Total moves: {len(move_history)}")
    print(f"Tactical moments: {len(tactical_moments)}")

    score = board.score()
    winner = "Black" if score > 0 else "White"
    print(f"Score: {abs(score):.1f} for {winner}")

    # Save final position
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_board_state(ax, board, f"Final Position (Move {len(move_history)})\nScore: {abs(score):.1f} for {winner}")
    plt.tight_layout()
    plt.savefig('training_plots/game_final_position.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: training_plots/game_final_position.png")


if __name__ == "__main__":
    validate_tactical_game()
