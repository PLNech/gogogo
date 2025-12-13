#!/usr/bin/env python3
"""Compare standard MCTS vs HybridMCTS tactical behavior.

Validates that the neurosymbolic approach improves tactical play by:
1. Setting up tactical positions (atari, ladders, snapbacks)
2. Comparing move selection between standard and hybrid MCTS
3. Measuring time and quality differences
"""

import argparse
import time
import numpy as np
import torch

from board import Board
from config import Config, DEFAULT
from model import load_checkpoint
from mcts import MCTS
from hybrid_mcts import HybridMCTS
from tactics import TacticalAnalyzer


def format_move(move, board_size):
    """Format move as Go coordinates."""
    if move == (-1, -1):
        return "PASS"
    r, c = move
    cols = "ABCDEFGHJKLMNOPQRST"[:board_size]
    return f"{cols[c]}{board_size - r}"


def board_to_str(board: Board, highlight=None) -> str:
    """Simple board string representation."""
    cols = "ABCDEFGHJKLMNOPQRST"[:board.size]
    header = "   " + " ".join(cols)
    lines = [header]

    for r in range(board.size):
        row_str = f"{board.size - r:2d} "
        for c in range(board.size):
            stone = board.board[r, c]
            if highlight and (r, c) == highlight:
                if stone == 1:
                    row_str += "\033[1;32m@\033[0m "
                elif stone == -1:
                    row_str += "\033[1;31m#\033[0m "
                else:
                    row_str += "\033[1;33m*\033[0m "
            elif stone == 1:
                row_str += "X "
            elif stone == -1:
                row_str += "O "
            else:
                row_str += ". "
        lines.append(row_str)

    lines.append(header)
    return "\n".join(lines)


def get_top_moves(policy, board_size, top_n=5):
    """Get top N moves from policy."""
    indices = np.argsort(policy)[-top_n:][::-1]
    moves = []
    for idx in indices:
        if idx == board_size ** 2:
            move = (-1, -1)
        else:
            move = (idx // board_size, idx % board_size)
        prob = policy[idx]
        moves.append((move, prob))
    return moves


def setup_atari_position(board_size=9):
    """Create a position with a black group in atari (1 liberty)."""
    board = Board(board_size)
    center = board_size // 2

    # Black stone at center
    board.board[center, center] = 1  # Black

    # Surround with white leaving exactly ONE liberty at (center-1, center)
    board.board[center+1, center] = -1  # White below
    board.board[center, center+1] = -1  # White right
    board.board[center, center-1] = -1  # White left
    # Leave (center-1, center) open as the only liberty

    board.current_player = -1  # White to play (can capture at center-1, center)
    return board


def setup_ladder_position(board_size=9):
    """Create a position where white can start a ladder on black."""
    board = Board(board_size)

    # Black group in atari with ladder potential
    # Black at (3,3), white at (3,4) and (4,3) - black has 2 libs at (2,3) and (3,2)
    board.board[3, 3] = 1   # Black
    board.board[3, 4] = -1  # White right
    board.board[4, 3] = -1  # White below
    # Black has 2 liberties: (2,3) and (3,2)
    # If white plays (2,3), black must extend to (3,2), then ladder continues

    board.current_player = -1  # White to play ladder
    return board


def setup_capture_race_position(board_size=9):
    """Create a position with adjacent groups, one in atari."""
    board = Board(board_size)

    # Black group with 2 liberties
    board.board[3, 3] = 1   # Black
    board.board[3, 4] = 1   # Black
    board.board[2, 3] = -1  # White above
    board.board[2, 4] = -1  # White above
    board.board[4, 3] = -1  # White below left
    # Black has 2 liberties: (3,2) and (4,4)

    # Add more white to make it more tactical
    board.board[3, 5] = -1  # White right - now black has 1 liberty at (3,2)!

    board.current_player = -1  # White to play (can capture at 3,2)
    return board


def compare_on_position(board, model, config, tactics, name="Position"):
    """Compare standard and hybrid MCTS on a position."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)

    print(f"\nBoard ({['Black', 'White'][board.current_player == -1]} to play):")
    print(board_to_str(board))

    # Check if tactical
    is_tactical = tactics.is_tactical_position(board)
    print(f"\nTactical position: {is_tactical}")

    # Standard MCTS
    std_mcts = MCTS(model, config, batch_size=8)
    start = time.time()
    std_policy = std_mcts.search(board, verbose=False)
    std_time = time.time() - start
    std_moves = get_top_moves(std_policy, config.board_size)

    # Hybrid MCTS
    hybrid_mcts = HybridMCTS(model, config, tactics, batch_size=8, tactical_weight=0.3)
    start = time.time()
    hybrid_policy = hybrid_mcts.search(board, verbose=False)
    hybrid_time = time.time() - start
    hybrid_moves = get_top_moves(hybrid_policy, config.board_size)

    # Display comparison
    print(f"\n{'Standard MCTS':^28} | {'Hybrid MCTS':^28}")
    print(f"{'Time: {:.3f}s'.format(std_time):^28} | {'Time: {:.3f}s'.format(hybrid_time):^28}")
    print("-" * 60)

    for i in range(5):
        std_move, std_prob = std_moves[i]
        hyb_move, hyb_prob = hybrid_moves[i]

        std_str = f"{format_move(std_move, config.board_size):4s} {std_prob:5.1%}"
        hyb_str = f"{format_move(hyb_move, config.board_size):4s} {hyb_prob:5.1%}"

        # Highlight differences
        if std_move != hyb_move:
            print(f"  {i+1}. {std_str:20s}  |   {i+1}. \033[1;33m{hyb_str}\033[0m")
        else:
            print(f"  {i+1}. {std_str:20s}  |   {i+1}. {hyb_str}")

    # Check for tactical adjustments
    diff = hybrid_policy - std_policy
    max_boost_idx = np.argmax(diff)
    max_penalty_idx = np.argmin(diff)

    if max_boost_idx < config.board_size ** 2:
        boost_move = (max_boost_idx // config.board_size, max_boost_idx % config.board_size)
        boost_str = format_move(boost_move, config.board_size)
    else:
        boost_str = "PASS"

    if max_penalty_idx < config.board_size ** 2:
        penalty_move = (max_penalty_idx // config.board_size, max_penalty_idx % config.board_size)
        penalty_str = format_move(penalty_move, config.board_size)
    else:
        penalty_str = "PASS"

    print(f"\nTactical adjustments:")
    print(f"  Biggest boost:   {boost_str:4s} (+{diff[max_boost_idx]:+.3f})")
    print(f"  Biggest penalty: {penalty_str:4s} ({diff[max_penalty_idx]:+.3f})")

    return std_moves[0][0], hybrid_moves[0][0]


def play_comparison_game(model, config, tactics, max_moves=60):
    """Play a game comparing standard vs hybrid move choices."""
    print(f"\n{'#'*60}")
    print("COMPARISON GAME: Standard vs Hybrid MCTS")
    print('#'*60)

    board = Board(config.board_size)
    std_mcts = MCTS(model, config, batch_size=8)
    hybrid_mcts = HybridMCTS(model, config, tactics, batch_size=8)

    disagreements = []
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        # Get both policies
        std_policy = std_mcts.search(board, verbose=False)
        hybrid_policy = hybrid_mcts.search(board, verbose=False)

        std_action = np.argmax(std_policy)
        hybrid_action = np.argmax(hybrid_policy)

        if std_action != hybrid_action:
            std_move = (-1, -1) if std_action == config.board_size ** 2 else (std_action // config.board_size, std_action % config.board_size)
            hyb_move = (-1, -1) if hybrid_action == config.board_size ** 2 else (hybrid_action // config.board_size, hybrid_action % config.board_size)

            is_tactical = tactics.is_tactical_position(board)
            disagreements.append({
                'move': move_count + 1,
                'std': format_move(std_move, config.board_size),
                'hybrid': format_move(hyb_move, config.board_size),
                'tactical': is_tactical
            })

        # Play hybrid's move (presumably better in tactical situations)
        if hybrid_action == config.board_size ** 2:
            board.pass_move()
        else:
            r, c = hybrid_action // config.board_size, hybrid_action % config.board_size
            board.play(r, c)

        move_count += 1

    # Summary
    print(f"\nGame finished after {move_count} moves")
    print(f"Score: {board.score():+.1f}")
    print(f"\nDisagreements: {len(disagreements)} moves ({100*len(disagreements)/move_count:.1f}%)")

    if disagreements:
        print("\nKey disagreements:")
        for d in disagreements[:10]:  # Show first 10
            tactical_mark = " [TACTICAL]" if d['tactical'] else ""
            print(f"  Move {d['move']:3d}: Std={d['std']:4s} vs Hybrid={d['hybrid']:4s}{tactical_mark}")

    return disagreements


def main():
    parser = argparse.ArgumentParser(description='Compare MCTS variants')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/supervised_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--simulations', type=int, default=100,
                        help='MCTS simulations per move')
    parser.add_argument('--game', action='store_true',
                        help='Play a full comparison game')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    config = DEFAULT
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, step = load_checkpoint(args.checkpoint, config)
    config = model.config  # Use config from checkpoint
    config.mcts_simulations = args.simulations
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Model loaded (step {step})")
    print(f"Board size: {config.board_size}x{config.board_size}")
    print(f"Device: {config.device}")
    print(f"Simulations: {config.mcts_simulations}")

    # Initialize tactical analyzer
    tactics = TacticalAnalyzer()

    # Test on specific positions
    positions = [
        (setup_atari_position(config.board_size), "Atari Position"),
        (setup_ladder_position(config.board_size), "Ladder Setup"),
        (setup_capture_race_position(config.board_size), "Capture Race"),
    ]

    for board, name in positions:
        if board.size == config.board_size:
            compare_on_position(board, model, config, tactics, name)
        else:
            print(f"\nSkipping {name} - board size mismatch")

    # Also test on empty board opening
    empty_board = Board(config.board_size)
    compare_on_position(empty_board, model, config, tactics, "Empty Board (Opening)")

    # Play comparison game if requested
    if args.game:
        play_comparison_game(model, config, tactics)


if __name__ == '__main__':
    main()
