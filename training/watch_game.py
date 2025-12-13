#!/usr/bin/env python3
"""Watch AI games and validate quality.

This script plays games with the trained model and displays them move-by-move
to verify the AI is learning proper Go concepts.

Usage:
    # Watch a game with the latest checkpoint
    python watch_game.py --checkpoint checkpoints/supervised_best.pt

    # Watch with HybridMCTS (tactical enhancements)
    python watch_game.py --checkpoint checkpoints/supervised_best.pt --hybrid

    # Watch multiple games
    python watch_game.py --checkpoint checkpoints/supervised_best.pt --games 3
"""

import argparse
import time
import sys
import numpy as np
import torch

from board import Board
from config import Config, DEFAULT
from model import load_checkpoint
from mcts import MCTS
from selfplay import get_game_stats


def board_to_str(board: Board, last_move=None, move_num=0) -> str:
    """Convert board to colored ASCII string."""
    # Column labels
    cols = "ABCDEFGHJKLMNOPQRST"[:board.size]  # Skip 'I'
    header = "   " + " ".join(cols)

    lines = [header]

    for r in range(board.size):
        row_str = f"{board.size - r:2d} "
        for c in range(board.size):
            stone = board.board[r, c]
            if last_move and last_move == (r, c):
                # Highlight last move
                if stone == 1:
                    row_str += "\033[1;32m@\033[0m "  # Green for last black
                else:
                    row_str += "\033[1;31m#\033[0m "  # Red for last white
            elif stone == 1:
                row_str += "X "  # Black
            elif stone == -1:
                row_str += "O "  # White
            else:
                # Show star points on larger boards
                is_star = is_star_point(r, c, board.size)
                row_str += "+ " if is_star else ". "
        row_str += f"{board.size - r:2d}"
        lines.append(row_str)

    lines.append(header)
    return "\n".join(lines)


def is_star_point(row: int, col: int, size: int) -> bool:
    """Check if position is a star point."""
    if size == 9:
        stars = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
    elif size == 13:
        stars = [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
    elif size == 19:
        stars = [(3, 3), (3, 9), (3, 15),
                 (9, 3), (9, 9), (9, 15),
                 (15, 3), (15, 9), (15, 15)]
    else:
        return False
    return (row, col) in stars


def format_move(move, board_size):
    """Format move as Go coordinates (e.g., D4)."""
    if move == (-1, -1):
        return "PASS"
    r, c = move
    cols = "ABCDEFGHJKLMNOPQRST"[:board_size]
    return f"{cols[c]}{board_size - r}"


def analyze_position(board: Board, mcts, top_n: int = 5):
    """Analyze position and show top move candidates."""
    policy = mcts.search(board, verbose=False)

    # Get top moves
    indices = np.argsort(policy)[-top_n:][::-1]
    moves = []
    for idx in indices:
        if idx == board.size ** 2:
            move = (-1, -1)
        else:
            move = (idx // board.size, idx % board.size)
        prob = policy[idx]
        moves.append((move, prob))

    return moves


def print_analysis(board: Board, moves, board_size: int):
    """Print move analysis."""
    print("\nTop moves:")
    for i, (move, prob) in enumerate(moves):
        move_str = format_move(move, board_size)
        bar = "#" * int(prob * 40)
        print(f"  {i+1}. {move_str:4s} {prob:5.1%} {bar}")


def watch_game(model, config: Config, use_hybrid: bool = False,
               tactical_weight: float = 0.3, delay: float = 0.5,
               analyze: bool = False):
    """Watch a single game played by the AI."""
    board = Board(config.board_size)

    if use_hybrid:
        from hybrid_mcts import HybridMCTS
        from tactics import TacticalAnalyzer
        tactics = TacticalAnalyzer()
        mcts = HybridMCTS(model, config, tactics, batch_size=8,
                         tactical_weight=tactical_weight)
        print("Using HybridMCTS with tactical enhancements")
    else:
        mcts = MCTS(model, config, batch_size=8)
        print("Using standard MCTS")

    print(f"\nStarting game on {config.board_size}x{config.board_size} board...")
    print(f"MCTS simulations: {config.mcts_simulations}")
    print("-" * 50)

    move_history = []
    last_move = None

    while not board.is_game_over() and len(move_history) < board.size ** 2 * 2:
        # Clear screen and show board
        print("\033[2J\033[H")  # ANSI clear screen
        print(f"Move {len(move_history) + 1}")
        print(f"{'Black (X)' if board.current_player == 1 else 'White (O)'} to play")
        print()
        print(board_to_str(board, last_move, len(move_history)))

        # Show stats
        stats = get_game_stats(board)
        print(f"\nBlack: {stats['black_stones']} stones, {stats['black_groups']} groups")
        print(f"White: {stats['white_stones']} stones, {stats['white_groups']} groups")
        print(f"Score estimate: {stats['score']:+.1f} (+ = Black ahead)")

        # Get move
        if analyze:
            top_moves = analyze_position(board, mcts)
            print_analysis(board, top_moves, config.board_size)

        policy = mcts.search(board, verbose=False)

        # Use temperature 0 for deterministic play
        action_idx = np.argmax(policy)
        if action_idx == board.size ** 2:
            action = (-1, -1)
        else:
            action = (action_idx // board.size, action_idx % board.size)

        move_str = format_move(action, config.board_size)
        print(f"\nPlayed: {move_str}")

        # Play the move
        if action == (-1, -1):
            board.pass_move()
        else:
            captures = board.play(action[0], action[1])
            if captures > 0:
                print(f"Captured {captures} stone(s)!")

        last_move = action
        move_history.append(action)

        time.sleep(delay)

    # Game over
    print("\033[2J\033[H")
    print("GAME OVER")
    print()
    print(board_to_str(board, last_move, len(move_history)))

    # Final score
    score = board.score()
    print(f"\nFinal score: {abs(score):.1f} point(s) for {'Black' if score > 0 else 'White'}")
    print(f"Total moves: {len(move_history)}")

    # Print move list
    print("\nMove history:")
    for i, move in enumerate(move_history):
        player = "B" if i % 2 == 0 else "W"
        move_str = format_move(move, config.board_size)
        print(f"{i+1:3d}. {player}: {move_str}", end="  ")
        if (i + 1) % 5 == 0:
            print()
    print()

    return score, move_history


def analyze_game_quality(score: float, move_history: list, board_size: int):
    """Analyze game quality and look for issues."""
    print("\n" + "=" * 50)
    print("GAME QUALITY ANALYSIS")
    print("=" * 50)

    issues = []

    # Check for excessive passing
    pass_count = sum(1 for m in move_history if m == (-1, -1))
    if pass_count > 4:
        issues.append(f"Excessive passing: {pass_count} passes")

    # Check game length
    expected_moves = board_size * board_size * 0.6  # ~60% of board typically filled
    if len(move_history) < expected_moves * 0.3:
        issues.append(f"Very short game: only {len(move_history)} moves")

    # Check for repeated positions (could indicate issues)
    # This is a simple check; real implementation would use Zobrist hashing
    if len(move_history) > board_size * board_size:
        issues.append("Game too long - possible loop or inefficient play")

    # Check score magnitude
    if abs(score) > board_size * board_size * 0.5:
        issues.append(f"Lopsided score: {score:.1f} - one player dominated")

    if issues:
        print("Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No major issues detected. Game appears reasonable.")

    # Positive signs
    print("\nPositive signs to look for:")
    print("  - Groups with 2+ eyes survive")
    print("  - No pointless self-atari")
    print("  - Capture races resolved correctly")
    print("  - Corners/edges played before center (opening)")
    print("  - Endgame: small yose moves")


def main():
    parser = argparse.ArgumentParser(description='Watch AI games')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--games', type=int, default=1,
                        help='Number of games to watch')
    parser.add_argument('--hybrid', action='store_true',
                        help='Use HybridMCTS with tactical enhancements')
    parser.add_argument('--tactical-weight', type=float, default=0.3,
                        help='Weight for tactical adjustments (0-1)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between moves in seconds')
    parser.add_argument('--analyze', action='store_true',
                        help='Show move analysis')
    parser.add_argument('--simulations', type=int, default=200,
                        help='MCTS simulations per move')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    from config import DEFAULT
    config = DEFAULT
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, step = load_checkpoint(args.checkpoint, config)
    config = model.config
    config.mcts_simulations = args.simulations
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Model loaded (step {step})")
    print(f"Board size: {config.board_size}x{config.board_size}")
    print(f"Device: {config.device}")

    for game_idx in range(args.games):
        print(f"\n{'='*50}")
        print(f"GAME {game_idx + 1} of {args.games}")
        print(f"{'='*50}")

        score, history = watch_game(
            model, config,
            use_hybrid=args.hybrid,
            tactical_weight=args.tactical_weight,
            delay=args.delay,
            analyze=args.analyze
        )

        analyze_game_quality(score, history, config.board_size)

        if game_idx < args.games - 1:
            input("\nPress Enter to watch next game...")


if __name__ == '__main__':
    main()
