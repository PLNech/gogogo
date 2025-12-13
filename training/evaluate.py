#!/usr/bin/env python3
"""Network evaluation and introspection toolkit.

Evaluate trained models on:
- Joseki sequences (opening patterns)
- Capture scenarios (can it spot the kill?)
- Defense scenarios (can it save the group?)
- Life/death basics
- Policy visualization
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from board import Board
from model import load_checkpoint
from config import Config


@dataclass
class TestPosition:
    """A test position with expected answer(s)."""
    name: str
    description: str
    board_size: int
    black_stones: List[Tuple[int, int]]  # (row, col)
    white_stones: List[Tuple[int, int]]
    to_play: int  # 1=black, -1=white
    correct_moves: List[Tuple[int, int]]  # Acceptable answers
    category: str  # joseki, capture, defense, life_death


# =============================================================================
# TEST POSITIONS
# =============================================================================

# 9x9 CAPTURE TESTS (for Atari Go training evaluation)
CAPTURE_TESTS_9x9 = [
    TestPosition(
        name="9x9_capture_center",
        description="Capture single stone in center",
        board_size=9,
        black_stones=[(4, 3), (4, 5), (3, 4)],
        white_stones=[(4, 4)],  # White stone with 1 liberty at (5,4)
        to_play=1,
        correct_moves=[(5, 4)],
        category="capture"
    ),
    TestPosition(
        name="9x9_capture_edge",
        description="Capture stone on edge",
        board_size=9,
        black_stones=[(4, 0), (3, 1), (5, 1)],
        white_stones=[(4, 1)],  # Edge stone in atari
        to_play=1,
        correct_moves=[(4, 2)],
        category="capture"
    ),
    TestPosition(
        name="9x9_capture_corner",
        description="Capture stone in corner",
        board_size=9,
        black_stones=[(0, 1), (1, 0)],
        white_stones=[(0, 0)],  # Corner stone - already dead if we play
        to_play=1,
        correct_moves=[],  # Any move wins, corner is dead
        category="capture"
    ),
    TestPosition(
        name="9x9_capture_two",
        description="Capture two connected stones",
        board_size=9,
        black_stones=[(3, 3), (3, 5), (4, 2), (4, 6), (5, 3), (5, 5)],
        white_stones=[(4, 3), (4, 4), (4, 5)],  # 3 stones, liberties at (3,4) and (5,4)
        to_play=1,
        correct_moves=[(3, 4), (5, 4)],  # Either captures
        category="capture"
    ),
    TestPosition(
        name="9x9_escape_atari",
        description="Extend to escape atari",
        board_size=9,
        black_stones=[(4, 4)],  # Black in atari
        white_stones=[(4, 3), (4, 5), (3, 4)],
        to_play=1,
        correct_moves=[(5, 4)],  # Extend to escape
        category="defense"
    ),
    TestPosition(
        name="9x9_ladder_start",
        description="Start a ladder",
        board_size=9,
        black_stones=[(4, 4), (5, 3)],
        white_stones=[(4, 3)],  # Can be laddered
        to_play=1,
        correct_moves=[(3, 3), (4, 2)],  # Either continues ladder
        category="capture"
    ),
]

# 19x19 CAPTURE TESTS
CAPTURE_TESTS = [
    TestPosition(
        name="capture_1",
        description="Capture single stone in atari",
        board_size=19,
        black_stones=[(9, 8), (9, 10), (8, 9)],
        white_stones=[(9, 9)],  # White stone with 1 liberty at (10,9)
        to_play=1,
        correct_moves=[(10, 9)],
        category="capture"
    ),
    TestPosition(
        name="capture_corner",
        description="Capture stone in corner",
        board_size=19,
        black_stones=[(0, 1), (1, 0)],
        white_stones=[(0, 0)],  # Corner stone in atari
        to_play=1,
        correct_moves=[],  # Already captured! Just checking recognition
        category="capture"
    ),
    TestPosition(
        name="capture_2_stones",
        description="Capture two connected stones",
        board_size=19,
        black_stones=[(8, 8), (8, 10), (9, 7), (9, 11), (10, 8), (10, 10)],
        white_stones=[(9, 8), (9, 9), (9, 10)],  # 3 white stones, 1 liberty
        to_play=1,
        correct_moves=[(8, 9), (10, 9)],  # Either captures
        category="capture"
    ),
    TestPosition(
        name="ladder_start",
        description="Start a ladder capture",
        board_size=19,
        black_stones=[(9, 9), (10, 8)],
        white_stones=[(9, 8)],  # White can be laddered
        to_play=1,
        correct_moves=[(8, 8), (9, 7)],  # Either continues ladder
        category="capture"
    ),
]

DEFENSE_TESTS = [
    TestPosition(
        name="escape_atari",
        description="Extend to escape atari",
        board_size=19,
        black_stones=[(9, 9)],  # Black in atari
        white_stones=[(9, 8), (9, 10), (8, 9)],
        to_play=1,
        correct_moves=[(10, 9)],  # Extend to safety
        category="defense"
    ),
    TestPosition(
        name="connect_to_live",
        description="Connect two groups",
        board_size=19,
        black_stones=[(9, 8), (9, 10)],  # Two black stones
        white_stones=[(8, 8), (8, 9), (8, 10), (10, 8), (10, 10)],  # Surrounding
        to_play=1,
        correct_moves=[(9, 9)],  # Connect!
        category="defense"
    ),
]

LIFE_DEATH_TESTS = [
    TestPosition(
        name="make_eye_space",
        description="Expand eye space to live",
        board_size=19,
        black_stones=[(2, 2), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3), (4, 4)],
        white_stones=[(1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1), (5, 2), (5, 3), (5, 4), (2, 5), (3, 5), (4, 5)],
        to_play=1,
        correct_moves=[(3, 3)],  # Make eye shape
        category="life_death"
    ),
]

JOSEKI_TESTS = [
    TestPosition(
        name="star_point_approach",
        description="Respond to star point approach",
        board_size=19,
        black_stones=[(3, 3)],  # 4-4 point (0-indexed: 3,3)
        white_stones=[(2, 5)],  # Approach at 3-6
        to_play=1,
        correct_moves=[(5, 3), (3, 5), (5, 2), (2, 3)],  # Common responses
        category="joseki"
    ),
    TestPosition(
        name="33_invasion",
        description="Respond to 3-3 invasion",
        board_size=19,
        black_stones=[(3, 3)],  # 4-4 point
        white_stones=[(2, 2)],  # 3-3 invasion
        to_play=1,
        correct_moves=[(2, 3), (3, 2)],  # Block one side
        category="joseki"
    ),
    TestPosition(
        name="komoku_approach",
        description="Approach the 3-4 point",
        board_size=19,
        black_stones=[(3, 2)],  # 4-3 point (komoku)
        white_stones=[],
        to_play=-1,  # White to approach
        correct_moves=[(2, 4), (4, 4), (5, 3)],  # Standard approaches
        category="joseki"
    ),
]

ALL_TESTS = CAPTURE_TESTS + CAPTURE_TESTS_9x9 + DEFENSE_TESTS + LIFE_DEATH_TESTS + JOSEKI_TESTS


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def setup_board(test: TestPosition) -> Board:
    """Create board from test position."""
    board = Board(test.board_size)

    # Place stones (alternate to maintain valid game state)
    # This is a simplification - we directly set the board
    for r, c in test.black_stones:
        board.board[r, c] = 1
    for r, c in test.white_stones:
        board.board[r, c] = -1

    board.current_player = test.to_play
    return board


def get_policy(model, board: Board, use_tactical: bool = True) -> np.ndarray:
    """Get policy distribution from model."""
    model.eval()
    with torch.no_grad():
        tensor = board.to_tensor(use_tactical_features=use_tactical)
        x = torch.FloatTensor(tensor).unsqueeze(0).to(next(model.parameters()).device)
        log_policy, value = model(x)
        policy = torch.exp(log_policy).cpu().numpy()[0]
        value = value.cpu().numpy()[0, 0]
    return policy, value


def policy_to_moves(policy: np.ndarray, board_size: int, top_k: int = 5) -> List[Tuple[Tuple[int, int], float]]:
    """Convert policy to list of (move, probability) sorted by probability."""
    moves = []
    for idx in range(len(policy)):
        if idx == board_size ** 2:
            move = (-1, -1)  # Pass
        else:
            move = (idx // board_size, idx % board_size)
        moves.append((move, policy[idx]))

    moves.sort(key=lambda x: x[1], reverse=True)
    return moves[:top_k]


def visualize_policy(policy: np.ndarray, board: Board, top_k: int = 10) -> str:
    """Create ASCII visualization of policy on board."""
    size = board.size
    lines = []

    # Header
    col_labels = '   ' + ' '.join(f'{c:2d}' for c in range(size))
    lines.append(col_labels)

    # Normalize policy for display (excluding pass)
    board_policy = policy[:-1].reshape(size, size)
    max_p = board_policy.max()

    for r in range(size):
        row_str = f'{r:2d} '
        for c in range(size):
            if board.board[r, c] == 1:
                row_str += ' X '
            elif board.board[r, c] == -1:
                row_str += ' O '
            else:
                # Show policy intensity
                p = board_policy[r, c]
                if p > max_p * 0.5:
                    row_str += ' # '  # High probability
                elif p > max_p * 0.2:
                    row_str += ' + '  # Medium
                elif p > max_p * 0.05:
                    row_str += ' . '  # Low
                else:
                    row_str += ' - '  # Very low
        lines.append(row_str)

    # Top moves
    top_moves = policy_to_moves(policy, size, top_k)
    lines.append('')
    lines.append(f'Top {top_k} moves:')
    for i, (move, prob) in enumerate(top_moves):
        if move == (-1, -1):
            move_str = 'pass'
        else:
            move_str = f'({move[0]}, {move[1]})'
        lines.append(f'  {i+1}. {move_str}: {prob:.1%}')

    # Pass probability
    pass_prob = policy[-1]
    lines.append(f'  Pass: {pass_prob:.1%}')

    return '\n'.join(lines)


def evaluate_position(model, test: TestPosition, use_tactical: bool = True, verbose: bool = True) -> Dict:
    """Evaluate model on a single test position."""
    board = setup_board(test)
    policy, value = get_policy(model, board, use_tactical)

    # Check if correct move is in top-k
    top_moves = policy_to_moves(policy, board.size, top_k=10)
    top_move_coords = [m[0] for m in top_moves]

    # Find rank of correct move
    correct_rank = None
    correct_prob = 0.0
    for correct in test.correct_moves:
        for rank, (move, prob) in enumerate(top_moves):
            if move == correct:
                if correct_rank is None or rank < correct_rank:
                    correct_rank = rank
                    correct_prob = prob
                break

    # Is top-1 correct?
    top1_correct = top_moves[0][0] in test.correct_moves if test.correct_moves else False

    # Is correct in top-3?
    top3_moves = [m[0] for m in top_moves[:3]]
    top3_correct = any(c in top3_moves for c in test.correct_moves) if test.correct_moves else False

    result = {
        'name': test.name,
        'category': test.category,
        'top1_correct': top1_correct,
        'top3_correct': top3_correct,
        'correct_rank': correct_rank,
        'correct_prob': correct_prob,
        'top1_move': top_moves[0][0],
        'top1_prob': top_moves[0][1],
        'value': value,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Test: {test.name}")
        print(f"Description: {test.description}")
        print(f"Category: {test.category}")
        print(f"To play: {'Black' if test.to_play == 1 else 'White'}")
        print(f"Correct moves: {test.correct_moves}")
        print()
        print(visualize_policy(policy, board))
        print()
        print(f"Value estimate: {value:+.3f} ({'Black' if value > 0 else 'White'} favored)")
        print(f"Top-1 correct: {'YES' if top1_correct else 'NO'}")
        print(f"Top-3 correct: {'YES' if top3_correct else 'NO'}")
        if correct_rank is not None:
            print(f"Correct move rank: {correct_rank + 1} ({correct_prob:.1%})")
        print('='*60)

    return result


def run_evaluation(model, tests: List[TestPosition], use_tactical: bool = True, verbose: bool = True) -> Dict:
    """Run evaluation on all test positions."""
    results = []

    for test in tests:
        # Skip if board size doesn't match model
        if test.board_size != model.config.board_size:
            if verbose:
                print(f"Skipping {test.name}: board size {test.board_size} != model {model.config.board_size}")
            continue

        result = evaluate_position(model, test, use_tactical, verbose)
        results.append(result)

    # Aggregate stats
    if not results:
        return {'error': 'No compatible tests found'}

    stats = {
        'total': len(results),
        'top1_accuracy': sum(r['top1_correct'] for r in results) / len(results),
        'top3_accuracy': sum(r['top3_correct'] for r in results) / len(results),
        'by_category': {},
        'results': results,
    }

    # Stats by category
    categories = set(r['category'] for r in results)
    for cat in categories:
        cat_results = [r for r in results if r['category'] == cat]
        stats['by_category'][cat] = {
            'total': len(cat_results),
            'top1_accuracy': sum(r['top1_correct'] for r in cat_results) / len(cat_results),
            'top3_accuracy': sum(r['top3_correct'] for r in cat_results) / len(cat_results),
        }

    return stats


def print_summary(stats: Dict):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    if 'error' in stats:
        print(f"Error: {stats['error']}")
        print("="*60)
        return

    print(f"Total tests: {stats['total']}")
    print(f"Top-1 accuracy: {stats['top1_accuracy']:.1%}")
    print(f"Top-3 accuracy: {stats['top3_accuracy']:.1%}")
    print()
    print("By category:")
    for cat, cat_stats in stats['by_category'].items():
        print(f"  {cat}: {cat_stats['top1_accuracy']:.1%} top-1, {cat_stats['top3_accuracy']:.1%} top-3 ({cat_stats['total']} tests)")
    print("="*60)


def interactive_mode(model, use_tactical: bool = True):
    """Interactive mode for exploring positions."""
    print("\nInteractive mode. Enter positions to analyze.")
    print("Commands:")
    print("  'q' - quit")
    print("  'new <size>' - new empty board")
    print("  'play <row> <col>' - play a move")
    print("  'show' - show current policy")
    print("  'value' - show value estimate")
    print()

    board = Board(model.config.board_size)

    while True:
        cmd = input("> ").strip().lower().split()
        if not cmd:
            continue

        if cmd[0] == 'q':
            break
        elif cmd[0] == 'new':
            size = int(cmd[1]) if len(cmd) > 1 else model.config.board_size
            board = Board(size)
            print(f"New {size}x{size} board")
        elif cmd[0] == 'play':
            if len(cmd) >= 3:
                r, c = int(cmd[1]), int(cmd[2])
                if board.is_valid_move(r, c):
                    board.play(r, c)
                    print(board)
                else:
                    print("Invalid move")
            else:
                print("Usage: play <row> <col>")
        elif cmd[0] == 'show':
            policy, value = get_policy(model, board, use_tactical)
            print(visualize_policy(policy, board))
            print(f"Value: {value:+.3f}")
        elif cmd[0] == 'value':
            _, value = get_policy(model, board, use_tactical)
            print(f"Value: {value:+.3f} ({'Black' if value > 0 else 'White'} favored)")
        elif cmd[0] == 'board':
            print(board)
        else:
            print(f"Unknown command: {cmd[0]}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Go network')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--category', type=str, default=None,
                        choices=['capture', 'defense', 'life_death', 'joseki'],
                        help='Run only specific category')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive exploration mode')
    parser.add_argument('--no-tactical', action='store_true',
                        help='Disable tactical features')
    parser.add_argument('--quiet', action='store_true',
                        help='Only show summary')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    config = Config()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, step = load_checkpoint(args.checkpoint, config)
    print(f"Loaded model (step {step}, board_size={model.config.board_size})")

    use_tactical = not args.no_tactical and model.config.input_planes == 27
    if use_tactical:
        print("Using tactical features (27 planes)")
    else:
        print("Using basic features (17 planes)")

    if args.interactive:
        interactive_mode(model, use_tactical)
    else:
        # Filter tests by category if specified
        tests = ALL_TESTS
        if args.category:
            tests = [t for t in tests if t.category == args.category]

        # Run evaluation
        stats = run_evaluation(model, tests, use_tactical, verbose=not args.quiet)
        print_summary(stats)


if __name__ == '__main__':
    main()
