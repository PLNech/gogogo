#!/usr/bin/env python3
"""Tactical Test Suite for Go AI agents.

Tests captures, snapbacks, ladders - adapts to any board size.
Use to validate agent architecture variants.

Usage:
    python test_tactics.py --checkpoint checkpoints/supervised_best.pt
    python test_tactics.py --checkpoint checkpoints/supervised_best.pt --variant direct
    python test_tactics.py --checkpoint checkpoints/supervised_best.pt --variant hybrid
"""

import argparse
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
from board import Board
from config import DEFAULT
from model import load_checkpoint


@dataclass
class TacticalTest:
    """A single tactical test case."""
    name: str
    category: str  # capture, snapback, ladder, escape, connect
    board_setup: List[Tuple[int, int, int]]  # (row, col, color)
    player: int  # Who to play (1=black, -1=white)
    correct_moves: List[Tuple[int, int]]  # Acceptable correct moves
    wrong_moves: List[Tuple[int, int]]  # Definitely wrong moves
    description: str


def create_test_suite(board_size: int = 19) -> List[TacticalTest]:
    """Create tactical test cases centered on board."""
    tests = []
    c = board_size // 2  # Center

    # ===== CAPTURES =====

    # 1. Single stone capture
    tests.append(TacticalTest(
        name="Single stone capture",
        category="capture",
        board_setup=[
            (c, c, 1),       # Black center
            (c, c+1, -1),    # White right
            (c, c-1, -1),    # White left
            (c-1, c, -1),    # White above
        ],
        player=-1,
        correct_moves=[(c+1, c)],
        wrong_moves=[(0, 0)],
        description="White captures single black stone"
    ))

    # 2. Two stone capture
    tests.append(TacticalTest(
        name="Two stone capture",
        category="capture",
        board_setup=[
            (c, c, 1), (c, c+1, 1),       # Two black horizontal
            (c, c-1, -1), (c, c+2, -1),   # White sides
            (c-1, c, -1), (c-1, c+1, -1), # White above
            (c+1, c+1, -1),               # White below right
        ],
        player=-1,
        correct_moves=[(c+1, c)],
        wrong_moves=[(0, 0)],
        description="White captures two black stones"
    ))

    # 3. Three stone capture (high value!)
    tests.append(TacticalTest(
        name="Three stone capture",
        category="capture",
        board_setup=[
            (c, c, 1), (c, c+1, 1), (c, c+2, 1),  # Three black
            (c, c-1, -1), (c, c+3, -1),            # White ends
            (c-1, c, -1), (c-1, c+1, -1), (c-1, c+2, -1),  # White above
            (c+1, c+1, -1), (c+1, c+2, -1),        # White below (partial)
        ],
        player=-1,
        correct_moves=[(c+1, c)],
        wrong_moves=[(0, 0)],
        description="White captures three black stones"
    ))

    # ===== ESCAPE FROM ATARI =====

    # 4. Extend to escape
    tests.append(TacticalTest(
        name="Extend to escape",
        category="escape",
        board_setup=[
            (c, c, 1),       # Black
            (c, c+1, -1),    # White right
            (c-1, c, -1),    # White above
            (c+1, c, -1),    # White below - atari!
        ],
        player=1,
        correct_moves=[(c, c-1)],  # Extend left
        wrong_moves=[(0, 0)],
        description="Black extends to escape atari"
    ))

    # 5. Connect to escape
    tests.append(TacticalTest(
        name="Connect to escape",
        category="escape",
        board_setup=[
            (c, c, 1),       # Black in danger
            (c, c-2, 1),     # Friendly black nearby
            (c, c+1, -1),    # White right
            (c-1, c, -1),    # White above
            (c+1, c, -1),    # White below - atari!
        ],
        player=1,
        correct_moves=[(c, c-1)],  # Connect
        wrong_moves=[(0, 0)],
        description="Black connects to escape"
    ))

    # ===== LADDERS =====

    # 6. Start ladder
    tests.append(TacticalTest(
        name="Start ladder",
        category="ladder",
        board_setup=[
            (c, c, 1),       # Black
            (c, c+1, -1),    # White right
            (c+1, c, -1),    # White below
        ],
        player=-1,
        correct_moves=[(c-1, c), (c, c-1)],  # Atari moves
        wrong_moves=[(0, 0)],
        description="White starts ladder"
    ))

    # 7. Continue ladder
    tests.append(TacticalTest(
        name="Continue ladder",
        category="ladder",
        board_setup=[
            (c, c, 1), (c, c-1, 1),  # Black running
            (c, c+1, -1),             # White
            (c+1, c, -1),
            (c-1, c, -1),             # Atari!
        ],
        player=-1,
        correct_moves=[(c-1, c-1), (c, c-2)],  # Continue chase
        wrong_moves=[],
        description="White continues ladder"
    ))

    # ===== SNAPBACK =====

    # 8. Throw-in for snapback
    tests.append(TacticalTest(
        name="Snapback throw-in",
        category="snapback",
        board_setup=[
            # Black tiger's mouth shape
            (c-1, c, 1), (c-1, c+1, 1),
            (c+1, c, 1), (c+1, c+1, 1),
            (c, c-1, 1), (c, c+2, 1),
            # White around
            (c-2, c, -1), (c-2, c+1, -1),
            (c+2, c, -1), (c+2, c+1, -1),
            (c-1, c-1, -1), (c+1, c-1, -1),
            (c-1, c+2, -1), (c+1, c+2, -1),
        ],
        player=-1,
        correct_moves=[(c, c), (c, c+1)],  # Throw-in points
        wrong_moves=[(0, 0)],
        description="White throws in for snapback"
    ))

    # ===== CONNECT/CUT =====

    # 9. Must connect
    tests.append(TacticalTest(
        name="Must connect",
        category="connect",
        board_setup=[
            (c-1, c, 1), (c-1, c+1, 1),  # Black group 1
            (c+1, c, 1), (c+1, c+1, 1),  # Black group 2
            (c, c-1, -1),                 # White threatening
            (c, c+2, -1),
        ],
        player=1,
        correct_moves=[(c, c), (c, c+1)],
        wrong_moves=[(0, 0)],
        description="Black must connect groups"
    ))

    # 10. Cut opponent
    tests.append(TacticalTest(
        name="Cut opponent",
        category="cut",
        board_setup=[
            (c-1, c, -1), (c-1, c+1, -1),  # White group 1
            (c+1, c, -1), (c+1, c+1, -1),  # White group 2
            (c, c-1, 1),                    # Black nearby
            (c, c+2, 1),
        ],
        player=1,
        correct_moves=[(c, c), (c, c+1)],
        wrong_moves=[(0, 0)],
        description="Black cuts white groups"
    ))

    return tests


def board_from_setup(setup: List[Tuple[int, int, int]], size: int = 19) -> Board:
    """Create board from setup list."""
    board = Board(size)
    for r, c, color in setup:
        if 0 <= r < size and 0 <= c < size:
            board.board[r, c] = color
    return board


def format_move(move: Tuple[int, int], size: int = 19) -> str:
    """Format move as Go coordinates."""
    if move == (-1, -1):
        return "PASS"
    r, c = move
    cols = "ABCDEFGHJKLMNOPQRST"[:size]
    return f"{cols[c]}{size - r}"


class DirectPolicyAgent:
    """Uses raw neural network policy (no MCTS)."""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.use_tactical = config.input_planes == 27

    def get_move(self, board: Board) -> Tuple[int, int]:
        tensor = board.to_tensor(use_tactical_features=self.use_tactical)
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(np.expand_dims(tensor, 0))
            log_policy, _ = self.model(x)
            policy = torch.exp(log_policy).cpu().numpy()[0]

        # Mask illegal
        size = board.size
        legal = board.get_legal_moves()
        legal_mask = np.zeros(size * size + 1)
        for move in legal:
            if move == (-1, -1):
                legal_mask[size * size] = 1
            else:
                legal_mask[move[0] * size + move[1]] = 1

        policy = policy[:len(legal_mask)] * legal_mask
        if policy.sum() > 0:
            policy = policy / policy.sum()

        action_idx = np.argmax(policy)
        if action_idx == size * size:
            return (-1, -1)
        return (action_idx // size, action_idx % size)


class HybridAgent:
    """Direct policy + tactical boost for tactical positions.

    Uses ADDITIVE boosts for tactical moves to overcome bad NN priors.
    """

    def __init__(self, model, config, additive_weight: float = 0.05):
        self.model = model
        self.config = config
        self.use_tactical = config.input_planes == 27
        self.additive_weight = additive_weight  # Add this much per boost point
        from tactics import TacticalAnalyzer
        self.tactics = TacticalAnalyzer()

    def get_move(self, board: Board) -> Tuple[int, int]:
        tensor = board.to_tensor(use_tactical_features=self.use_tactical)
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(np.expand_dims(tensor, 0))
            log_policy, _ = self.model(x)
            policy = torch.exp(log_policy).cpu().numpy()[0]

        size = board.size

        # Apply tactical boosts (additive for significant boosts)
        if self.tactics.is_tactical_position(board):
            for move in board.get_legal_moves():
                if move == (-1, -1):
                    continue
                boost = self.tactics.get_tactical_boost(board, move)
                idx = move[0] * size + move[1]
                if idx < len(policy):
                    if boost > 1.0:
                        # ADDITIVE: add bonus proportional to boost
                        policy[idx] += self.additive_weight * (boost - 1.0)
                    elif boost < 1.0:
                        # Still multiplicative for penalties
                        policy[idx] *= boost

        # Mask illegal
        legal = board.get_legal_moves()
        legal_mask = np.zeros(size * size + 1)
        for move in legal:
            if move == (-1, -1):
                legal_mask[size * size] = 1
            else:
                legal_mask[move[0] * size + move[1]] = 1

        policy = policy[:len(legal_mask)] * legal_mask
        if policy.sum() > 0:
            policy = policy / policy.sum()

        action_idx = np.argmax(policy)
        if action_idx == size * size:
            return (-1, -1)
        return (action_idx // size, action_idx % size)


def run_test(test: TacticalTest, agent, board_size: int, verbose: bool = True) -> bool:
    """Run a single test case."""
    board = board_from_setup(test.board_setup, board_size)
    board.current_player = test.player

    move = agent.get_move(board)
    passed = move in test.correct_moves
    failed_hard = move in test.wrong_moves

    if verbose:
        status = "✓ PASS" if passed else ("✗ FAIL" if failed_hard else "~ MISS")
        move_str = format_move(move, board_size)
        correct_str = ", ".join(format_move(m, board_size) for m in test.correct_moves)
        print(f"  {status} {test.name}")
        print(f"       Played: {move_str}, Expected: {correct_str}")

    return passed


def run_suite(agent, tests: List[TacticalTest], board_size: int, verbose: bool = True) -> dict:
    """Run full test suite."""
    results = {"passed": 0, "failed": 0, "by_category": {}}

    for test in tests:
        if test.category not in results["by_category"]:
            results["by_category"][test.category] = {"passed": 0, "total": 0}

        results["by_category"][test.category]["total"] += 1
        passed = run_test(test, agent, board_size, verbose)

        if passed:
            results["passed"] += 1
            results["by_category"][test.category]["passed"] += 1
        else:
            results["failed"] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description='Tactical Test Suite')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/supervised_best.pt')
    parser.add_argument('--variant', type=str, default='all',
                        choices=['direct', 'hybrid', 'all'])
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    config = DEFAULT
    config.device = 'cpu'
    model, step = load_checkpoint(args.checkpoint, config)
    config = model.config
    board_size = config.board_size
    print(f"Model step: {step}, board size: {board_size}x{board_size}\n")

    # Create tests
    tests = create_test_suite(board_size)
    print(f"Running {len(tests)} tactical tests\n")

    # Setup variants
    variants = []
    if args.variant in ['direct', 'all']:
        variants.append(("Direct Policy", DirectPolicyAgent(model, config)))
    if args.variant in ['hybrid', 'all']:
        variants.append(("Hybrid (Policy+Tactical)", HybridAgent(model, config, additive_weight=0.10)))

    all_results = {}

    for name, agent in variants:
        print("=" * 50)
        print(f"  {name}")
        print("=" * 50)

        results = run_suite(agent, tests, board_size)
        all_results[name] = results

        pct = 100 * results['passed'] / len(tests)
        print(f"\n  Total: {results['passed']}/{len(tests)} ({pct:.0f}%)")
        for cat, r in results["by_category"].items():
            print(f"    {cat}: {r['passed']}/{r['total']}")
        print()

    # Summary
    if len(variants) > 1:
        print("=" * 50)
        print("  SUMMARY")
        print("=" * 50)
        for name, res in all_results.items():
            pct = 100 * res['passed'] / len(tests)
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            print(f"  {name:30s} {bar} {res['passed']}/{len(tests)}")


if __name__ == "__main__":
    main()
