#!/usr/bin/env python3
"""Benchmark suite for evaluating Go model quality.

Runs model inference on categorized test positions and reports accuracy metrics.

Usage:
    poetry run python benchmark.py --checkpoint checkpoints/go_9x9_final.pt
    poetry run python benchmark.py --checkpoint checkpoints/supervised_best.pt --board-size 19
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from board import Board
from model import GoNet, load_checkpoint
from config import Config


class BenchmarkPosition:
    """A single benchmark position."""

    def __init__(self, data: dict):
        self.name = data['name']
        self.board_size = data['board_size']
        self.black_stones = data.get('black_stones', [])
        self.white_stones = data.get('white_stones', [])
        self.to_play = data['to_play']
        self.expected_moves = data.get('expected_moves', [])
        self.avoid_moves = data.get('avoid_moves', [])
        self.category = data['category']
        self.difficulty = data.get('difficulty', 'medium')
        self.description = data.get('description', '')

    def create_board(self) -> Board:
        """Create a Board object from this position."""
        board = Board(self.board_size)

        # Place stones
        for r, c in self.black_stones:
            board.board[r, c] = 1  # Black = 1

        for r, c in self.white_stones:
            board.board[r, c] = -1  # White = -1

        # Set current player
        board.current_player = 1 if self.to_play == 'black' else -1

        return board


class BenchmarkRunner:
    """Run benchmarks on a model."""

    def __init__(self, model: GoNet, config: Config, use_tactical: bool = False):
        self.model = model
        self.config = config
        self.use_tactical = use_tactical
        self.model.eval()

    def evaluate_position(self, pos: BenchmarkPosition) -> Dict:
        """Evaluate a single position.

        Returns:
            Dictionary with:
            - top1_correct: bool - Was #1 move an expected move?
            - top5_correct: bool - Was any expected move in top 5?
            - avoided_bad: bool - Did model avoid bad moves?
            - top1_move: tuple - The model's best move
            - move_ranks: list - Rank of each expected move
            - policy: np.array - Full policy output
        """
        board = pos.create_board()

        # Run inference
        with torch.no_grad():
            tensor = board.to_tensor(use_tactical_features=self.use_tactical)
            x = torch.FloatTensor(tensor).unsqueeze(0).to(self.config.device)
            policy_logits, value = self.model(x)
            policy = torch.exp(policy_logits).cpu().numpy()[0]

        size = board.size

        # Get sorted moves by probability
        move_probs = []
        for r in range(size):
            for c in range(size):
                idx = r * size + c
                move_probs.append(((r, c), policy[idx]))

        move_probs.sort(key=lambda x: x[1], reverse=True)

        # Top 5 moves
        top5_moves = [m[0] for m in move_probs[:5]]
        top1_move = top5_moves[0] if top5_moves else None

        # Check expected moves
        expected_set = set(tuple(m) for m in pos.expected_moves)
        avoid_set = set(tuple(m) for m in pos.avoid_moves)

        top1_correct = top1_move in expected_set if expected_set else True
        top5_correct = any(m in expected_set for m in top5_moves) if expected_set else True

        # Check if avoided bad moves
        avoided_bad = top1_move not in avoid_set if avoid_set else True

        # Get ranks of expected moves
        move_ranks = []
        for move in pos.expected_moves:
            move_tuple = tuple(move)
            for rank, (m, _) in enumerate(move_probs):
                if m == move_tuple:
                    move_ranks.append(rank + 1)
                    break

        return {
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
            'avoided_bad': avoided_bad,
            'top1_move': top1_move,
            'move_ranks': move_ranks,
            'value': float(value.cpu().numpy()[0, 0]),
            'policy': policy
        }


def load_benchmarks(benchmark_dir: str, board_size: int) -> List[BenchmarkPosition]:
    """Load all benchmark positions from directory."""
    positions = []
    benchmark_path = Path(benchmark_dir)

    for json_file in benchmark_path.rglob('*.json'):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Handle both single position and list formats
            if isinstance(data, list):
                for item in data:
                    if item.get('board_size', board_size) == board_size:
                        positions.append(BenchmarkPosition(item))
            elif isinstance(data, dict):
                if data.get('board_size', board_size) == board_size:
                    positions.append(BenchmarkPosition(data))

        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return positions


def run_benchmark(checkpoint_path: str, board_size: int = 9,
                  benchmark_dir: str = 'benchmarks', verbose: bool = True) -> Dict:
    """Run full benchmark suite.

    Returns dictionary with category-level and overall results.
    """
    # Load model
    config = Config(board_size=board_size)
    model, step = load_checkpoint(checkpoint_path, config)

    # Auto-detect tactical features
    use_tactical = False
    model_config = model.config
    if hasattr(model_config, 'tactical_features') and model_config.tactical_features:
        use_tactical = True
    elif hasattr(model_config, 'input_planes') and model_config.input_planes > 17:
        use_tactical = True

    if verbose:
        print(f"Loaded model: {checkpoint_path}")
        print(f"  Board size: {model_config.board_size}x{model_config.board_size}")
        print(f"  Tactical features: {use_tactical}")
        print()

    # Load positions
    positions = load_benchmarks(benchmark_dir, board_size)
    if not positions:
        print(f"No benchmark positions found for {board_size}x{board_size}")
        return {}

    if verbose:
        print(f"Loaded {len(positions)} benchmark positions")
        print()

    # Run benchmark
    runner = BenchmarkRunner(model, config, use_tactical)

    results_by_category: Dict[str, List[Dict]] = defaultdict(list)
    all_results = []

    for pos in positions:
        result = runner.evaluate_position(pos)
        result['name'] = pos.name
        result['category'] = pos.category
        result['difficulty'] = pos.difficulty

        results_by_category[pos.category].append(result)
        all_results.append(result)

    # Aggregate results
    summary = {
        'overall': aggregate_results(all_results),
        'by_category': {},
        'by_difficulty': {}
    }

    for category, results in results_by_category.items():
        summary['by_category'][category] = aggregate_results(results)

    # Group by difficulty
    for diff in ['easy', 'medium', 'hard']:
        diff_results = [r for r in all_results if r.get('difficulty') == diff]
        if diff_results:
            summary['by_difficulty'][diff] = aggregate_results(diff_results)

    if verbose:
        print_summary(summary)

    return summary


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate results into summary statistics."""
    if not results:
        return {'count': 0}

    return {
        'count': len(results),
        'top1_accuracy': sum(1 for r in results if r['top1_correct']) / len(results),
        'top5_accuracy': sum(1 for r in results if r['top5_correct']) / len(results),
        'avoided_bad': sum(1 for r in results if r['avoided_bad']) / len(results),
        'avg_rank': np.mean([r['move_ranks'][0] for r in results if r['move_ranks']]) if any(r['move_ranks'] for r in results) else float('inf'),
    }


def print_summary(summary: Dict):
    """Print formatted summary."""
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    overall = summary['overall']
    print(f"\nOverall ({overall['count']} positions):")
    print(f"  Top-1 Accuracy: {overall['top1_accuracy']:.1%}")
    print(f"  Top-5 Accuracy: {overall['top5_accuracy']:.1%}")
    print(f"  Avoided Bad:    {overall['avoided_bad']:.1%}")
    if overall.get('avg_rank', float('inf')) < float('inf'):
        print(f"  Avg Rank:       {overall['avg_rank']:.1f}")

    print("\nBy Category:")
    for cat, stats in sorted(summary['by_category'].items()):
        print(f"  {cat.capitalize():12} {stats['count']:3} pos | "
              f"Top-1: {stats['top1_accuracy']:5.1%} | "
              f"Top-5: {stats['top5_accuracy']:5.1%}")

    print("\nBy Difficulty:")
    for diff, stats in sorted(summary['by_difficulty'].items()):
        print(f"  {diff.capitalize():12} {stats['count']:3} pos | "
              f"Top-1: {stats['top1_accuracy']:5.1%} | "
              f"Top-5: {stats['top5_accuracy']:5.1%}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Go model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--board-size", type=int, default=9, help="Board size (default: 9)")
    parser.add_argument("--benchmark-dir", default="benchmarks", help="Benchmark directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    summary = run_benchmark(
        args.checkpoint,
        board_size=args.board_size,
        benchmark_dir=args.benchmark_dir,
        verbose=not args.quiet
    )

    # Return success if any positions were tested
    return 0 if summary.get('overall', {}).get('count', 0) > 0 else 1


if __name__ == "__main__":
    exit(main())
