#!/usr/bin/env python3
"""Validate benchmark positions using KataGo.

This script runs KataGo on benchmark positions to verify they're correctly labeled.
KataGo should agree with our expected moves for simple tactical positions.

Requirements:
    - KataGo installed and in PATH (or set KATAGO_PATH)
    - KataGo model file (set KATAGO_MODEL or use default)

Usage:
    poetry run python validate_benchmarks.py
    KATAGO_PATH=/path/to/katago poetry run python validate_benchmarks.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Try to find KataGo
KATAGO_PATH = os.environ.get('KATAGO_PATH', 'katago')
KATAGO_MODEL = os.environ.get('KATAGO_MODEL', '')
KATAGO_CONFIG = os.environ.get('KATAGO_CONFIG', '')


def board_to_gtp_moves(black_stones: List, white_stones: List, board_size: int) -> List[str]:
    """Convert stone positions to GTP move format."""
    moves = []
    # Interleave black and white stones
    max_len = max(len(black_stones), len(white_stones))
    for i in range(max_len):
        if i < len(black_stones):
            r, c = black_stones[i]
            moves.append(f"play black {_coord_to_gtp(r, c, board_size)}")
        if i < len(white_stones):
            r, c = white_stones[i]
            moves.append(f"play white {_coord_to_gtp(r, c, board_size)}")
    return moves


def _coord_to_gtp(row: int, col: int, board_size: int) -> str:
    """Convert (row, col) to GTP coordinate (e.g., 'D4')."""
    # GTP uses letters A-T (skipping I) for columns, numbers 1-19 for rows (from bottom)
    letters = 'ABCDEFGHJKLMNOPQRST'  # Skip I
    letter = letters[col]
    number = board_size - row
    return f"{letter}{number}"


def _gtp_to_coord(gtp: str, board_size: int) -> tuple:
    """Convert GTP coordinate to (row, col)."""
    letters = 'ABCDEFGHJKLMNOPQRST'
    col = letters.index(gtp[0].upper())
    row = board_size - int(gtp[1:])
    return (row, col)


def run_katago_analysis(position: Dict, visits: int = 100) -> Optional[Dict]:
    """Run KataGo analysis on a position.

    Returns:
        Dictionary with KataGo's top moves and their winrates, or None if failed.
    """
    if not os.path.exists(KATAGO_PATH) and KATAGO_PATH == 'katago':
        # Try to find katago
        result = subprocess.run(['which', 'katago'], capture_output=True, text=True)
        if result.returncode != 0:
            print("KataGo not found. Install it or set KATAGO_PATH environment variable.")
            return None

    board_size = position['board_size']

    # Build GTP commands
    commands = [
        f"boardsize {board_size}",
        "clear_board",
        "komi 6.5",
    ]

    # Add stones to board
    for r, c in position.get('black_stones', []):
        commands.append(f"play black {_coord_to_gtp(r, c, board_size)}")
    for r, c in position.get('white_stones', []):
        commands.append(f"play white {_coord_to_gtp(r, c, board_size)}")

    # Request analysis
    color = position['to_play']
    commands.append(f"genmove {color}")
    commands.append("quit")

    # Run KataGo
    gtp_input = "\n".join(commands) + "\n"

    try:
        args = [KATAGO_PATH, "gtp"]
        if KATAGO_MODEL:
            args.extend(["-model", KATAGO_MODEL])
        if KATAGO_CONFIG:
            args.extend(["-config", KATAGO_CONFIG])

        result = subprocess.run(
            args,
            input=gtp_input,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"KataGo error: {result.stderr}")
            return None

        # Parse output to find the move
        for line in result.stdout.strip().split('\n'):
            if line.startswith('=') and len(line) > 2:
                # Extract move from response like "= D4"
                parts = line.split()
                if len(parts) >= 2:
                    move_gtp = parts[1]
                    if move_gtp.lower() == 'pass':
                        return {'top_move': 'pass', 'moves': []}
                    try:
                        top_move = _gtp_to_coord(move_gtp, board_size)
                        return {'top_move': top_move, 'moves': [(top_move, 1.0)]}
                    except (ValueError, IndexError):
                        pass

        return None

    except FileNotFoundError:
        print(f"KataGo not found at {KATAGO_PATH}")
        return None
    except subprocess.TimeoutExpired:
        print("KataGo timed out")
        return None
    except Exception as e:
        print(f"Error running KataGo: {e}")
        return None


def validate_benchmarks(benchmark_dir: str = 'benchmarks', board_size: int = 9):
    """Validate benchmark positions against KataGo."""

    # Load positions
    positions = []
    benchmark_path = Path(benchmark_dir)

    for json_file in benchmark_path.rglob('*.json'):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if item.get('board_size', board_size) == board_size:
                        positions.append((json_file.name, item))
            elif isinstance(data, dict):
                if data.get('board_size', board_size) == board_size:
                    positions.append((json_file.name, data))
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    if not positions:
        print(f"No positions found for {board_size}x{board_size}")
        return

    print(f"Validating {len(positions)} positions against KataGo...")
    print()

    agreed = 0
    disagreed = 0
    failed = 0

    for filename, pos in positions:
        name = pos.get('name', 'unnamed')
        expected = pos.get('expected_moves', [])
        expected_set = set(tuple(m) for m in expected)

        result = run_katago_analysis(pos)

        if result is None:
            failed += 1
            print(f"[FAIL] {name}: KataGo failed")
            continue

        katago_move = result['top_move']

        if katago_move == 'pass':
            if not expected_set:
                agreed += 1
                print(f"[OK] {name}: KataGo passes (expected)")
            else:
                disagreed += 1
                print(f"[!!] {name}: KataGo passes but expected {expected}")
        elif katago_move in expected_set:
            agreed += 1
            print(f"[OK] {name}: KataGo plays {katago_move}")
        else:
            # Check if KataGo's move is in avoid list
            avoid = pos.get('avoid_moves', [])
            avoid_set = set(tuple(m) for m in avoid)

            if katago_move in avoid_set:
                disagreed += 1
                print(f"[!!] {name}: KataGo plays {katago_move} (we say avoid!)")
            elif not expected_set:
                agreed += 1
                print(f"[OK] {name}: KataGo plays {katago_move} (no specific expected)")
            else:
                disagreed += 1
                print(f"[??] {name}: KataGo plays {katago_move}, expected {expected}")

    print()
    print("=" * 50)
    print(f"Results: {agreed} agreed, {disagreed} disagreed, {failed} failed")
    print(f"Agreement rate: {agreed / (agreed + disagreed) * 100:.1f}%" if (agreed + disagreed) > 0 else "N/A")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate benchmarks with KataGo")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--benchmark-dir", default="benchmarks")
    args = parser.parse_args()

    validate_benchmarks(args.benchmark_dir, args.board_size)


if __name__ == "__main__":
    main()
