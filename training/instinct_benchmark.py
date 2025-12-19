#!/usr/bin/env python3
"""Instinct Benchmark Generator for Go AI.

Generates 100+ test positions per instinct across:
- Board sizes (5x5, 7x7, 9x9, 13x13, 19x19)
- Colors (black to play, white to play)
- Board regions (corner, edge, center)
- Difficulty levels (easy, medium, hard)

The 8 Go Instincts:
1. Capture - Take opponent stones in atari
2. Escape - Save your stones in atari
3. Extend - Extend from weak groups to get liberties
4. Cut - Separate opponent stones
5. Block - Prevent opponent from connecting/extending
6. Connect - Join your groups together
7. Defend - Protect weak points (e.g., eyes)
8. Atari - Put opponent stones in atari

Usage:
    poetry run python instinct_benchmark.py --generate
    poetry run python instinct_benchmark.py --test checkpoints/model.pt
"""

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any, Optional
import numpy as np

from board import Board


# Supported board sizes
BOARD_SIZES = [5, 7, 9, 13, 19]

# Regions: corner, edge, center offsets from (0,0) for different board sizes
def get_regions(size: int) -> Dict[str, List[Tuple[int, int]]]:
    """Get anchor points for different board regions."""
    mid = size // 2
    margin = min(3, size // 4)

    corners = [(margin, margin), (margin, size - margin - 1),
               (size - margin - 1, margin), (size - margin - 1, size - margin - 1)]
    edges = [(margin, mid), (mid, margin), (mid, size - margin - 1), (size - margin - 1, mid)]
    center = [(mid, mid)]

    return {'corner': corners, 'edge': edges, 'center': center}


@dataclass
class GeneratedPosition:
    """A generated benchmark position."""
    name: str
    board_size: int
    black_stones: List[Tuple[int, int]]
    white_stones: List[Tuple[int, int]]
    to_play: str
    expected_moves: List[Tuple[int, int]]
    avoid_moves: List[Tuple[int, int]] = field(default_factory=list)
    category: str = ""
    difficulty: str = "medium"
    description: str = ""
    region: str = "center"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'board_size': self.board_size,
            'black_stones': [list(s) for s in self.black_stones],
            'white_stones': [list(s) for s in self.white_stones],
            'to_play': self.to_play,
            'expected_moves': [list(m) for m in self.expected_moves],
            'avoid_moves': [list(m) for m in self.avoid_moves],
            'category': self.category,
            'difficulty': self.difficulty,
            'description': self.description,
            'region': self.region,
        }

    def validate(self) -> bool:
        """Check that position is valid."""
        board = Board(self.board_size)
        for r, c in self.black_stones:
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                return False
            board.board[r, c] = 1
        for r, c in self.white_stones:
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                return False
            if board.board[r, c] != 0:
                return False  # Overlap
            board.board[r, c] = -1
        # Check expected moves are empty
        for r, c in self.expected_moves:
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                return False
            if board.board[r, c] != 0:
                return False
        return True


class InstinctGenerator:
    """Generate positions for a specific instinct."""

    def __init__(self, board_size: int):
        self.size = board_size
        self.positions: List[GeneratedPosition] = []

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def _neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Get orthogonal neighbors."""
        return [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                if self._in_bounds(r+dr, c+dc)]

    def _translate(self, stones: List[Tuple[int, int]], dr: int, dc: int) -> List[Tuple[int, int]]:
        """Translate a pattern."""
        return [(r+dr, c+dc) for r, c in stones]

    def _rotate_90(self, stones: List[Tuple[int, int]], center: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Rotate 90 degrees around center."""
        cr, cc = center
        return [(cc - (c - cc) + cr - cr, cr + (r - cr) + cc - cc) for r, c in stones]

    def _flip_color(self, pos: GeneratedPosition) -> GeneratedPosition:
        """Swap black and white, flip who plays."""
        return GeneratedPosition(
            name=pos.name + "_flipped",
            board_size=pos.board_size,
            black_stones=pos.white_stones,
            white_stones=pos.black_stones,
            to_play='white' if pos.to_play == 'black' else 'black',
            expected_moves=pos.expected_moves,
            avoid_moves=pos.avoid_moves,
            category=pos.category,
            difficulty=pos.difficulty,
            description=pos.description,
            region=pos.region,
        )


class CaptureGenerator(InstinctGenerator):
    """Generate capture positions - opponent in atari, take them."""

    def generate(self, count: int = 25) -> List[GeneratedPosition]:
        positions = []
        regions = get_regions(self.size)

        idx = 0
        for region_name, anchors in regions.items():
            for anchor in anchors:
                ar, ac = anchor

                # Pattern 1: Single stone in atari (3 liberties blocked)
                # . X .
                # X O X
                # . . .
                if self._in_bounds(ar, ac) and self._in_bounds(ar+1, ac):
                    black = [(ar, ac-1), (ar, ac+1), (ar-1, ac)] if self._in_bounds(ar, ac-1) and self._in_bounds(ar, ac+1) and self._in_bounds(ar-1, ac) else []
                    if len(black) == 3:
                        pos = GeneratedPosition(
                            name=f"capture_single_{idx}",
                            board_size=self.size,
                            black_stones=black,
                            white_stones=[(ar, ac)],
                            to_play='black',
                            expected_moves=[(ar+1, ac)],  # Capture point
                            category='capture',
                            difficulty='easy',
                            description='Capture single stone in atari',
                            region=region_name,
                        )
                        if pos.validate():
                            positions.append(pos)
                            positions.append(self._flip_color(pos))
                            idx += 1

                # Pattern 2: Two stones in atari (ladder-like)
                # X O .
                # X O .
                # . X .
                if self._in_bounds(ar+2, ac+1):
                    black = [(ar, ac-1), (ar+1, ac-1), (ar+2, ac)] if self._in_bounds(ar, ac-1) and self._in_bounds(ar+1, ac-1) else []
                    white = [(ar, ac), (ar+1, ac)]
                    if len(black) == 3:
                        pos = GeneratedPosition(
                            name=f"capture_two_{idx}",
                            board_size=self.size,
                            black_stones=black,
                            white_stones=white,
                            to_play='black',
                            expected_moves=[(ar, ac+1), (ar+1, ac+1)],  # Either captures
                            category='capture',
                            difficulty='easy',
                            description='Capture two stones in atari',
                            region=region_name,
                        )
                        if pos.validate():
                            positions.append(pos)
                            positions.append(self._flip_color(pos))
                            idx += 1

                # Pattern 3: Snapback - sacrifice then capture
                # . X X .
                # X O O X
                # X O . X
                # . X X .
                if self._in_bounds(ar+3, ac+3) and ar >= 1 and ac >= 1:
                    black = [(ar, ac), (ar, ac+1), (ar+1, ac-1), (ar+2, ac-1),
                             (ar+1, ac+2), (ar+2, ac+2), (ar+3, ac), (ar+3, ac+1)]
                    white = [(ar+1, ac), (ar+1, ac+1), (ar+2, ac)]
                    capture_point = (ar+2, ac+1)

                    # Filter to in-bounds
                    black = [(r,c) for r,c in black if self._in_bounds(r, c)]
                    white = [(r,c) for r,c in white if self._in_bounds(r, c)]

                    if len(black) >= 6 and len(white) == 3 and self._in_bounds(*capture_point):
                        pos = GeneratedPosition(
                            name=f"capture_snapback_{idx}",
                            board_size=self.size,
                            black_stones=black,
                            white_stones=white,
                            to_play='black',
                            expected_moves=[capture_point],
                            category='capture',
                            difficulty='hard',
                            description='Snapback capture',
                            region=region_name,
                        )
                        if pos.validate():
                            positions.append(pos)
                            positions.append(self._flip_color(pos))
                            idx += 1

                if len(positions) >= count * 2:
                    break
            if len(positions) >= count * 2:
                break

        return positions[:count * 2]


class EscapeGenerator(InstinctGenerator):
    """Generate escape positions - your stones in atari, save them."""

    def generate(self, count: int = 25) -> List[GeneratedPosition]:
        positions = []
        regions = get_regions(self.size)

        idx = 0
        for region_name, anchors in regions.items():
            for anchor in anchors:
                ar, ac = anchor

                # Pattern 1: Single stone in atari, extend to escape
                # . O .
                # O X O
                # . . .
                if self._in_bounds(ar+1, ac) and self._in_bounds(ar, ac-1) and self._in_bounds(ar, ac+1) and self._in_bounds(ar-1, ac):
                    white = [(ar, ac-1), (ar, ac+1), (ar-1, ac)]
                    escape_point = (ar+1, ac)

                    pos = GeneratedPosition(
                        name=f"escape_extend_{idx}",
                        board_size=self.size,
                        black_stones=[(ar, ac)],
                        white_stones=white,
                        to_play='black',
                        expected_moves=[escape_point],
                        category='escape',
                        difficulty='easy',
                        description='Escape by extending',
                        region=region_name,
                    )
                    if pos.validate():
                        positions.append(pos)
                        positions.append(self._flip_color(pos))
                        idx += 1

                # Pattern 2: Two stones in atari, one escape direction
                # O X X .
                # O . O .
                if self._in_bounds(ar+1, ac+2) and self._in_bounds(ar, ac-1):
                    white = [(ar, ac-1), (ar+1, ac-1), (ar+1, ac+1)]
                    black = [(ar, ac), (ar, ac+1)]
                    escape = (ar, ac+2)

                    if self._in_bounds(*escape):
                        pos = GeneratedPosition(
                            name=f"escape_group_{idx}",
                            board_size=self.size,
                            black_stones=black,
                            white_stones=white,
                            to_play='black',
                            expected_moves=[escape],
                            category='escape',
                            difficulty='medium',
                            description='Escape group in atari',
                            region=region_name,
                        )
                        if pos.validate():
                            positions.append(pos)
                            positions.append(self._flip_color(pos))
                            idx += 1

                # Pattern 3: Connect to escape
                # O X . X
                # . O O .
                if self._in_bounds(ar+1, ac+2) and self._in_bounds(ar, ac-1):
                    white = [(ar, ac-1), (ar+1, ac), (ar+1, ac+1)]
                    black = [(ar, ac), (ar, ac+2)]
                    connect_escape = (ar, ac+1)  # Connect to friend

                    if self._in_bounds(*connect_escape):
                        pos = GeneratedPosition(
                            name=f"escape_connect_{idx}",
                            board_size=self.size,
                            black_stones=black,
                            white_stones=white,
                            to_play='black',
                            expected_moves=[connect_escape],
                            category='escape',
                            difficulty='medium',
                            description='Escape by connecting',
                            region=region_name,
                        )
                        if pos.validate():
                            positions.append(pos)
                            positions.append(self._flip_color(pos))
                            idx += 1

                if len(positions) >= count * 2:
                    break
            if len(positions) >= count * 2:
                break

        return positions[:count * 2]


class AtariGenerator(InstinctGenerator):
    """Generate atari positions - put opponent in atari."""

    def generate(self, count: int = 25) -> List[GeneratedPosition]:
        positions = []
        regions = get_regions(self.size)

        idx = 0
        for region_name, anchors in regions.items():
            for anchor in anchors:
                ar, ac = anchor

                # Pattern 1: Put single stone in atari
                # . . .
                # X O .
                # . X .
                if self._in_bounds(ar+1, ac+1) and self._in_bounds(ar, ac-1):
                    black = [(ar, ac-1), (ar+1, ac)]
                    white = [(ar, ac)]
                    atari_point = (ar-1, ac)  # This creates atari

                    if self._in_bounds(*atari_point):
                        pos = GeneratedPosition(
                            name=f"atari_single_{idx}",
                            board_size=self.size,
                            black_stones=black,
                            white_stones=white,
                            to_play='black',
                            expected_moves=[atari_point],
                            category='atari',
                            difficulty='easy',
                            description='Put single stone in atari',
                            region=region_name,
                        )
                        if pos.validate():
                            positions.append(pos)
                            positions.append(self._flip_color(pos))
                            idx += 1

                # Pattern 2: Double atari
                # . O . O .
                # . . X . .
                if self._in_bounds(ar+1, ac+2) and self._in_bounds(ar, ac-2):
                    white = [(ar, ac-1), (ar, ac+1)]
                    black = []
                    double_atari = (ar, ac)

                    pos = GeneratedPosition(
                        name=f"atari_double_{idx}",
                        board_size=self.size,
                        black_stones=black,
                        white_stones=white,
                        to_play='black',
                        expected_moves=[double_atari],
                        category='atari',
                        difficulty='medium',
                        description='Double atari',
                        region=region_name,
                    )
                    if pos.validate():
                        positions.append(pos)
                        positions.append(self._flip_color(pos))
                        idx += 1

                if len(positions) >= count * 2:
                    break
            if len(positions) >= count * 2:
                break

        return positions[:count * 2]


class ConnectGenerator(InstinctGenerator):
    """Generate connect positions - join your groups."""

    def generate(self, count: int = 25) -> List[GeneratedPosition]:
        positions = []
        regions = get_regions(self.size)

        idx = 0
        for region_name, anchors in regions.items():
            for anchor in anchors:
                ar, ac = anchor

                # Pattern 1: Simple connect two stones
                # X . X
                if self._in_bounds(ar, ac+2):
                    black = [(ar, ac), (ar, ac+2)]
                    connect = (ar, ac+1)

                    pos = GeneratedPosition(
                        name=f"connect_simple_{idx}",
                        board_size=self.size,
                        black_stones=black,
                        white_stones=[],
                        to_play='black',
                        expected_moves=[connect],
                        category='connect',
                        difficulty='easy',
                        description='Simple connect',
                        region=region_name,
                    )
                    if pos.validate():
                        positions.append(pos)
                        positions.append(self._flip_color(pos))
                        idx += 1

                # Pattern 2: Threatened cut - connect under pressure
                # X . X
                # . O .
                if self._in_bounds(ar+1, ac+2) and self._in_bounds(ar+1, ac):
                    black = [(ar, ac), (ar, ac+2)]
                    white = [(ar+1, ac+1)]
                    connect = (ar, ac+1)  # Must connect or get cut

                    pos = GeneratedPosition(
                        name=f"connect_threatened_{idx}",
                        board_size=self.size,
                        black_stones=black,
                        white_stones=white,
                        to_play='black',
                        expected_moves=[connect],
                        category='connect',
                        difficulty='medium',
                        description='Connect under threat',
                        region=region_name,
                    )
                    if pos.validate():
                        positions.append(pos)
                        positions.append(self._flip_color(pos))
                        idx += 1

                # Pattern 3: Bamboo joint
                # X . X
                # . . .
                # X . X
                if self._in_bounds(ar+2, ac+2):
                    black = [(ar, ac), (ar, ac+2), (ar+2, ac), (ar+2, ac+2)]
                    connect = (ar+1, ac+1)  # Bamboo center

                    pos = GeneratedPosition(
                        name=f"connect_bamboo_{idx}",
                        board_size=self.size,
                        black_stones=black,
                        white_stones=[],
                        to_play='black',
                        expected_moves=[connect],
                        category='connect',
                        difficulty='easy',
                        description='Bamboo joint connect',
                        region=region_name,
                    )
                    if pos.validate():
                        positions.append(pos)
                        positions.append(self._flip_color(pos))
                        idx += 1

                if len(positions) >= count * 2:
                    break
            if len(positions) >= count * 2:
                break

        return positions[:count * 2]


class CutGenerator(InstinctGenerator):
    """Generate cut positions - separate opponent groups."""

    def generate(self, count: int = 25) -> List[GeneratedPosition]:
        positions = []
        regions = get_regions(self.size)

        idx = 0
        for region_name, anchors in regions.items():
            for anchor in anchors:
                ar, ac = anchor

                # Pattern 1: Simple cut
                # O . O
                if self._in_bounds(ar, ac+2):
                    white = [(ar, ac), (ar, ac+2)]
                    cut = (ar, ac+1)

                    pos = GeneratedPosition(
                        name=f"cut_simple_{idx}",
                        board_size=self.size,
                        black_stones=[],
                        white_stones=white,
                        to_play='black',
                        expected_moves=[cut],
                        category='cut',
                        difficulty='easy',
                        description='Simple cut',
                        region=region_name,
                    )
                    if pos.validate():
                        positions.append(pos)
                        positions.append(self._flip_color(pos))
                        idx += 1

                # Pattern 2: Diagonal cut
                # . O .
                # . . O
                if self._in_bounds(ar+1, ac+2) and self._in_bounds(ar, ac+1):
                    white = [(ar, ac+1), (ar+1, ac+2)]
                    cut = (ar+1, ac+1)  # Cuts diagonally connected stones

                    pos = GeneratedPosition(
                        name=f"cut_diagonal_{idx}",
                        board_size=self.size,
                        black_stones=[],
                        white_stones=white,
                        to_play='black',
                        expected_moves=[cut],
                        category='cut',
                        difficulty='medium',
                        description='Diagonal cut',
                        region=region_name,
                    )
                    if pos.validate():
                        positions.append(pos)
                        positions.append(self._flip_color(pos))
                        idx += 1

                # Pattern 3: Cut with support
                # X O . O
                # . . . .
                if self._in_bounds(ar, ac+3):
                    white = [(ar, ac+1), (ar, ac+3)]
                    black = [(ar, ac)]  # Support
                    cut = (ar, ac+2)

                    pos = GeneratedPosition(
                        name=f"cut_supported_{idx}",
                        board_size=self.size,
                        black_stones=black,
                        white_stones=white,
                        to_play='black',
                        expected_moves=[cut],
                        category='cut',
                        difficulty='medium',
                        description='Cut with support',
                        region=region_name,
                    )
                    if pos.validate():
                        positions.append(pos)
                        positions.append(self._flip_color(pos))
                        idx += 1

                if len(positions) >= count * 2:
                    break
            if len(positions) >= count * 2:
                break

        return positions[:count * 2]


class BlockGenerator(InstinctGenerator):
    """Generate block positions - prevent opponent extension."""

    def generate(self, count: int = 25) -> List[GeneratedPosition]:
        positions = []
        regions = get_regions(self.size)

        idx = 0
        for region_name, anchors in regions.items():
            for anchor in anchors:
                ar, ac = anchor

                # Pattern 1: Block extension toward corner/edge
                # X . . O
                # . . . .
                if self._in_bounds(ar, ac+3):
                    black = [(ar, ac)]
                    white = [(ar, ac+3)]
                    block = (ar, ac+2)  # Block white's expansion

                    pos = GeneratedPosition(
                        name=f"block_expansion_{idx}",
                        board_size=self.size,
                        black_stones=black,
                        white_stones=white,
                        to_play='black',
                        expected_moves=[block],
                        category='block',
                        difficulty='easy',
                        description='Block expansion',
                        region=region_name,
                    )
                    if pos.validate():
                        positions.append(pos)
                        positions.append(self._flip_color(pos))
                        idx += 1

                # Pattern 2: Block connection attempt
                # X . O O
                # . . . .
                if self._in_bounds(ar, ac+3):
                    black = [(ar, ac)]
                    white = [(ar, ac+2), (ar, ac+3)]
                    block = (ar, ac+1)  # Prevent white from connecting

                    pos = GeneratedPosition(
                        name=f"block_connect_{idx}",
                        board_size=self.size,
                        black_stones=black,
                        white_stones=white,
                        to_play='black',
                        expected_moves=[block],
                        category='block',
                        difficulty='medium',
                        description='Block connection',
                        region=region_name,
                    )
                    if pos.validate():
                        positions.append(pos)
                        positions.append(self._flip_color(pos))
                        idx += 1

                if len(positions) >= count * 2:
                    break
            if len(positions) >= count * 2:
                break

        return positions[:count * 2]


class ExtendGenerator(InstinctGenerator):
    """Generate extend positions - extend weak groups for liberties."""

    def generate(self, count: int = 25) -> List[GeneratedPosition]:
        positions = []
        regions = get_regions(self.size)

        idx = 0
        for region_name, anchors in regions.items():
            for anchor in anchors:
                ar, ac = anchor

                # Pattern 1: Extend single stone
                # . X .
                # . . .
                if self._in_bounds(ar+1, ac+1) and self._in_bounds(ar, ac-1):
                    black = [(ar, ac)]
                    extend_points = [(ar+1, ac), (ar, ac+1), (ar, ac-1), (ar-1, ac)]
                    extend_points = [p for p in extend_points if self._in_bounds(*p)]

                    if extend_points:
                        pos = GeneratedPosition(
                            name=f"extend_single_{idx}",
                            board_size=self.size,
                            black_stones=black,
                            white_stones=[],
                            to_play='black',
                            expected_moves=extend_points,  # Any extension is good
                            category='extend',
                            difficulty='easy',
                            description='Extend single stone',
                            region=region_name,
                        )
                        if pos.validate():
                            positions.append(pos)
                            positions.append(self._flip_color(pos))
                            idx += 1

                # Pattern 2: Extend under pressure
                # O X .
                # . . .
                if self._in_bounds(ar+1, ac+1) and self._in_bounds(ar, ac-1):
                    black = [(ar, ac)]
                    white = [(ar, ac-1)]
                    extend = (ar, ac+1)  # Extend away from pressure

                    pos = GeneratedPosition(
                        name=f"extend_pressure_{idx}",
                        board_size=self.size,
                        black_stones=black,
                        white_stones=white,
                        to_play='black',
                        expected_moves=[extend],
                        category='extend',
                        difficulty='medium',
                        description='Extend away from pressure',
                        region=region_name,
                    )
                    if pos.validate():
                        positions.append(pos)
                        positions.append(self._flip_color(pos))
                        idx += 1

                if len(positions) >= count * 2:
                    break
            if len(positions) >= count * 2:
                break

        return positions[:count * 2]


class DefendGenerator(InstinctGenerator):
    """Generate defend positions - protect weak points/eyes."""

    def generate(self, count: int = 25) -> List[GeneratedPosition]:
        positions = []
        regions = get_regions(self.size)

        idx = 0
        for region_name, anchors in regions.items():
            for anchor in anchors:
                ar, ac = anchor

                # Pattern 1: Defend cutting point
                # X . X
                # . O .
                # X . X
                if self._in_bounds(ar+2, ac+2):
                    black = [(ar, ac), (ar, ac+2), (ar+2, ac), (ar+2, ac+2)]
                    white = [(ar+1, ac+1)]  # Threatens to cut
                    defend = (ar+1, ac)  # Or (ar, ac+1), etc.
                    defend_options = [(ar+1, ac), (ar, ac+1), (ar+1, ac+2), (ar+2, ac+1)]
                    defend_options = [p for p in defend_options if self._in_bounds(*p)]

                    if defend_options:
                        pos = GeneratedPosition(
                            name=f"defend_cut_{idx}",
                            board_size=self.size,
                            black_stones=black,
                            white_stones=white,
                            to_play='black',
                            expected_moves=defend_options,
                            category='defend',
                            difficulty='medium',
                            description='Defend cutting point',
                            region=region_name,
                        )
                        if pos.validate():
                            positions.append(pos)
                            positions.append(self._flip_color(pos))
                            idx += 1

                # Pattern 2: Defend eye shape
                # X X X
                # X . X
                # X X X
                if self._in_bounds(ar+2, ac+2):
                    black = [(ar, ac), (ar, ac+1), (ar, ac+2),
                             (ar+1, ac), (ar+1, ac+2),
                             (ar+2, ac), (ar+2, ac+1), (ar+2, ac+2)]
                    defend = (ar+1, ac+1)  # Fill the eye point (defend from invasion)

                    pos = GeneratedPosition(
                        name=f"defend_eye_{idx}",
                        board_size=self.size,
                        black_stones=black,
                        white_stones=[],
                        to_play='black',
                        expected_moves=[defend],
                        avoid_moves=[],  # Could argue this is bad to fill eye
                        category='defend',
                        difficulty='hard',
                        description='Defend eye shape',
                        region=region_name,
                    )
                    # Note: this is actually a false eye defense situation
                    if pos.validate():
                        positions.append(pos)
                        idx += 1

                if len(positions) >= count * 2:
                    break
            if len(positions) >= count * 2:
                break

        return positions[:count * 2]


# All generators
INSTINCT_GENERATORS = {
    'capture': CaptureGenerator,
    'escape': EscapeGenerator,
    'atari': AtariGenerator,
    'connect': ConnectGenerator,
    'cut': CutGenerator,
    'block': BlockGenerator,
    'extend': ExtendGenerator,
    'defend': DefendGenerator,
}


def generate_all_benchmarks(output_dir: str = 'benchmarks/instincts',
                            positions_per_instinct: int = 25) -> Dict[str, List[GeneratedPosition]]:
    """Generate all instinct benchmarks for all board sizes."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_positions = {name: [] for name in INSTINCT_GENERATORS}

    for board_size in BOARD_SIZES:
        print(f"\nGenerating for {board_size}x{board_size}...")

        for instinct_name, generator_cls in INSTINCT_GENERATORS.items():
            generator = generator_cls(board_size)
            positions = generator.generate(positions_per_instinct)

            # Filter valid positions
            valid = [p for p in positions if p.validate()]
            all_positions[instinct_name].extend(valid)

            print(f"  {instinct_name}: {len(valid)} positions")

    # Save to files
    for instinct_name, positions in all_positions.items():
        filepath = output_path / f"{instinct_name}.json"
        data = [p.to_dict() for p in positions]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(positions)} {instinct_name} positions to {filepath}")

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK GENERATION SUMMARY")
    print("=" * 60)
    total = 0
    for instinct_name, positions in all_positions.items():
        print(f"  {instinct_name:12}: {len(positions):4} positions")
        total += len(positions)
    print(f"  {'TOTAL':12}: {total:4} positions")
    print("=" * 60)

    return all_positions


def run_instinct_benchmark(checkpoint_path: str, benchmark_dir: str = 'benchmarks/instincts',
                           verbose: bool = True) -> Dict:
    """Run instinct benchmark on a model."""
    from benchmark import BenchmarkRunner, load_benchmarks, aggregate_results, print_summary
    from model import load_checkpoint
    from config import Config

    # Load model and detect supported board sizes
    print(f"Loading model: {checkpoint_path}")

    # Try to load and get config
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_config = checkpoint.get('config', Config())

    supported_sizes = [model_config.board_size]  # Default to single size

    results = {}

    for board_size in supported_sizes:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing {board_size}x{board_size}")
            print('='*60)

        config = Config(board_size=board_size)
        model, step = load_checkpoint(checkpoint_path, config)

        # Load benchmarks for this size
        positions = load_benchmarks(benchmark_dir, board_size)
        if not positions:
            print(f"  No positions for {board_size}x{board_size}")
            continue

        runner = BenchmarkRunner(model, config)

        # Group by category (instinct)
        by_instinct = {}
        for pos in positions:
            result = runner.evaluate_position(pos)
            result['name'] = pos.name
            result['category'] = pos.category

            if pos.category not in by_instinct:
                by_instinct[pos.category] = []
            by_instinct[pos.category].append(result)

        # Aggregate
        for instinct, instinct_results in by_instinct.items():
            agg = aggregate_results(instinct_results)
            if instinct not in results:
                results[instinct] = {'count': 0, 'correct': 0}
            results[instinct]['count'] += agg['count']
            results[instinct]['correct'] += int(agg['top1_accuracy'] * agg['count'])

            if verbose:
                print(f"  {instinct:12}: {agg['top1_accuracy']:5.1%} ({agg['count']} positions)")

    # Final summary
    if verbose:
        print("\n" + "=" * 60)
        print("INSTINCT BENCHMARK RESULTS")
        print("=" * 60)
        total_count = 0
        total_correct = 0
        for instinct, data in sorted(results.items()):
            acc = data['correct'] / data['count'] if data['count'] > 0 else 0
            print(f"  {instinct:12}: {acc:5.1%} ({data['correct']}/{data['count']})")
            total_count += data['count']
            total_correct += data['correct']

        overall = total_correct / total_count if total_count > 0 else 0
        print(f"  {'OVERALL':12}: {overall:5.1%} ({total_correct}/{total_count})")
        print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Instinct Benchmark Generator")
    parser.add_argument("--generate", action="store_true", help="Generate benchmark positions")
    parser.add_argument("--test", type=str, help="Test a model checkpoint")
    parser.add_argument("--output-dir", default="benchmarks/instincts", help="Output directory")
    parser.add_argument("--positions", type=int, default=25, help="Positions per instinct per board size")
    args = parser.parse_args()

    if args.generate:
        generate_all_benchmarks(args.output_dir, args.positions)
    elif args.test:
        run_instinct_benchmark(args.test, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
