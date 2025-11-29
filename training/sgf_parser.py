"""Simple SGF parser for extracting games."""
import re
from pathlib import Path
from typing import List, Tuple, Optional
from board import Board
import numpy as np


def sgf_to_coords(sgf_pos: str, board_size: int) -> Optional[Tuple[int, int]]:
    """Convert SGF coordinates (e.g., 'pd') to board coordinates."""
    if not sgf_pos or len(sgf_pos) < 2:
        return None  # Pass move

    col = ord(sgf_pos[0]) - ord('a')
    row = ord(sgf_pos[1]) - ord('a')

    if col >= board_size or row >= board_size:
        return None

    return (row, col)


def parse_sgf_file(sgf_path: str, target_size: int = 19) -> Optional[List[Tuple[np.ndarray, int, int]]]:
    """Parse SGF file and extract (board_tensor, row, col) training samples.

    Returns list of (board_state, move_row, move_col) tuples.
    Only returns games of target_size.
    """
    try:
        with open(sgf_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return None

    # Extract board size
    size_match = re.search(r'SZ\[(\d+)\]', content)
    if size_match:
        board_size = int(size_match.group(1))
        if board_size != target_size:
            return None
    else:
        board_size = 19  # Default
        if board_size != target_size:
            return None

    # Extract moves: B[xx] or W[xx]
    move_pattern = re.compile(r';([BW])\[([a-z]*)\]')
    moves = move_pattern.findall(content)

    if len(moves) < 20:  # Skip very short games
        return None

    # Replay game and collect training samples
    board = Board(board_size)
    samples = []

    for color_str, sgf_pos in moves:
        # Get board state before move
        board_tensor = board.to_tensor()

        # Parse move
        coords = sgf_to_coords(sgf_pos, board_size)

        if coords is None:
            # Pass move - skip for training
            board.pass_move()
            continue

        row, col = coords

        # Verify move is legal
        if not board.is_valid_move(row, col):
            # Illegal move in SGF - skip this game
            return None

        # Record training sample: (board before move, move coordinates)
        samples.append((board_tensor, row, col))

        # Play move
        try:
            board.play(row, col)
        except Exception:
            # Game has errors - skip
            return None

    return samples if len(samples) >= 20 else None


def load_sgf_dataset(data_dir: str, target_size: int = 19, max_games: int = None, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load all SGF files from directory and create training dataset.

    Returns:
        states: (N, planes, size, size) board states
        moves: (N, 2) move coordinates (row, col)
    """
    # Check cache first
    cache_dir = Path('dataset_cache')
    cache_dir.mkdir(exist_ok=True)
    cache_key = f"size{target_size}_max{max_games or 'all'}"
    cache_file = cache_dir / f"{cache_key}.npz"

    if use_cache and cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        data = np.load(cache_file)
        return data['states'], data['moves']

    sgf_files = list(Path(data_dir).rglob('*.sgf'))

    if max_games:
        sgf_files = sgf_files[:max_games]

    all_states = []
    all_moves = []

    print(f"Loading {len(sgf_files)} SGF files...")

    for i, sgf_path in enumerate(sgf_files):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(sgf_files)} files, collected {len(all_states)} positions")

        samples = parse_sgf_file(str(sgf_path), target_size)
        if samples is None:
            continue

        for board_tensor, row, col in samples:
            all_states.append(board_tensor)
            all_moves.append([row, col])

    print(f"Loaded {len(all_states)} training positions from {len(sgf_files)} games")

    states = np.array(all_states, dtype=np.float32)
    moves = np.array(all_moves, dtype=np.int64)

    # Save to cache
    if use_cache:
        print(f"Saving to cache: {cache_file}")
        np.savez(cache_file, states=states, moves=moves)

    return states, moves


if __name__ == '__main__':
    # Test
    states, moves = load_sgf_dataset('data/games', target_size=9, max_games=100)
    print(f"Dataset: {states.shape}, {moves.shape}")
    print(f"Sample move: {moves[0]}")
