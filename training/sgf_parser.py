"""Simple SGF parser for extracting games."""
import re
import os
from pathlib import Path
from typing import List, Tuple, Optional
from functools import partial
from multiprocessing import Pool, cpu_count
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


def parse_game_result(content: str) -> Optional[float]:
    """Parse game result from SGF content.

    Returns:
        +1.0 if Black wins
        -1.0 if White wins
        None if result unclear/void
    """
    # RE[B+...] or RE[W+...] or RE[B+Resign] etc.
    result_match = re.search(r'RE\[([BW])\+', content)
    if result_match:
        winner = result_match.group(1)
        return 1.0 if winner == 'B' else -1.0

    # Also check for RE[0] (jigo/draw) - rare but possible
    if re.search(r'RE\[0\]', content):
        return 0.0

    return None


def parse_sgf_file(sgf_path: str, target_size: int = 19, tactical_features: bool = False,
                   include_value: bool = False, include_ownership: bool = False,
                   include_opponent_move: bool = False) -> Optional[List[Tuple]]:
    """Parse SGF file and extract training samples.

    Returns list of tuples (fields depend on options):
        - Base: (board_state, move_row, move_col)
        - With value: adds game_result
        - With ownership: adds ownership_map (same for all positions in game)
        - With opponent_move: adds (opp_row, opp_col) - opponent's response

    Args:
        tactical_features: If True, include neuro-symbolic tactical planes (27 total)
        include_value: If True, include game result for value head training
        include_ownership: If True, include ownership map from final board state
                          (KataGo's key insight: 361x more signal per game)
        include_opponent_move: If True, include opponent's next move for auxiliary target
                              (KataGo: 1.30x speedup from predicting opponent's response)
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

    # Extract game result if needed
    game_result = None
    if include_value:
        game_result = parse_game_result(content)
        if game_result is None:
            return None  # Skip games without clear result

    # Extract moves: B[xx] or W[xx]
    move_pattern = re.compile(r';([BW])\[([a-z]*)\]')
    moves = move_pattern.findall(content)

    if len(moves) < 20:  # Skip very short games
        return None

    # First pass: replay entire game to get final board state (for ownership)
    # Second pass: collect samples with ownership target
    board = Board(board_size)
    positions_before_move = []  # Store (tensor, row, col, current_player) tuples

    for color_str, sgf_pos in moves:
        # Get board state before move
        board_tensor = board.to_tensor(use_tactical_features=tactical_features).astype(np.float16)
        current_player = board.current_player

        # Parse move
        coords = sgf_to_coords(sgf_pos, board_size)

        if coords is None:
            # Pass move - skip for training but continue game
            board.pass_move()
            continue

        row, col = coords

        # Verify move is legal
        if not board.is_valid_move(row, col):
            return None

        # Store position data
        positions_before_move.append((board_tensor, row, col, current_player))

        # Play move
        try:
            board.play(row, col)
        except Exception:
            return None

    if len(positions_before_move) < 20:
        return None

    # Compute ownership from FINAL board state (applies to ALL positions)
    final_ownership = None
    if include_ownership:
        final_ownership = board.ownership_map().astype(np.float16)

    # Build samples with all requested data
    samples = []
    n_positions = len(positions_before_move)

    for i, (board_tensor, row, col, current_player) in enumerate(positions_before_move):
        sample = [board_tensor, row, col]

        if include_value:
            # Value from current player's perspective
            value_target = game_result if current_player == 1 else -game_result
            sample.append(value_target)

        if include_ownership:
            # Ownership from current player's perspective
            # If white to play, negate ownership (so +1 = good for current player)
            ownership_target = final_ownership if current_player == 1 else -final_ownership
            sample.append(ownership_target)

        if include_opponent_move:
            # Opponent's response is the next move in the sequence
            # Skip last position (no opponent response available)
            if i + 1 < n_positions:
                _, opp_row, opp_col, _ = positions_before_move[i + 1]
                sample.append(opp_row)
                sample.append(opp_col)
            else:
                # Last position - skip it when opponent move is required
                continue

        samples.append(tuple(sample))

    return samples


def has_tactical_activity(board_tensor: np.ndarray) -> bool:
    """Check if position has tactical activity (atari or capture opportunities).

    Tactical planes (when tactical_features=True):
        - Plane 20: Opponent groups with 1 liberty (atari)
        - Plane 23: Capture moves available

    Returns True if either plane has non-zero values.
    """
    if board_tensor.shape[0] < 24:
        return False  # Basic features only, no tactical planes
    # Plane 20: opponent atari, Plane 23: capture moves
    return board_tensor[20].any() or board_tensor[23].any()


def load_sgf_dataset(data_dir: str, target_size: int = 19, max_games: int = None,
                     use_cache: bool = True, tactical_features: bool = False,
                     max_positions: int = None, include_value: bool = False,
                     include_tactical_mask: bool = False, include_ownership: bool = False,
                     include_opponent_move: bool = False):
    """Load all SGF files from directory and create training dataset.

    Args:
        tactical_features: If True, include neuro-symbolic tactical planes (27 instead of 17)
        max_positions: Maximum number of positions to load (memory safety)
        include_value: If True, also return value targets from game results
        include_tactical_mask: If True, also return boolean mask for tactical positions
        include_ownership: If True, also return ownership maps from final board state
                          (KataGo's key insight: 361x more signal per game)
        include_opponent_move: If True, also return opponent's next move for auxiliary target
                              (KataGo: 1.30x speedup)

    Returns:
        states: (N, planes, size, size) board states
        moves: (N, 2) move coordinates (row, col)
        values: (N,) value targets (only if include_value=True)
        tactical_mask: (N,) boolean mask (only if include_tactical_mask=True)
        ownership: (N, size, size) ownership maps (only if include_ownership=True)
        opponent_moves: (N, 2) opponent move coordinates (only if include_opponent_move=True)
    """
    # Check cache first
    cache_dir = Path('dataset_cache')
    cache_dir.mkdir(exist_ok=True)
    tactical_suffix = "_tactical" if tactical_features else ""
    pos_suffix = f"_pos{max_positions}" if max_positions else ""
    value_suffix = "_value" if include_value else ""
    mask_suffix = "_mask" if include_tactical_mask else ""
    ownership_suffix = "_ownership" if include_ownership else ""
    opponent_suffix = "_opponent" if include_opponent_move else ""
    cache_key = f"size{target_size}_max{max_games or 'all'}{tactical_suffix}{pos_suffix}{value_suffix}{mask_suffix}{ownership_suffix}{opponent_suffix}"
    cache_file = cache_dir / f"{cache_key}.npz"

    if use_cache and cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        data = np.load(cache_file)
        result = [data['states'], data['moves']]
        if include_value:
            result.append(data['values'])
        if include_tactical_mask:
            result.append(data['tactical_mask'])
        if include_ownership:
            result.append(data['ownership'])
        if include_opponent_move:
            result.append(data['opponent_moves'])
        return tuple(result) if len(result) > 2 else (result[0], result[1])

    sgf_files = list(Path(data_dir).rglob('*.sgf'))

    if max_games:
        sgf_files = sgf_files[:max_games]

    all_states = []
    all_moves = []
    all_values = [] if include_value else None
    all_tactical = [] if include_tactical_mask else None
    all_ownership = [] if include_ownership else None
    all_opponent_moves = [] if include_opponent_move else None
    position_limit_reached = False

    feature_type = "tactical (27 planes)" if tactical_features else "basic (17 planes)"
    value_str = " + value targets" if include_value else ""
    ownership_str = " + ownership" if include_ownership else ""
    opponent_str = " + opponent move" if include_opponent_move else ""
    n_workers = min(cpu_count(), 16)  # Use up to 16 workers
    limit_str = f", max {max_positions} positions" if max_positions else ""
    print(f"Loading {len(sgf_files)} SGF files with {feature_type}{value_str}{ownership_str}{opponent_str} using {n_workers} workers{limit_str}...")

    # Create partial function with fixed arguments
    parse_fn = partial(parse_sgf_file, target_size=target_size, tactical_features=tactical_features,
                      include_value=include_value, include_ownership=include_ownership,
                      include_opponent_move=include_opponent_move)
    sgf_paths = [str(p) for p in sgf_files]

    # Parallel loading with position limit
    with Pool(n_workers) as pool:
        for i, samples in enumerate(pool.imap(parse_fn, sgf_paths, chunksize=50)):
            if samples is not None:
                for sample in samples:
                    # Unpack sample tuple based on what was requested
                    # Order: (board_tensor, row, col, [value], [ownership], [opp_row, opp_col])
                    idx = 0
                    board_tensor = sample[idx]; idx += 1
                    row = sample[idx]; idx += 1
                    col = sample[idx]; idx += 1

                    if include_value:
                        value = sample[idx]; idx += 1
                        all_values.append(value)

                    if include_ownership:
                        ownership = sample[idx]; idx += 1
                        all_ownership.append(ownership)

                    if include_opponent_move:
                        opp_row = sample[idx]; idx += 1
                        opp_col = sample[idx]; idx += 1
                        all_opponent_moves.append([opp_row, opp_col])

                    all_states.append(board_tensor)
                    all_moves.append([row, col])

                    # Track tactical activity for curriculum learning
                    if include_tactical_mask:
                        is_tactical = has_tactical_activity(board_tensor)
                        all_tactical.append(is_tactical)

                    # Check position limit
                    if max_positions and len(all_states) >= max_positions:
                        position_limit_reached = True
                        break

                if position_limit_reached:
                    print(f"Reached position limit ({max_positions})")
                    break

            if (i + 1) % 500 == 0:
                mem_mb = len(all_states) * all_states[0].nbytes / (1024 * 1024) if all_states else 0
                print(f"Processed {i + 1}/{len(sgf_files)} files, {len(all_states)} positions (~{mem_mb:.0f}MB)")

    print(f"Loaded {len(all_states)} training positions from {len(sgf_files)} games")

    # Use float16 for storage to save memory (will convert to float32 during training)
    states = np.array(all_states, dtype=np.float16)
    moves = np.array(all_moves, dtype=np.int64)

    mem_gb = states.nbytes / (1024**3)
    print(f"Dataset size: {mem_gb:.2f}GB")

    # Build result arrays
    values = np.array(all_values, dtype=np.float32) if include_value else None
    tactical_mask = np.array(all_tactical, dtype=np.bool_) if include_tactical_mask else None
    ownership = np.array(all_ownership, dtype=np.float16) if include_ownership else None
    opponent_moves = np.array(all_opponent_moves, dtype=np.int64) if include_opponent_move else None

    if include_tactical_mask:
        n_tactical = tactical_mask.sum()
        print(f"Tactical positions: {n_tactical:,} ({100*n_tactical/len(tactical_mask):.1f}%)")

    if include_ownership:
        ownership_mem_mb = ownership.nbytes / (1024 * 1024)
        print(f"Ownership maps: {ownership.shape} ({ownership_mem_mb:.0f}MB)")

    if include_opponent_move:
        print(f"Opponent moves: {opponent_moves.shape}")

    # Save to cache
    if use_cache:
        print(f"Saving to cache: {cache_file}")
        save_dict = {'states': states, 'moves': moves}
        if include_value:
            save_dict['values'] = values
        if include_tactical_mask:
            save_dict['tactical_mask'] = tactical_mask
        if include_ownership:
            save_dict['ownership'] = ownership
        if include_opponent_move:
            save_dict['opponent_moves'] = opponent_moves
        np.savez_compressed(cache_file, **save_dict)

    # Return results based on what was requested
    result = [states, moves]
    if include_value:
        result.append(values)
    if include_tactical_mask:
        result.append(tactical_mask)
    if include_ownership:
        result.append(ownership)
    if include_opponent_move:
        result.append(opponent_moves)
    return tuple(result) if len(result) > 2 else (result[0], result[1])


if __name__ == '__main__':
    # Test
    states, moves = load_sgf_dataset('data/games', target_size=9, max_games=100)
    print(f"Dataset: {states.shape}, {moves.shape}")
    print(f"Sample move: {moves[0]}")
