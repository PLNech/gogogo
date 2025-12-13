#!/usr/bin/env python3
"""Compare network predictions vs actual pro moves in SGF games.

Shows where the network agrees/disagrees with pro play and identifies
patterns of strength and weakness.
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import re

from board import Board
from model import load_checkpoint
from config import Config


def parse_sgf_moves(sgf_content: str) -> List[Tuple[str, Tuple[int, int]]]:
    """Parse SGF content and return list of (color, (row, col)) moves."""
    moves = []
    # Match ;B[xx] or ;W[xx] or B[xx] or W[xx]
    pattern = r'([BW])\[([a-s]{0,2})\]'

    for match in re.finditer(pattern, sgf_content):
        color, coord = match.groups()
        if len(coord) == 2:
            c = ord(coord[0]) - ord('a')
            r = ord(coord[1]) - ord('a')
            moves.append((color, (r, c)))
        elif len(coord) == 0:
            # Pass move
            moves.append((color, (-1, -1)))

    return moves


def get_network_prediction(model, board: Board, use_tactical: bool = True) -> Tuple[np.ndarray, float]:
    """Get policy and value from network."""
    model.eval()
    with torch.no_grad():
        tensor = board.to_tensor(use_tactical_features=use_tactical)
        x = torch.FloatTensor(tensor).unsqueeze(0).to(next(model.parameters()).device)
        outputs = model(x)
        log_policy = outputs[0]
        value = outputs[1]
        policy = torch.exp(log_policy).cpu().numpy()[0]
        value_scalar = value.cpu().numpy()[0, 0]
    return policy, value_scalar


def analyze_game(model, sgf_path: str, use_tactical: bool = True, verbose: bool = False) -> Dict:
    """Analyze a single game and return statistics."""
    with open(sgf_path, 'r') as f:
        content = f.read()

    moves = parse_sgf_moves(content)
    if not moves:
        return {'error': 'No moves found'}

    # Determine board size from SGF
    size_match = re.search(r'SZ\[(\d+)\]', content)
    board_size = int(size_match.group(1)) if size_match else 19

    if board_size != model.config.board_size:
        return {'error': f'Board size mismatch: {board_size} vs {model.config.board_size}'}

    board = Board(board_size)
    stats = {
        'total_moves': 0,
        'top1_correct': 0,
        'top3_correct': 0,
        'top5_correct': 0,
        'top10_correct': 0,
        'rank_sum': 0,
        'prob_sum': 0.0,
        'by_phase': {
            'opening': {'total': 0, 'top1': 0, 'top3': 0},  # moves 1-50
            'middle': {'total': 0, 'top1': 0, 'top3': 0},   # moves 51-150
            'endgame': {'total': 0, 'top1': 0, 'top3': 0},  # moves 151+
        },
        'disagreements': [],  # Interesting cases where network strongly disagrees
    }

    col_labels = 'ABCDEFGHJKLMNOPQRST'[:board_size]

    for move_num, (color, move) in enumerate(moves):
        if move == (-1, -1):  # Pass
            board.pass_move()
            continue

        r, c = move

        # Get network prediction before playing the move
        policy, value = get_network_prediction(model, board, use_tactical)

        # Find rank of actual move
        board_policy = policy[:-1].reshape(board_size, board_size)
        actual_prob = board_policy[r, c]

        # Get all legal move probabilities
        legal_probs = []
        for row in range(board_size):
            for col in range(board_size):
                if board.is_valid_move(row, col):
                    legal_probs.append((board_policy[row, col], (row, col)))
        legal_probs.sort(reverse=True)

        # Find rank
        rank = None
        for i, (prob, pos) in enumerate(legal_probs):
            if pos == move:
                rank = i + 1
                break

        if rank is None:
            # Move was illegal according to our board (shouldn't happen)
            if verbose:
                print(f"  Move {move_num}: {col_labels[c]}{board_size-r} - ILLEGAL?")
            board.play(r, c)  # Try to play anyway
            continue

        # Update statistics
        stats['total_moves'] += 1
        stats['rank_sum'] += rank
        stats['prob_sum'] += actual_prob

        if rank == 1:
            stats['top1_correct'] += 1
        if rank <= 3:
            stats['top3_correct'] += 1
        if rank <= 5:
            stats['top5_correct'] += 1
        if rank <= 10:
            stats['top10_correct'] += 1

        # Phase statistics
        if move_num < 50:
            phase = 'opening'
        elif move_num < 150:
            phase = 'middle'
        else:
            phase = 'endgame'

        stats['by_phase'][phase]['total'] += 1
        if rank == 1:
            stats['by_phase'][phase]['top1'] += 1
        if rank <= 3:
            stats['by_phase'][phase]['top3'] += 1

        # Track interesting disagreements
        if rank > 10 and actual_prob < 0.01:
            top_move = legal_probs[0][1] if legal_probs else None
            top_prob = legal_probs[0][0] if legal_probs else 0
            if top_move:
                stats['disagreements'].append({
                    'move_num': move_num,
                    'actual': f"{col_labels[c]}{board_size-r}",
                    'actual_rank': rank,
                    'actual_prob': actual_prob,
                    'network_top': f"{col_labels[top_move[1]]}{board_size-top_move[0]}",
                    'network_prob': top_prob,
                })

        if verbose and move_num < 30:  # Show first 30 moves
            actual_coord = f"{col_labels[c]}{board_size-r}"
            top_coord = f"{col_labels[legal_probs[0][1][1]]}{board_size-legal_probs[0][1][0]}" if legal_probs else "?"
            mark = "✓" if rank <= 3 else "✗"
            print(f"  {move_num:3d}. {color}[{actual_coord}] rank={rank:2d} ({actual_prob:5.1%}) | net: {top_coord} ({legal_probs[0][0]:5.1%}) {mark}")

        # Play the move
        try:
            board.play(r, c)
        except:
            break  # Game might have illegal moves due to parsing issues

    return stats


def print_game_summary(stats: Dict, game_name: str = ""):
    """Print summary for a single game."""
    if 'error' in stats:
        print(f"  {game_name}: {stats['error']}")
        return

    n = stats['total_moves']
    if n == 0:
        print(f"  {game_name}: No moves analyzed")
        return

    top1 = stats['top1_correct'] / n * 100
    top3 = stats['top3_correct'] / n * 100
    avg_rank = stats['rank_sum'] / n
    avg_prob = stats['prob_sum'] / n * 100

    print(f"  {game_name}: {n} moves | Top-1: {top1:.1f}% | Top-3: {top3:.1f}% | Avg rank: {avg_rank:.1f} | Avg prob: {avg_prob:.1f}%")


def print_aggregate_summary(all_stats: List[Dict]):
    """Print aggregate summary across all games."""
    total_moves = sum(s.get('total_moves', 0) for s in all_stats)
    if total_moves == 0:
        print("No moves analyzed")
        return

    top1 = sum(s.get('top1_correct', 0) for s in all_stats) / total_moves * 100
    top3 = sum(s.get('top3_correct', 0) for s in all_stats) / total_moves * 100
    top5 = sum(s.get('top5_correct', 0) for s in all_stats) / total_moves * 100
    top10 = sum(s.get('top10_correct', 0) for s in all_stats) / total_moves * 100
    avg_rank = sum(s.get('rank_sum', 0) for s in all_stats) / total_moves
    avg_prob = sum(s.get('prob_sum', 0) for s in all_stats) / total_moves * 100

    print("\n" + "=" * 60)
    print("AGGREGATE SUMMARY")
    print("=" * 60)
    print(f"Total moves analyzed: {total_moves:,}")
    print(f"Top-1 accuracy:  {top1:5.1f}%")
    print(f"Top-3 accuracy:  {top3:5.1f}%")
    print(f"Top-5 accuracy:  {top5:5.1f}%")
    print(f"Top-10 accuracy: {top10:5.1f}%")
    print(f"Average rank:    {avg_rank:5.1f}")
    print(f"Average prob:    {avg_prob:5.1f}%")

    # By phase
    print("\nBy game phase:")
    for phase in ['opening', 'middle', 'endgame']:
        phase_total = sum(s.get('by_phase', {}).get(phase, {}).get('total', 0) for s in all_stats)
        if phase_total > 0:
            phase_top1 = sum(s.get('by_phase', {}).get(phase, {}).get('top1', 0) for s in all_stats) / phase_total * 100
            phase_top3 = sum(s.get('by_phase', {}).get(phase, {}).get('top3', 0) for s in all_stats) / phase_total * 100
            print(f"  {phase:8s}: {phase_total:5d} moves | Top-1: {phase_top1:5.1f}% | Top-3: {phase_top3:5.1f}%")

    # Notable disagreements
    all_disagreements = []
    for s in all_stats:
        all_disagreements.extend(s.get('disagreements', []))

    if all_disagreements:
        print(f"\nNotable disagreements ({len(all_disagreements)} total):")
        # Show first few
        for d in all_disagreements[:5]:
            print(f"  Move {d['move_num']}: Pro played {d['actual']} (rank {d['actual_rank']}, {d['actual_prob']:.1%})")
            print(f"           Network preferred {d['network_top']} ({d['network_prob']:.1%})")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Compare network predictions vs pro moves')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('sgf', type=str, nargs='+', help='SGF file(s) or directory to analyze')
    parser.add_argument('--max-games', type=int, default=10, help='Max games to analyze')
    parser.add_argument('--verbose', action='store_true', help='Show move-by-move analysis')
    parser.add_argument('--no-tactical', action='store_true', help='Disable tactical features')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    config = Config()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, step = load_checkpoint(args.checkpoint, config)
    print(f"Loaded model (step {step}, board_size={model.config.board_size})")

    use_tactical = not args.no_tactical and model.config.input_planes == 27
    print(f"Using {'tactical' if use_tactical else 'basic'} features\n")

    # Collect SGF files
    sgf_files = []
    for path in args.sgf:
        p = Path(path)
        if p.is_dir():
            sgf_files.extend(sorted(p.glob('**/*.sgf'))[:args.max_games])
        elif p.suffix == '.sgf':
            sgf_files.append(p)

    if not sgf_files:
        print("No SGF files found")
        return

    print(f"Analyzing {len(sgf_files)} games...")

    all_stats = []
    for i, sgf_path in enumerate(sgf_files[:args.max_games]):
        print(f"\n[{i+1}/{len(sgf_files)}] {sgf_path.name}")
        stats = analyze_game(model, str(sgf_path), use_tactical, verbose=args.verbose)
        all_stats.append(stats)
        print_game_summary(stats, sgf_path.stem)

    print_aggregate_summary(all_stats)


if __name__ == '__main__':
    main()
