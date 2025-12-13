#!/usr/bin/env python3
"""Generate a showcase game for the blog with tactical commentary."""

import argparse
import os
import torch
from pathlib import Path

from board import Board
from config import Config
from model import GoNet, create_model
from hybrid_mcts import HybridMCTS
from tactics import TacticalAnalyzer
from selfplay import GameRecord


def generate_showcase_game(model_path: str, board_size: int = None,
                           simulations: int = 400, output_dir: str = "showcase"):
    """Generate a single game with rich tactical commentary."""

    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', Config())
    # Use checkpoint's board_size unless explicitly overridden
    if board_size is not None:
        config.board_size = board_size
    print(f"Using board size: {config.board_size}x{config.board_size}")
    config.num_simulations = simulations

    # Use GPU if available, else CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    config.device = str(device)
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Setup hybrid MCTS
    tactics = TacticalAnalyzer()
    mcts = HybridMCTS(model, config, tactics, batch_size=8, tactical_weight=0.5)

    # Play game
    board = Board(config.board_size)
    record = GameRecord(board_size=config.board_size)

    move_count = 0
    max_moves = config.board_size ** 2 * 2

    print(f"\n{'='*50}")
    print(f"Showcase Game: {config.board_size}x{config.board_size}")
    print(f"{'='*50}\n")

    tactical_moments = []

    while not board.is_game_over() and move_count < max_moves:
        # Search with tactical analysis
        policy = mcts.search(board, verbose=False)

        # Select move
        temp = 0.3 if move_count < 10 else 0.0
        action = mcts.select_action(board, temp)

        # Check for tactical events BEFORE the move
        comment_parts = []

        # Check if this move is a capture
        if action != (-1, -1):
            row, col = action
            # Check adjacent groups for potential capture
            captured_already = set()
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < config.board_size and 0 <= nc < config.board_size:
                    if (nr, nc) not in captured_already and board.board[nr, nc] == -board.current_player:
                        # Opponent stone - check liberties
                        group = board.get_group(nr, nc)
                        liberties = board.count_liberties(group)
                        if liberties == 1:  # Will be captured (this move takes last liberty)
                            captured_already.update(group)
                            comment_parts.append(f"Capture {len(group)} stones")
                            tactical_moments.append({
                                'move': move_count + 1,
                                'type': 'capture',
                                'stones': len(group)
                            })

            # Check for snapback (returns stone count gained, 0 if not snapback)
            snapback_gain = tactics.detect_snapback(board, action)
            if snapback_gain > 0:
                comment_parts.append(f"Snapback! (+{snapback_gain} stones)")
                tactical_moments.append({
                    'move': move_count + 1,
                    'type': 'snapback',
                    'gain': snapback_gain
                })

        comment = "; ".join(comment_parts) if comment_parts else ""

        # Record
        record.add(
            board.to_tensor(),
            policy,
            board.current_player,
            move=action,
            comment=comment
        )

        # Print move
        player = "Black" if board.current_player == 1 else "White"
        if action == (-1, -1):
            move_str = "Pass"
        else:
            col_letter = chr(ord('A') + action[1])
            if action[1] >= 8:  # Skip 'I'
                col_letter = chr(ord('A') + action[1] + 1)
            move_str = f"{col_letter}{config.board_size - action[0]}"

        print(f"Move {move_count + 1}: {player} {move_str}", end="")
        if comment:
            print(f"  [{comment}]")
        else:
            print()

        # Play move
        if action == (-1, -1):
            board.pass_move()
        else:
            board.play(action[0], action[1])

        move_count += 1

    # Final score
    score = board.score()
    if score > 0:
        result = f"B+{score:.1f}"
        winner = "Black"
    elif score < 0:
        result = f"W+{-score:.1f}"
        winner = "White"
    else:
        result = "0"
        winner = "Draw"

    record.winner = 1 if score > 0 else (-1 if score < 0 else 0)
    record.result = result
    record.score = score

    print(f"\n{'='*50}")
    print(f"Game Over: {result}")
    print(f"Total moves: {move_count}")
    print(f"{'='*50}\n")

    # Print tactical summary
    if tactical_moments:
        print("Tactical Highlights:")
        for moment in tactical_moments:
            if moment['type'] == 'capture':
                print(f"  Move {moment['move']}: Capture {moment['stones']} stones")
            elif moment['type'] == 'snapback':
                print(f"  Move {moment['move']}: Snapback! (+{moment['gain']})")
            elif moment['type'] == 'ladder':
                print(f"  Move {moment['move']}: Ladder works!")
        print()

    # Save SGF
    os.makedirs(output_dir, exist_ok=True)
    sgf_path = os.path.join(output_dir, "showcase_game.sgf")
    with open(sgf_path, 'w') as f:
        f.write(record.to_sgf(result=result,
                              black_name="GoGoGo Hybrid",
                              white_name="GoGoGo Hybrid"))
    print(f"Saved SGF: {sgf_path}")

    # Also save as blog-ready format
    blog_path = os.path.join(output_dir, "showcase_game_blog.sgf")
    sgf_content = record.to_sgf(result=result,
                                black_name="GoGoGo Neural+Symbolic",
                                white_name="GoGoGo Neural+Symbolic")
    # Add game info comment
    header_comment = f"C[Showcase game from GoGoGo hybrid MCTS.\\n"
    header_comment += f"Board: {config.board_size}x{config.board_size}\\n"
    header_comment += f"Simulations: {simulations}\\n"
    header_comment += f"Result: {result}]"
    sgf_content = sgf_content.replace("RE[", f"{header_comment}\nRE[")

    with open(blog_path, 'w') as f:
        f.write(sgf_content)
    print(f"Saved blog SGF: {blog_path}")

    return record, tactical_moments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate showcase game")
    parser.add_argument("--model", default="checkpoints/best_model.pt",
                       help="Model checkpoint path")
    parser.add_argument("--size", type=int, default=None,
                       help="Board size (default: use checkpoint's size)")
    parser.add_argument("--simulations", type=int, default=400,
                       help="MCTS simulations (default: 400)")
    parser.add_argument("--output", default="showcase",
                       help="Output directory (default: showcase)")

    args = parser.parse_args()

    generate_showcase_game(
        model_path=args.model,
        board_size=args.size,
        simulations=args.simulations,
        output_dir=args.output
    )
