#!/usr/bin/env python3
"""Introspection script to test model predictions at different stages."""
import argparse
import torch
import numpy as np
from config import Config, DEFAULT
from model import load_checkpoint
from board import Board


def print_board(board: Board, highlight=None, probs=None):
    """Print board with optional move probabilities."""
    size = board.size

    # Column labels
    cols = "ABCDEFGHJKLMNOPQRST"[:size]  # Skip I
    print(f"\n    {' '.join(cols)}")
    print(f"   ┌{'─' * (size * 2 - 1)}┐")

    for r in range(size):
        row_str = f"{size - r:2d} │"
        for c in range(size):
            if board.board[r, c] == 1:
                ch = "●"  # Black
            elif board.board[r, c] == -1:
                ch = "○"  # White
            elif highlight and (r, c) == highlight:
                ch = "◆"  # Highlighted move
            elif probs is not None and probs[r, c] > 0.05:
                # Show hot spots
                ch = "·"
            else:
                ch = "."
            row_str += ch + " "
        print(f"{row_str[:-1]}│ {size - r}")

    print(f"   └{'─' * (size * 2 - 1)}┘")
    print(f"    {' '.join(cols)}\n")


def get_top_moves(model, board: Board, top_k=5):
    """Get top k moves from model."""
    model.eval()
    # Check if model expects tactical features (27 planes)
    use_tactical = getattr(model.config, 'input_planes', 17) == 27
    with torch.no_grad():
        tensor = torch.FloatTensor(board.to_tensor(use_tactical_features=use_tactical)).unsqueeze(0)
        tensor = tensor.to(next(model.parameters()).device)
        log_policy, value = model(tensor)
        policy = torch.exp(log_policy).cpu().numpy()[0]
        value = value.cpu().item()  # Convert to Python float

    # Get top moves
    size = board.size
    moves = []
    for idx in policy.argsort()[::-1][:top_k * 2]:  # Get extra in case of illegal
        if idx == size * size:
            move = "pass"
            prob = policy[idx]
        else:
            r, c = idx // size, idx % size
            if board.board[r, c] == 0:  # Legal
                cols = "ABCDEFGHJKLMNOPQRST"
                move = f"{cols[c]}{size - r}"
                prob = policy[idx]
                moves.append((move, prob, (r, c)))
        if len(moves) >= top_k:
            break

    return moves, policy[:-1].reshape(size, size), value  # Exclude pass for heatmap


def test_opening(model, board_size=19):
    """Test 1: Does it play reasonable opening moves?"""
    print("=" * 60)
    print("TEST 1: OPENING (Empty Board)")
    print("=" * 60)
    print("Hypothesis: Should prefer corners/edges, star points, 3-4 points")
    print("Bad sign: Center moves, edge 1st/2nd line")

    board = Board(board_size)
    moves, probs, value = get_top_moves(model, board, top_k=5)

    print(f"\nModel's top 5 first moves:")
    for i, (move, prob, _) in enumerate(moves, 1):
        print(f"  {i}. {move}: {prob:.1%}")

    print(f"\nValue estimate: {value:.3f} (should be ~0 for empty board)")

    # Visualize probability heatmap
    print("\nProbability heatmap (· = >5% probability):")
    print_board(board, highlight=moves[0][2] if moves else None, probs=probs)

    # Check if top moves are sensible
    sensible_first_moves = {'D4', 'Q4', 'D16', 'Q16',  # 4-4 points
                           'D3', 'Q3', 'D17', 'Q17',  # 3-4 points
                           'C4', 'R4', 'C16', 'R16',  # 3-4 other way
                           'D10', 'Q10', 'K4', 'K16', 'K10'}  # Star points

    top_move = moves[0][0] if moves else "none"
    verdict = "✓ GOOD" if top_move in sensible_first_moves else "✗ SUSPICIOUS"
    print(f"Verdict: {verdict} (top move: {top_move})")

    return moves


def test_local_shape(model, board_size=19):
    """Test 2: Does it understand basic local patterns?"""
    print("\n" + "=" * 60)
    print("TEST 2: LOCAL SHAPE (Extend from stone)")
    print("=" * 60)
    print("Setup: Single black stone at D4")
    print("Hypothesis: Should extend/approach, not tenuki to random corner")

    board = Board(board_size)
    board.play(15, 3)  # D4 in 19x19 coordinates

    moves, probs, value = get_top_moves(model, board, top_k=5)

    print(f"\nModel's top 5 responses:")
    for i, (move, prob, _) in enumerate(moves, 1):
        print(f"  {i}. {move}: {prob:.1%}")

    print_board(board, highlight=moves[0][2] if moves else None, probs=probs)

    # Check if moves are local or approach
    local_responses = {'C3', 'C4', 'C5', 'C6', 'D3', 'D5', 'D6',
                       'E3', 'E4', 'E5', 'F3', 'F4', 'G3',
                       'Q4', 'Q16', 'D16', 'R4', 'R16'}  # Or other corners

    top_move = moves[0][0] if moves else "none"
    verdict = "✓ GOOD" if top_move in local_responses else "? CHECK"
    print(f"Verdict: {verdict} (top move: {top_move})")

    return moves


def test_capture(model, board_size=19):
    """Test 3: Does it see obvious captures?"""
    print("\n" + "=" * 60)
    print("TEST 3: CAPTURE (Atari situation)")
    print("=" * 60)
    print("Setup: White stone at E6 in atari (1 liberty at E7)")
    print("Hypothesis: Should see the capture at E7")

    board = Board(board_size)
    # Create REAL atari situation: white E6 surrounded on 3 sides
    board.play(14, 4)   # Black E5
    board.play(13, 4)   # White E6
    board.play(13, 3)   # Black D6
    board.play(3, 15)   # White Q16 (filler)
    board.play(13, 5)   # Black F6 - white E6 now in atari!
    board.play(3, 16)   # White R16 (filler)
    # Now black to play, E7 captures

    moves, probs, value = get_top_moves(model, board, top_k=5)

    print(f"\nModel's top 5 moves (Black to play):")
    for i, (move, prob, _) in enumerate(moves, 1):
        print(f"  {i}. {move}: {prob:.1%}")

    print_board(board, highlight=moves[0][2] if moves else None, probs=probs)

    capture_move = "E7"  # This captures the white stone at E6
    top_move = moves[0][0] if moves else "none"

    if top_move == capture_move:
        verdict = "✓ SEES CAPTURE"
    elif capture_move in [m[0] for m in moves[:3]]:
        verdict = "~ KNOWS CAPTURE (not top)"
    else:
        verdict = "✗ MISSES CAPTURE"

    print(f"Verdict: {verdict} (capture at {capture_move}, top move: {top_move})")

    return moves


def test_self_play(model, board_size=9, max_moves=30):
    """Test 4: Watch it play against itself."""
    print("\n" + "=" * 60)
    print(f"TEST 4: SELF-PLAY ({board_size}x{board_size}, first {max_moves} moves)")
    print("=" * 60)
    print("Watching the model play against itself...")

    board = Board(board_size)
    move_list = []

    for i in range(max_moves):
        if board.is_game_over():
            break

        moves, _, _ = get_top_moves(model, board, top_k=1)
        if not moves:
            board.pass_move()
            move_list.append("pass")
            continue

        move_str, prob, (r, c) = moves[0]

        if board.board[r, c] != 0:
            board.pass_move()
            move_list.append("pass")
        else:
            board.play(r, c)
            color = "B" if board.current_player == -1 else "W"  # After play, player switched
            move_list.append(f"{color}:{move_str}")

    print(f"\nGame record ({len(move_list)} moves):")
    for i in range(0, len(move_list), 10):
        chunk = move_list[i:i+10]
        print(f"  {i+1:2d}. " + " ".join(chunk))

    print("\nFinal position:")
    print_board(board)

    # Simple assessment
    b_stones = np.sum(board.board == 1)
    w_stones = np.sum(board.board == -1)
    print(f"Stones: Black={b_stones}, White={w_stones}")

    return move_list


def main():
    parser = argparse.ArgumentParser(description='Introspect model predictions')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/supervised_best.pt',
                        help='Path to checkpoint')
    parser.add_argument('--board-size', type=int, default=19,
                        help='Board size for tests')
    parser.add_argument('--self-play-size', type=int, default=9,
                        help='Board size for self-play test')
    args = parser.parse_args()

    # Load model
    config = DEFAULT
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading checkpoint: {args.checkpoint}")
    model, step = load_checkpoint(args.checkpoint, config)
    config = model.config
    print(f"Model: {config.num_blocks} blocks, {config.num_filters} filters")
    print(f"Input planes: {getattr(config, 'input_planes', 17)} ({'tactical' if getattr(config, 'input_planes', 17) == 27 else 'basic'})")
    print(f"Device: {config.device}")
    print(f"Trained for {step} steps")

    # Adjust board size if model was trained on different size
    board_size = args.board_size if args.board_size == config.board_size else config.board_size

    # Run tests
    test_opening(model, board_size)
    test_local_shape(model, board_size)
    test_capture(model, board_size)
    # Self-play must use model's board size
    test_self_play(model, board_size=config.board_size, max_moves=50)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Run this periodically during training to see improvement!")
    print(f"  poetry run python introspect.py --checkpoint {args.checkpoint}")


if __name__ == '__main__':
    main()
