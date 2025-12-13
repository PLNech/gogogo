#!/usr/bin/env python3
"""Compare 3 move selection variants:
1. Direct policy (no MCTS) - just use neural network output
2. MCTS with value weight = 0 (policy-only search)
3. MCTS with high policy weight (c_puct = 5.0 instead of 1.5)
"""

import numpy as np
import torch
from board import Board
from config import DEFAULT
from model import load_checkpoint
from mcts import MCTS, MCTSNode
import time


def format_move(move, board_size=19):
    if move == (-1, -1):
        return "PASS"
    r, c = move
    cols = "ABCDEFGHJKLMNOPQRST"[:board_size]
    return f"{cols[c]}{board_size - r}"


def get_direct_policy_move(model, board, config):
    """Variant 1: Direct neural network policy (no MCTS)."""
    use_tactical = config.input_planes == 27
    tensor = board.to_tensor(use_tactical_features=use_tactical)

    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(np.expand_dims(tensor, 0))
        log_policy, value = model(x)
        policy = torch.exp(log_policy).cpu().numpy()[0]

    # Mask illegal moves
    legal = board.get_legal_moves()
    legal_mask = np.zeros(362)
    for move in legal:
        if move == (-1, -1):
            legal_mask[361] = 1
        else:
            legal_mask[move[0] * 19 + move[1]] = 1

    policy = policy * legal_mask
    if policy.sum() > 0:
        policy = policy / policy.sum()

    action_idx = np.argmax(policy)
    if action_idx == 361:
        return (-1, -1), policy[action_idx]
    return (action_idx // 19, action_idx % 19), policy[action_idx]


class PolicyOnlyMCTS(MCTS):
    """Variant 2: MCTS that ignores value estimates (policy-only)."""

    def _batch_predict(self, tensors):
        """Override to return neutral values."""
        policies, values = super()._batch_predict(tensors)
        # Return 0 for all values - makes MCTS rely only on policy priors
        return policies, np.zeros_like(values)


class HighPolicyMCTS(MCTS):
    """Variant 3: MCTS with higher policy weight (c_puct=5.0)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override c_puct to weight policy more heavily
        self.config.c_puct = 5.0


def setup_test_positions():
    """Create 10 test positions covering different game phases."""
    positions = []

    # 1. Empty board (opening)
    b = Board(19)
    positions.append(("Empty board (opening)", b, "corner 4-4 or 3-4"))

    # 2. After one corner move
    b = Board(19)
    b.board[3, 3] = 1  # D16
    b.current_player = -1
    positions.append(("After D16", b, "opposite corner"))

    # 3. Two corners taken
    b = Board(19)
    b.board[3, 3] = 1   # D16
    b.board[15, 15] = -1  # Q4
    b.current_player = 1
    positions.append(("Two corners", b, "empty corner"))

    # 4. Approach move situation
    b = Board(19)
    b.board[3, 15] = 1  # Q16 black
    b.current_player = -1
    positions.append(("Approach Q16", b, "approach (R14, O17, etc)"))

    # 5. Simple atari - black group needs saving
    b = Board(19)
    b.board[9, 9] = 1   # Black at center
    b.board[9, 10] = -1  # White right
    b.board[10, 9] = -1  # White below
    b.board[8, 9] = -1   # White above - black in atari!
    b.current_player = 1  # Black to save
    positions.append(("Black in atari", b, "extend (J9 or K10)"))

    # 6. White can capture
    b = Board(19)
    b.board[9, 9] = 1   # Black
    b.board[9, 10] = -1  # White
    b.board[10, 9] = -1  # White
    b.board[8, 9] = -1   # White - black in atari
    b.current_player = -1  # White to capture at J9
    positions.append(("White captures", b, "capture at K11 (9,8)"))

    # 7. 3-stone capture
    b = Board(19)
    b.board[9, 9] = 1
    b.board[9, 10] = 1
    b.board[9, 11] = 1
    b.board[10, 9] = -1
    b.board[10, 10] = -1
    b.board[10, 11] = -1
    b.board[9, 8] = -1
    b.board[9, 12] = -1
    b.board[8, 9] = -1
    b.board[8, 11] = -1
    b.current_player = -1  # White captures at (8,10)
    positions.append(("3-stone capture", b, "L11 (8,10)"))

    # 8. Ko situation
    b = Board(19)
    b.board[9, 9] = 1
    b.board[9, 10] = -1
    b.board[10, 10] = 1
    b.board[10, 9] = -1
    b.board[8, 10] = 1
    b.board[9, 11] = 1
    b.board[10, 11] = -1
    b.board[11, 10] = -1
    b.current_player = 1
    positions.append(("Complex fight", b, "extend/connect"))

    # 9. Side extension
    b = Board(19)
    b.board[3, 3] = 1   # D16
    b.board[3, 9] = 1   # K16
    b.board[3, 15] = -1  # Q16
    b.current_player = 1
    positions.append(("Side extension", b, "extend along top"))

    # 10. Endgame - small moves
    b = Board(19)
    # Fill corners with stones to simulate endgame
    for r in range(4):
        for c in range(4):
            b.board[r, c] = 1
    for r in range(15, 19):
        for c in range(15, 19):
            b.board[r, c] = -1
    b.current_player = 1
    positions.append(("Endgame territory", b, "boundary move"))

    return positions


def evaluate_move_quality(move, expected_hint, board):
    """Simple heuristic to score move quality."""
    r, c = move
    size = board.size

    # Opening: corners are good
    if np.sum(np.abs(board.board)) < 5:
        # Check if it's a corner area move (3-4, 4-4, 3-3 points)
        corner_points = [
            (2, 2), (2, 3), (3, 2), (3, 3),  # Upper left
            (2, 15), (2, 16), (3, 15), (3, 16),  # Upper right
            (15, 2), (15, 3), (16, 2), (16, 3),  # Lower left
            (15, 15), (15, 16), (16, 15), (16, 16)  # Lower right
        ]
        if (r, c) in corner_points:
            return "GOOD", 2
        # Edge moves are bad in opening
        if r in [0, 1, 17, 18] or c in [0, 1, 17, 18]:
            return "BAD", 0
        return "OK", 1

    # Tactical: check if it's a capture/save move
    if "capture" in expected_hint.lower() or "extend" in expected_hint.lower():
        # Check if move is adjacent to stones
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                if board.board[nr, nc] != 0:
                    return "TACTICAL", 2

    return "NEUTRAL", 1


def main():
    # Load model
    print("Loading model...")
    config = DEFAULT
    config.device = 'cpu'
    model, step = load_checkpoint('checkpoints/supervised_best.pt', config)
    config = model.config
    config.mcts_simulations = 100
    config.device = 'cpu'

    print(f"Model step: {step}")
    print(f"Testing 3 variants on 10 positions\n")

    # Setup variants
    variants = {
        "Direct Policy": lambda b: get_direct_policy_move(model, b, config),
        "Policy-Only MCTS": None,  # Will use PolicyOnlyMCTS
        "High-Policy MCTS": None,  # Will use HighPolicyMCTS
    }

    # Create MCTS instances
    policy_mcts = PolicyOnlyMCTS(model, config, batch_size=8)
    high_policy_mcts = HighPolicyMCTS(model, config, batch_size=8)

    positions = setup_test_positions()

    results = {name: {"good": 0, "ok": 0, "bad": 0, "time": 0} for name in variants}

    print("=" * 80)
    for i, (name, board, expected) in enumerate(positions):
        print(f"\n{i+1}. {name}")
        print(f"   Expected: {expected}")
        print("-" * 60)

        # Test each variant
        for var_name in variants:
            start = time.time()

            if var_name == "Direct Policy":
                move, prob = get_direct_policy_move(model, board, config)
            elif var_name == "Policy-Only MCTS":
                policy = policy_mcts.search(board, verbose=False)
                action_idx = np.argmax(policy)
                move = (-1, -1) if action_idx == 361 else (action_idx // 19, action_idx % 19)
                prob = policy[action_idx]
            else:  # High-Policy MCTS
                policy = high_policy_mcts.search(board, verbose=False)
                action_idx = np.argmax(policy)
                move = (-1, -1) if action_idx == 361 else (action_idx // 19, action_idx % 19)
                prob = policy[action_idx]

            elapsed = time.time() - start
            results[var_name]["time"] += elapsed

            move_str = format_move(move)
            quality, score = evaluate_move_quality(move, expected, board)

            if quality == "GOOD" or quality == "TACTICAL":
                results[var_name]["good"] += 1
                symbol = "✓"
            elif quality == "BAD":
                results[var_name]["bad"] += 1
                symbol = "✗"
            else:
                results[var_name]["ok"] += 1
                symbol = "~"

            print(f"   {var_name:20s}: {move_str:5s} ({prob:5.1%}) [{quality:8s}] {symbol}  ({elapsed:.2f}s)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Variant':<25s} {'Good':>6s} {'OK':>6s} {'Bad':>6s} {'Score':>8s} {'Time':>8s}")
    print("-" * 60)

    for var_name in variants:
        r = results[var_name]
        score = r["good"] * 2 + r["ok"] * 1 + r["bad"] * 0
        print(f"{var_name:<25s} {r['good']:>6d} {r['ok']:>6d} {r['bad']:>6d} {score:>8d} {r['time']:>7.1f}s")

    # Winner
    scores = {name: results[name]["good"] * 2 + results[name]["ok"] for name in variants}
    winner = max(scores, key=scores.get)
    print(f"\n🏆 WINNER: {winner} (score: {scores[winner]})")


if __name__ == "__main__":
    main()
