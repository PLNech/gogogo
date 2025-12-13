#!/usr/bin/env python3
"""Generate tactical training positions for Atari Go.

Creates positions where the correct move is obvious:
- Capture scenarios (opponent in atari)
- Escape scenarios (self in atari)
- Simple ladders

These provide clear training signal for tactical learning.
"""
import numpy as np
import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
import random

from board import Board


@dataclass
class TacticalPosition:
    """A position with clear best move(s)."""
    board: np.ndarray
    correct_moves: List[Tuple[int, int]]
    to_play: int  # 1=black, -1=white
    category: str  # capture, escape, ladder


def generate_capture_position(size: int = 9) -> TacticalPosition:
    """Generate a position where one player can capture."""
    board = np.zeros((size, size), dtype=np.int8)

    # Random center position for the stone to capture
    center_r = random.randint(2, size - 3)
    center_c = random.randint(2, size - 3)

    # Place opponent stone
    board[center_r, center_c] = -1  # White stone to capture

    # Surround it, leaving one liberty
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    random.shuffle(directions)

    # Place 3 surrounding stones, leave 1 liberty
    for i, (dr, dc) in enumerate(directions[:3]):
        nr, nc = center_r + dr, center_c + dc
        board[nr, nc] = 1  # Black surrounding

    # The liberty is the correct move
    dr, dc = directions[3]
    correct_move = (center_r + dr, center_c + dc)

    return TacticalPosition(
        board=board,
        correct_moves=[correct_move],
        to_play=1,
        category="capture"
    )


def generate_escape_position(size: int = 9) -> TacticalPosition:
    """Generate a position where one player must escape atari."""
    board = np.zeros((size, size), dtype=np.int8)

    # Random center position
    center_r = random.randint(2, size - 3)
    center_c = random.randint(2, size - 3)

    # Place our stone (to escape)
    board[center_r, center_c] = 1  # Black stone in atari

    # Surround with opponent, leaving one liberty
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    random.shuffle(directions)

    # Place 3 opponent stones
    for i, (dr, dc) in enumerate(directions[:3]):
        nr, nc = center_r + dr, center_c + dc
        board[nr, nc] = -1  # White surrounding

    # The liberty is where we must extend
    dr, dc = directions[3]
    correct_move = (center_r + dr, center_c + dc)

    return TacticalPosition(
        board=board,
        correct_moves=[correct_move],
        to_play=1,
        category="escape"
    )


def generate_edge_capture(size: int = 9) -> TacticalPosition:
    """Generate edge capture position."""
    board = np.zeros((size, size), dtype=np.int8)

    # Pick random edge (not corner)
    edge = random.choice(['top', 'bottom', 'left', 'right'])

    if edge == 'top':
        r, c = 0, random.randint(2, size - 3)
        board[r, c] = -1  # White to capture
        board[r, c-1] = 1
        board[r, c+1] = 1
        correct = (r+1, c)
    elif edge == 'bottom':
        r, c = size-1, random.randint(2, size - 3)
        board[r, c] = -1
        board[r, c-1] = 1
        board[r, c+1] = 1
        correct = (r-1, c)
    elif edge == 'left':
        r, c = random.randint(2, size - 3), 0
        board[r, c] = -1
        board[r-1, c] = 1
        board[r+1, c] = 1
        correct = (r, c+1)
    else:  # right
        r, c = random.randint(2, size - 3), size-1
        board[r, c] = -1
        board[r-1, c] = 1
        board[r+1, c] = 1
        correct = (r, c-1)

    return TacticalPosition(
        board=board,
        correct_moves=[correct],
        to_play=1,
        category="capture"
    )


def generate_corner_capture(size: int = 9) -> TacticalPosition:
    """Generate corner capture position."""
    board = np.zeros((size, size), dtype=np.int8)

    corner = random.choice([(0, 0), (0, size-1), (size-1, 0), (size-1, size-1)])
    r, c = corner

    board[r, c] = -1  # White in corner

    # Place one black stone adjacent
    if r == 0 and c == 0:
        if random.random() < 0.5:
            board[0, 1] = 1
            correct = (1, 0)
        else:
            board[1, 0] = 1
            correct = (0, 1)
    elif r == 0 and c == size-1:
        if random.random() < 0.5:
            board[0, size-2] = 1
            correct = (1, size-1)
        else:
            board[1, size-1] = 1
            correct = (0, size-2)
    elif r == size-1 and c == 0:
        if random.random() < 0.5:
            board[size-1, 1] = 1
            correct = (size-2, 0)
        else:
            board[size-2, 0] = 1
            correct = (size-1, 1)
    else:  # bottom-right
        if random.random() < 0.5:
            board[size-1, size-2] = 1
            correct = (size-2, size-1)
        else:
            board[size-2, size-1] = 1
            correct = (size-1, size-2)

    return TacticalPosition(
        board=board,
        correct_moves=[correct],
        to_play=1,
        category="capture"
    )


def generate_two_stone_capture(size: int = 9) -> TacticalPosition:
    """Generate position to capture two connected stones."""
    board = np.zeros((size, size), dtype=np.int8)

    # Random position for two horizontal stones
    r = random.randint(2, size - 3)
    c = random.randint(2, size - 4)

    # Two white stones horizontally
    board[r, c] = -1
    board[r, c+1] = -1

    # Surround leaving one liberty
    # Top and bottom of both stones
    board[r-1, c] = 1
    board[r-1, c+1] = 1
    board[r+1, c] = 1
    board[r+1, c+1] = 1

    # One side
    side = random.choice(['left', 'right'])
    if side == 'left':
        board[r, c-1] = 1
        correct = (r, c+2)
    else:
        board[r, c+2] = 1
        correct = (r, c-1)

    return TacticalPosition(
        board=board,
        correct_moves=[correct],
        to_play=1,
        category="capture"
    )


def generate_ladder_position(size: int = 9) -> TacticalPosition:
    """Generate a ladder position where Black can capture running White.

    Classic ladder: White stone tries to escape, but Black keeps
    putting it in atari. The correct move is to play the ladder
    capturing move (diagonal chase).
    """
    board = np.zeros((size, size), dtype=np.int8)

    # Start position: White stone with 2 liberties, Black can start ladder
    # Place in area with room to run (need 3-4 steps of ladder space)
    # Direction: running toward bottom-right corner (most common)

    direction = random.choice(['br', 'bl', 'tr', 'tl'])  # bottom-right, etc.

    if direction == 'br':
        # White at (r,c), Black at (r,c-1) and (r-1,c)
        # White's liberties: (r,c+1) and (r+1,c)
        # Correct ladder move: (r+1,c) to start the chase
        r = random.randint(1, size - 5)
        c = random.randint(1, size - 5)
        board[r, c] = -1       # White stone being laddered
        board[r, c-1] = 1      # Black
        board[r-1, c] = 1      # Black
        correct = (r+1, c)     # Start ladder chase

    elif direction == 'bl':
        r = random.randint(1, size - 5)
        c = random.randint(4, size - 2)
        board[r, c] = -1
        board[r, c+1] = 1
        board[r-1, c] = 1
        correct = (r+1, c)

    elif direction == 'tr':
        r = random.randint(4, size - 2)
        c = random.randint(1, size - 5)
        board[r, c] = -1
        board[r, c-1] = 1
        board[r+1, c] = 1
        correct = (r-1, c)

    else:  # tl
        r = random.randint(4, size - 2)
        c = random.randint(4, size - 2)
        board[r, c] = -1
        board[r, c+1] = 1
        board[r+1, c] = 1
        correct = (r-1, c)

    return TacticalPosition(
        board=board,
        correct_moves=[correct],
        to_play=1,
        category="ladder"
    )


def generate_ladder_continuation(size: int = 9) -> TacticalPosition:
    """Generate ladder mid-sequence - White ran one step, continue chase."""
    board = np.zeros((size, size), dtype=np.int8)

    # After first ladder move, White extended, Black must continue
    r = random.randint(2, size - 5)
    c = random.randint(2, size - 5)

    # Ladder running down-right:
    # Initial: White at (r,c), Black at (r,c-1), (r-1,c)
    # White escaped to (r+1,c+1), Black played (r+1,c)
    # Now Black plays (r+1,c+2) or (r+2,c+1) depending on White's move

    variant = random.choice(['diag1', 'diag2'])

    if variant == 'diag1':
        # Two-step ladder
        board[r, c] = -1       # Original White (will be captured if ladder works)
        board[r+1, c+1] = -1   # White escape attempt
        board[r, c-1] = 1      # Black original
        board[r-1, c] = 1      # Black original
        board[r+1, c] = 1      # Black ladder move 1
        correct = (r+1, c+2)   # Continue ladder
    else:
        # Alternative shape
        board[r, c] = -1
        board[r+1, c+1] = -1
        board[r, c-1] = 1
        board[r-1, c] = 1
        board[r, c+1] = 1
        correct = (r+2, c+1)

    return TacticalPosition(
        board=board,
        correct_moves=[correct],
        to_play=1,
        category="ladder"
    )


def generate_net_position(size: int = 9) -> TacticalPosition:
    """Generate a net (geta) capture position.

    Net: Instead of chasing in a ladder, play a loose move that
    still captures because the stone can't escape in any direction.
    """
    board = np.zeros((size, size), dtype=np.int8)

    # White stone that can be netted
    r = random.randint(2, size - 4)
    c = random.randint(2, size - 4)

    board[r, c] = -1  # White to be netted

    # Black stones forming partial net
    variant = random.choice(['knight', 'diagonal'])

    if variant == 'knight':
        # Knight's move net
        board[r-1, c-1] = 1
        board[r-1, c+1] = 1
        board[r+1, c-1] = 1
        correct = (r+1, c+1)  # Complete the net with knight's move
    else:
        # Diagonal net
        board[r-1, c] = 1
        board[r, c-1] = 1
        board[r+1, c+1] = 1
        correct = (r+1, c-1)  # Tighten net

    return TacticalPosition(
        board=board,
        correct_moves=[correct],
        to_play=1,
        category="net"
    )


def generate_snapback(size: int = 9) -> TacticalPosition:
    """Generate a snapback position.

    Snapback: Sacrifice stones that opponent must capture,
    then recapture a larger group.
    """
    board = np.zeros((size, size), dtype=np.int8)

    r = random.randint(2, size - 4)
    c = random.randint(2, size - 4)

    # Classic snapback shape
    # White stones in atari that can capture Black, but Black snaps back
    board[r, c] = -1      # White
    board[r, c+1] = -1    # White - this group in atari
    board[r+1, c] = -1    # White

    board[r-1, c] = 1     # Black
    board[r-1, c+1] = 1   # Black
    board[r, c+2] = 1     # Black
    board[r+1, c+1] = 1   # Black - this one is in atari
    board[r+2, c] = 1     # Black

    # Correct move: let them capture, then snapback
    correct = (r+1, c-1)  # Forces the snapback

    return TacticalPosition(
        board=board,
        correct_moves=[correct],
        to_play=1,
        category="snapback"
    )


def generate_connect_position(size: int = 9) -> TacticalPosition:
    """Generate position where connecting two groups is critical."""
    board = np.zeros((size, size), dtype=np.int8)

    r = random.randint(2, size - 4)
    c = random.randint(2, size - 4)

    # Two Black groups that need to connect
    board[r, c] = 1        # Group 1
    board[r, c+2] = 1      # Group 2

    # White threatening to cut
    board[r-1, c+1] = -1
    board[r+1, c+1] = -1

    correct = (r, c+1)     # Connect!

    return TacticalPosition(
        board=board,
        correct_moves=[correct],
        to_play=1,
        category="connect"
    )


def position_to_tensor(pos: TacticalPosition, size: int = 9) -> np.ndarray:
    """Convert position to basic input tensor (17 planes)."""
    # Simplified: just 4 planes (current black, current white, ones, zeros)
    # The model expects 17 planes but we'll use basic representation
    tensor = np.zeros((17, size, size), dtype=np.float32)

    # Current player's stones (plane 0)
    tensor[0] = (pos.board == pos.to_play).astype(np.float32)
    # Opponent's stones (plane 1)
    tensor[1] = (pos.board == -pos.to_play).astype(np.float32)
    # All ones (plane 16 typically)
    tensor[16] = np.ones((size, size), dtype=np.float32)

    return tensor


def generate_tactical_dataset(num_positions: int = 10000, size: int = 9) -> Dict:
    """Generate a dataset of tactical positions.

    Returns dict with:
        - tensors: (N, 17, size, size) input tensors
        - policies: (N, size*size+1) one-hot policy targets
        - values: (N,) value targets (+1 for winning position)
    """
    generators = [
        # Basic captures (weight: 5)
        generate_capture_position,
        generate_capture_position,
        generate_escape_position,
        generate_escape_position,
        generate_edge_capture,
        generate_corner_capture,
        generate_two_stone_capture,
        # Ladders (weight: 4) - crucial tactical pattern
        generate_ladder_position,
        generate_ladder_position,
        generate_ladder_continuation,
        generate_ladder_continuation,
        # Advanced tactics (weight: 3)
        generate_net_position,
        generate_snapback,
        generate_connect_position,
    ]

    tensors = []
    policies = []
    values = []

    for _ in range(num_positions):
        gen = random.choice(generators)
        pos = gen(size)

        tensor = position_to_tensor(pos, size)
        tensors.append(tensor)

        # Policy: one-hot on correct move(s)
        policy = np.zeros(size * size + 1, dtype=np.float32)
        for (r, c) in pos.correct_moves:
            policy[r * size + c] = 1.0
        if policy.sum() > 0:
            policy /= policy.sum()
        else:
            # No correct moves - uniform
            policy[:] = 1.0 / len(policy)
        policies.append(policy)

        # Value: based on position type
        # Capture/ladder/net/snapback = winning (+1), escape/connect = neutral (0)
        if pos.category in ("capture", "ladder", "net", "snapback"):
            values.append(1.0)  # Winning position
        else:
            values.append(0.0)  # Escape/connect is neutral if we succeed

    return {
        'tensors': np.stack(tensors),
        'policies': np.stack(policies),
        'values': np.array(values, dtype=np.float32)
    }


def train_on_tactical(model, config, num_positions: int = 50000,
                      epochs: int = 10, batch_size: int = 128):
    """Train model on tactical positions."""
    import torch.nn.functional as F

    print(f"Generating {num_positions} tactical positions...")
    data = generate_tactical_dataset(num_positions, config.board_size)

    print(f"Training for {epochs} epochs...")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n = len(data['tensors'])
    indices = np.arange(n)

    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0
        total_correct = 0
        num_batches = 0

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]

            x = torch.FloatTensor(data['tensors'][batch_idx]).to(config.device)
            policy_target = torch.FloatTensor(data['policies'][batch_idx]).to(config.device)
            value_target = torch.FloatTensor(data['values'][batch_idx]).to(config.device)

            model.train()
            outputs = model(x)
            log_policy = outputs[0]
            value = outputs[1].squeeze()

            # Policy loss
            policy_loss = -torch.sum(policy_target * log_policy, dim=1).mean()

            # Value loss
            value_loss = F.mse_loss(value, value_target)

            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            pred = torch.argmax(torch.exp(log_policy), dim=1)
            target = torch.argmax(policy_target, dim=1)
            total_correct += (pred == target).sum().item()
            num_batches += 1

        acc = total_correct / n * 100
        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.3f}, accuracy={acc:.1f}%")

    return model


if __name__ == '__main__':
    import argparse
    from model import GoNet, load_checkpoint
    from config import Config

    parser = argparse.ArgumentParser(description='Train on tactical positions')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--positions', type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output', type=str, default='checkpoints/tactical_trained.pt')
    args = parser.parse_args()

    config = Config()
    config.board_size = 9
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.checkpoint:
        model, step = load_checkpoint(args.checkpoint, config)
        print(f"Loaded {args.checkpoint}")
    else:
        model = GoNet(config).to(config.device)
        print("Created new model")

    model = train_on_tactical(model, config, args.positions, args.epochs)

    # Save
    torch.save({
        'step': 0,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, args.output)
    print(f"Saved to {args.output}")
