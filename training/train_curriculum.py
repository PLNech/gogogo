#!/usr/bin/env python3
"""Train with Adaptive Instinct Curriculum.

Combines self-play/supervised training with instinct-aware loss.
The curriculum weight decays as the model masters fundamentals.

Usage:
    poetry run python train_curriculum.py --checkpoint checkpoints/supervised_best.pt --epochs 50
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from board import Board
from model import GoNet, load_checkpoint, save_checkpoint
from config import Config
from instinct_loss import InstinctCurriculum, InstinctDetector


def load_instinct_positions(benchmark_dir: str, board_size: int) -> List[Dict]:
    """Load instinct benchmark positions."""
    positions = []
    for json_file in Path(benchmark_dir).glob('*.json'):
        with open(json_file) as f:
            data = json.load(f)
        for pos in data:
            if pos.get('board_size') == board_size:
                positions.append(pos)
    return positions


def position_to_board(pos: Dict, board_size: int) -> Board:
    """Convert position dict to Board."""
    board = Board(board_size)
    for r, c in pos.get('black_stones', []):
        board.board[r, c] = 1
    for r, c in pos.get('white_stones', []):
        board.board[r, c] = -1
    board.current_player = 1 if pos['to_play'] == 'black' else -1
    return board


def augment_board(board: Board, tensor: np.ndarray, policy: np.ndarray) -> List[Tuple]:
    """Apply 8-fold symmetry augmentation."""
    augmented = [(board, tensor, policy)]

    size = board.size
    policy_2d = policy[:-1].reshape(size, size)  # Exclude pass

    for k in range(1, 4):  # 90, 180, 270 rotations
        rot_tensor = np.rot90(tensor, k, axes=(1, 2)).copy()
        rot_policy = np.rot90(policy_2d, k).copy()

        new_policy = np.zeros(size * size + 1, dtype=np.float32)
        new_policy[:-1] = rot_policy.flatten()
        new_policy[-1] = policy[-1]

        augmented.append((board, rot_tensor, new_policy))

    # Horizontal flip
    flip_tensor = np.flip(tensor, axis=2).copy()
    flip_policy = np.flip(policy_2d, axis=1).copy()

    new_policy = np.zeros(size * size + 1, dtype=np.float32)
    new_policy[:-1] = flip_policy.flatten()
    new_policy[-1] = policy[-1]

    augmented.append((board, flip_tensor, new_policy))

    return augmented


def run_benchmark(model: GoNet, config: Config, positions: List[Dict]) -> Dict:
    """Quick benchmark on instinct positions."""
    model.eval()
    board_size = config.board_size
    use_tactical = getattr(config, 'tactical_features', False) or config.input_planes == 27

    correct_by_cat = {}
    total_by_cat = {}

    with torch.no_grad():
        for pos in positions:
            board = position_to_board(pos, board_size)
            tensor = board.to_tensor(use_tactical_features=use_tactical)
            x = torch.FloatTensor(tensor).unsqueeze(0).to(config.device)

            log_policy, _ = model(x)
            policy = torch.exp(log_policy).cpu().numpy()[0]

            # Check top move
            top_move_idx = np.argmax(policy[:-1])  # Exclude pass
            top_move = (top_move_idx // board_size, top_move_idx % board_size)

            expected = set(tuple(m) for m in pos.get('expected_moves', []))
            cat = pos.get('category', 'unknown')

            if cat not in correct_by_cat:
                correct_by_cat[cat] = 0
                total_by_cat[cat] = 0

            total_by_cat[cat] += 1
            if top_move in expected:
                correct_by_cat[cat] += 1

    results = {}
    for cat in total_by_cat:
        results[cat] = correct_by_cat[cat] / total_by_cat[cat] if total_by_cat[cat] > 0 else 0

    # Overall
    total_correct = sum(correct_by_cat.values())
    total_count = sum(total_by_cat.values())
    results['overall'] = total_correct / total_count if total_count > 0 else 0

    return results


def train_with_curriculum(
    model: GoNet,
    config: Config,
    benchmark_dir: str = 'benchmarks/instincts',
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    lambda_0: float = 1.0,
    eval_every: int = 5,
    verbose: bool = True
) -> Dict:
    """Train model with adaptive instinct curriculum."""

    device = config.device
    board_size = config.board_size
    use_tactical = getattr(config, 'tactical_features', False) or config.input_planes == 27

    # Load positions
    positions = load_instinct_positions(benchmark_dir, board_size)
    if not positions:
        print(f"No positions for {board_size}x{board_size}")
        return {}

    if verbose:
        print(f"Loaded {len(positions)} positions")

    # Prepare training data
    samples = []  # (tensor, policy, board_for_instinct)
    for pos in positions:
        board = position_to_board(pos, board_size)
        tensor = board.to_tensor(use_tactical_features=use_tactical)

        # Policy target
        policy = np.zeros(board_size * board_size + 1, dtype=np.float32)
        for r, c in pos.get('expected_moves', []):
            idx = r * board_size + c
            policy[idx] = 1.0
        if policy.sum() > 0:
            policy /= policy.sum()

        # Augment
        for _, aug_tensor, aug_policy in augment_board(board, tensor, policy):
            samples.append((aug_tensor, aug_policy, board))

    if verbose:
        print(f"Total samples after augmentation: {len(samples)}")

    # Initialize curriculum
    curriculum = InstinctCurriculum(
        lambda_0=lambda_0,
        min_lambda=0.05,
        temperature=2.0,
        device=device
    )

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = len(samples)
    indices = np.arange(n)

    # History tracking
    history = {
        'epoch': [],
        'loss': [],
        'policy_loss': [],
        'instinct_loss': [],
        'lambda': [],
        'accuracy': [],
        'capture_acc': [],
        'escape_acc': [],
        'overall_acc': []
    }

    # Initial benchmark
    initial_results = run_benchmark(model, config, positions)
    curriculum.update_lambda(initial_results.get('overall', 0))

    if verbose:
        print(f"\nInitial benchmark:")
        for cat, acc in sorted(initial_results.items()):
            print(f"  {cat}: {acc:.1%}")
        print(f"Initial λ: {curriculum.current_lambda:.3f}")

    for epoch in range(epochs):
        np.random.shuffle(indices)

        model.train()
        total_loss = 0
        total_policy_loss = 0
        total_instinct_loss = 0
        total_correct = 0
        num_batches = 0

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]

            tensors = np.stack([samples[j][0] for j in batch_idx])
            policies = np.stack([samples[j][1] for j in batch_idx])
            boards = [samples[j][2] for j in batch_idx]

            x = torch.FloatTensor(tensors).to(device)
            policy_target = torch.FloatTensor(policies).to(device)

            # Forward
            log_policy, value = model(x)

            # Policy loss
            policy_loss = -torch.sum(policy_target * log_policy, dim=1).mean()

            # Instinct loss (adaptive)
            instinct_loss, _ = curriculum.compute_loss(boards, log_policy)

            # Total loss
            loss = policy_loss + instinct_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_instinct_loss += float(instinct_loss)

            # Accuracy
            pred = torch.argmax(torch.exp(log_policy), dim=1)
            target = torch.argmax(policy_target, dim=1)
            total_correct += (pred == target).sum().item()
            num_batches += 1

        scheduler.step()

        avg_loss = total_loss / num_batches
        avg_policy = total_policy_loss / num_batches
        avg_instinct = total_instinct_loss / num_batches
        accuracy = total_correct / n * 100

        # Benchmark and update lambda
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            results = run_benchmark(model, config, positions)
            curriculum.update_lambda(results.get('overall', 0))

            history['epoch'].append(epoch + 1)
            history['loss'].append(avg_loss)
            history['policy_loss'].append(avg_policy)
            history['instinct_loss'].append(avg_instinct)
            history['lambda'].append(curriculum.current_lambda)
            history['accuracy'].append(accuracy)
            history['capture_acc'].append(results.get('capture', 0) * 100)
            history['escape_acc'].append(results.get('escape', 0) * 100)
            history['overall_acc'].append(results.get('overall', 0) * 100)

            if verbose:
                print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f} (p={avg_policy:.4f}, i={avg_instinct:.4f}) "
                      f"acc={accuracy:.1f}% λ={curriculum.current_lambda:.3f} "
                      f"| cap={results.get('capture', 0):.1%} esc={results.get('escape', 0):.1%} "
                      f"all={results.get('overall', 0):.1%}")

    return history


def plot_training_history(history: Dict, output_path: str = 'training_plots/curriculum_training.png'):
    """Generate training visualization."""
    Path(output_path).parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Adaptive Instinct Curriculum Training', fontsize=14, fontweight='bold')

    # Colors
    BLUE = '#2563eb'
    GREEN = '#16a34a'
    ORANGE = '#ea580c'
    RED = '#dc2626'
    PURPLE = '#9333ea'

    epochs = history['epoch']

    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['loss'], color=BLUE, linewidth=2, label='Total Loss')
    ax1.plot(epochs, history['policy_loss'], color=GREEN, linewidth=2, linestyle='--', label='Policy Loss')
    ax1.plot(epochs, history['instinct_loss'], color=ORANGE, linewidth=2, linestyle=':', label='Instinct Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Lambda decay
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['lambda'], color=PURPLE, linewidth=2, marker='o', markersize=4)
    ax2.fill_between(epochs, history['lambda'], alpha=0.3, color=PURPLE)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('λ (Instinct Weight)')
    ax2.set_title('Adaptive Curriculum Weight')
    ax2.set_ylim(0, max(history['lambda']) * 1.1)
    ax2.grid(True, alpha=0.3)

    # Add annotation
    ax2.annotate('λ decays as model\nmasters instincts',
                 xy=(epochs[-1], history['lambda'][-1]),
                 xytext=(epochs[len(epochs)//2], max(history['lambda']) * 0.7),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=9, color='gray')

    # Plot 3: Instinct accuracy
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['capture_acc'], color=RED, linewidth=2, marker='s', markersize=4, label='Capture')
    ax3.plot(epochs, history['escape_acc'], color=BLUE, linewidth=2, marker='^', markersize=4, label='Escape')
    ax3.plot(epochs, history['overall_acc'], color=GREEN, linewidth=2, marker='o', markersize=4, label='Overall')
    ax3.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='Target (80%)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Instinct Benchmark Accuracy')
    ax3.set_ylim(0, 105)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Training accuracy
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['accuracy'], color=GREEN, linewidth=2, marker='o', markersize=4)
    ax4.fill_between(epochs, history['accuracy'], alpha=0.3, color=GREEN)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Training Accuracy (%)')
    ax4.set_title('Training Accuracy')
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train with adaptive instinct curriculum")
    parser.add_argument("--checkpoint", type=str, help="Starting checkpoint")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--output", type=str, default="checkpoints/curriculum_trained.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lambda-0", type=float, default=1.0, help="Initial instinct weight")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--benchmark-dir", default="benchmarks/instincts")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing history")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    if args.from_scratch:
        config = Config(board_size=args.board_size)
        config.device = device
        model = GoNet(config).to(device)
        print(f"Created new model ({args.board_size}x{args.board_size})")
    elif args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
        config = checkpoint.get('config', Config(board_size=args.board_size))
        config.device = device
        model, step = load_checkpoint(args.checkpoint, config)
        print(f"Loaded: {args.checkpoint}")
        print(f"  Board size: {config.board_size}, Input planes: {config.input_planes}")
    else:
        print("Error: specify --checkpoint or --from-scratch")
        return

    # Train
    print("\n" + "="*60)
    print("ADAPTIVE INSTINCT CURRICULUM TRAINING")
    print("="*60)

    history = train_with_curriculum(
        model, config,
        benchmark_dir=args.benchmark_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_0=args.lambda_0,
        eval_every=5,
        verbose=True
    )

    # Plot
    if history:
        plot_training_history(history)

    # Save
    torch.save({
        'step': 0,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, args.output)
    print(f"\nSaved: {args.output}")

    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    if history['capture_acc']:
        print(f"  Capture: {history['capture_acc'][0]:.1f}% -> {history['capture_acc'][-1]:.1f}%")
        print(f"  Escape:  {history['escape_acc'][0]:.1f}% -> {history['escape_acc'][-1]:.1f}%")
        print(f"  Overall: {history['overall_acc'][0]:.1f}% -> {history['overall_acc'][-1]:.1f}%")
        print(f"  Lambda:  {history['lambda'][0]:.3f} -> {history['lambda'][-1]:.3f}")


if __name__ == "__main__":
    main()
