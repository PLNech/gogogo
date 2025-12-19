#!/usr/bin/env python3
"""Train model on instinct benchmark positions.

Direct supervised learning on capture, escape, and other instinct positions
to build fundamental Go skills before self-play.

Usage:
    poetry run python train_instincts.py --checkpoint checkpoints/supervised_best.pt --epochs 50
    poetry run python train_instincts.py --from-scratch --epochs 100
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from board import Board
from model import GoNet, load_checkpoint, save_checkpoint
from config import Config


@dataclass
class InstinctSample:
    """A training sample from instinct benchmark."""
    tensor: np.ndarray
    policy: np.ndarray
    value: float
    category: str


def load_instinct_positions(benchmark_dir: str, board_size: int) -> List[Dict]:
    """Load all instinct benchmark positions for a board size."""
    positions = []
    benchmark_path = Path(benchmark_dir)

    for json_file in benchmark_path.glob('*.json'):
        with open(json_file) as f:
            data = json.load(f)

        for pos in data:
            if pos.get('board_size') == board_size:
                positions.append(pos)

    return positions


def position_to_sample(pos: Dict, board_size: int, use_tactical: bool = False) -> InstinctSample:
    """Convert a benchmark position to a training sample."""
    # Create board
    board = Board(board_size)

    for r, c in pos.get('black_stones', []):
        board.board[r, c] = 1
    for r, c in pos.get('white_stones', []):
        board.board[r, c] = -1

    board.current_player = 1 if pos['to_play'] == 'black' else -1

    # Get tensor
    tensor = board.to_tensor(use_tactical_features=use_tactical)

    # Create policy target (one-hot on expected moves)
    policy = np.zeros(board_size * board_size + 1, dtype=np.float32)
    for r, c in pos.get('expected_moves', []):
        idx = r * board_size + c
        policy[idx] = 1.0

    if policy.sum() > 0:
        policy /= policy.sum()
    else:
        # Fallback to uniform if no expected moves
        policy[:] = 1.0 / len(policy)

    # Value: positive for capture (we gain), neutral for escape/connect
    category = pos.get('category', '')
    if category in ('capture', 'atari', 'cut'):
        value = 0.5  # Positive: we're winning
    elif category in ('escape', 'defend'):
        value = -0.2  # Slightly negative: we're under pressure but can survive
    else:
        value = 0.0  # Neutral

    return InstinctSample(
        tensor=tensor,
        policy=policy,
        value=value,
        category=category
    )


def augment_sample(sample: InstinctSample, board_size: int) -> List[InstinctSample]:
    """Apply data augmentation (rotations and reflections)."""
    samples = [sample]

    # The tensor shape is (C, H, W)
    tensor = sample.tensor
    policy = sample.policy[:-1].reshape(board_size, board_size)  # Exclude pass

    # 8 symmetries: 4 rotations x 2 reflections
    for k in range(1, 4):  # 90, 180, 270 degree rotations
        rot_tensor = np.rot90(tensor, k, axes=(1, 2)).copy()
        rot_policy = np.rot90(policy, k).copy()

        new_policy = np.zeros(board_size * board_size + 1, dtype=np.float32)
        new_policy[:-1] = rot_policy.flatten()
        new_policy[-1] = sample.policy[-1]  # Pass probability

        samples.append(InstinctSample(
            tensor=rot_tensor,
            policy=new_policy,
            value=sample.value,
            category=sample.category
        ))

    # Horizontal flip
    flip_tensor = np.flip(tensor, axis=2).copy()
    flip_policy = np.flip(policy, axis=1).copy()

    new_policy = np.zeros(board_size * board_size + 1, dtype=np.float32)
    new_policy[:-1] = flip_policy.flatten()
    new_policy[-1] = sample.policy[-1]

    samples.append(InstinctSample(
        tensor=flip_tensor,
        policy=new_policy,
        value=sample.value,
        category=sample.category
    ))

    return samples


def train_on_instincts(
    model: GoNet,
    config: Config,
    benchmark_dir: str = 'benchmarks/instincts',
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    augment: bool = True,
    category_weights: Dict[str, float] = None,
    verbose: bool = True
) -> Dict:
    """Train model on instinct benchmark positions.

    Args:
        model: Model to train
        config: Configuration
        benchmark_dir: Directory with benchmark JSON files
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        augment: Whether to use data augmentation
        category_weights: Optional per-category loss weighting
        verbose: Print progress

    Returns:
        Training metrics
    """
    device = config.device
    board_size = config.board_size
    use_tactical = getattr(config, 'tactical_features', False) or config.input_planes == 27

    # Load positions
    positions = load_instinct_positions(benchmark_dir, board_size)
    if not positions:
        print(f"No positions found for {board_size}x{board_size} in {benchmark_dir}")
        return {}

    if verbose:
        print(f"Loaded {len(positions)} positions for {board_size}x{board_size}")

    # Convert to samples
    samples = []
    for pos in positions:
        sample = position_to_sample(pos, board_size, use_tactical)
        if augment:
            samples.extend(augment_sample(sample, board_size))
        else:
            samples.append(sample)

    if verbose:
        print(f"Total samples after augmentation: {len(samples)}")

        # Count by category
        by_cat = {}
        for s in samples:
            by_cat[s.category] = by_cat.get(s.category, 0) + 1
        for cat, count in sorted(by_cat.items()):
            print(f"  {cat}: {count}")

    # Default category weights (emphasize capture/escape)
    if category_weights is None:
        category_weights = {
            'capture': 2.0,
            'escape': 2.0,
            'atari': 1.5,
            'defend': 1.5,
            'connect': 1.0,
            'cut': 1.0,
            'extend': 1.0,
            'block': 1.0,
        }

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = len(samples)
    indices = np.arange(n)

    history = {'loss': [], 'policy_loss': [], 'value_loss': [], 'accuracy': []}

    for epoch in range(epochs):
        np.random.shuffle(indices)

        model.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_correct = 0
        num_batches = 0

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]

            # Prepare batch
            tensors = np.stack([samples[j].tensor for j in batch_idx])
            policies = np.stack([samples[j].policy for j in batch_idx])
            values = np.array([samples[j].value for j in batch_idx], dtype=np.float32)
            weights = np.array([category_weights.get(samples[j].category, 1.0) for j in batch_idx], dtype=np.float32)

            x = torch.FloatTensor(tensors).to(device)
            policy_target = torch.FloatTensor(policies).to(device)
            value_target = torch.FloatTensor(values).to(device)
            weight = torch.FloatTensor(weights).to(device)

            # Forward
            log_policy, value_pred = model(x)
            value_pred = value_pred.squeeze()

            # Weighted policy loss (cross-entropy)
            policy_loss = -torch.sum(policy_target * log_policy, dim=1)
            policy_loss = (policy_loss * weight).mean()

            # Value loss
            value_loss = F.mse_loss(value_pred, value_target)

            # Total loss
            loss = policy_loss + 0.5 * value_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

            # Accuracy
            pred = torch.argmax(torch.exp(log_policy), dim=1)
            target = torch.argmax(policy_target, dim=1)
            total_correct += (pred == target).sum().item()
            num_batches += 1

        scheduler.step()

        avg_loss = total_loss / num_batches
        avg_policy = total_policy_loss / num_batches
        avg_value = total_value_loss / num_batches
        accuracy = total_correct / n * 100

        history['loss'].append(avg_loss)
        history['policy_loss'].append(avg_policy)
        history['value_loss'].append(avg_value)
        history['accuracy'].append(accuracy)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f} (p={avg_policy:.4f}, v={avg_value:.4f}) acc={accuracy:.1f}%")

    return history


def run_quick_benchmark(model: GoNet, config: Config, benchmark_dir: str = 'benchmarks/instincts') -> Dict:
    """Run a quick benchmark to check instinct accuracy."""
    from benchmark import BenchmarkRunner, load_benchmarks, aggregate_results

    model.eval()
    board_size = config.board_size

    positions = load_benchmarks(benchmark_dir, board_size)
    if not positions:
        return {}

    # Use tactical features if model requires them
    use_tactical = getattr(config, 'tactical_features', False) or config.input_planes == 27
    runner = BenchmarkRunner(model, config, use_tactical=use_tactical)

    by_instinct = {}
    for pos in positions:
        result = runner.evaluate_position(pos)
        cat = pos.category
        if cat not in by_instinct:
            by_instinct[cat] = []
        by_instinct[cat].append(result)

    results = {}
    for instinct, instinct_results in by_instinct.items():
        agg = aggregate_results(instinct_results)
        results[instinct] = {
            'accuracy': agg['top1_accuracy'],
            'count': agg['count']
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Train on instinct benchmarks")
    parser.add_argument("--checkpoint", type=str, help="Starting checkpoint")
    parser.add_argument("--from-scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--output", type=str, default="checkpoints/instinct_trained.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--benchmark-dir", default="benchmarks/instincts")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load or create model
    if args.from_scratch:
        config = Config(board_size=args.board_size)
        config.device = device
        model = GoNet(config).to(device)
        print(f"Created new model ({args.board_size}x{args.board_size})")
    elif args.checkpoint:
        # Load config from checkpoint
        checkpoint = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
        config = checkpoint.get('config', Config(board_size=args.board_size))
        config.device = device

        # Override board size if specified
        if args.board_size != 9:  # 9 is default, so only override if explicitly set
            config.board_size = args.board_size

        model, step = load_checkpoint(args.checkpoint, config)
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"  Board size: {config.board_size}x{config.board_size}")
        print(f"  Input planes: {config.input_planes}")
        print(f"  Tactical features: {getattr(config, 'tactical_features', False)}")
    else:
        print("Error: specify --checkpoint or --from-scratch")
        return

    # Pre-training benchmark
    print("\n" + "="*60)
    print("PRE-TRAINING BENCHMARK")
    print("="*60)
    pre_results = run_quick_benchmark(model, config, args.benchmark_dir)
    for inst, data in sorted(pre_results.items()):
        print(f"  {inst:12}: {data['accuracy']:5.1%} ({data['count']})")

    # Train
    print("\n" + "="*60)
    print("TRAINING ON INSTINCTS")
    print("="*60)

    history = train_on_instincts(
        model, config,
        benchmark_dir=args.benchmark_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        augment=not args.no_augment,
        verbose=True
    )

    # Post-training benchmark
    print("\n" + "="*60)
    print("POST-TRAINING BENCHMARK")
    print("="*60)
    post_results = run_quick_benchmark(model, config, args.benchmark_dir)
    for inst, data in sorted(post_results.items()):
        pre_acc = pre_results.get(inst, {}).get('accuracy', 0)
        delta = data['accuracy'] - pre_acc
        arrow = "+" if delta > 0 else ""
        print(f"  {inst:12}: {data['accuracy']:5.1%} ({arrow}{delta*100:.1f}%)")

    # Save
    save_checkpoint(model, 0, config, args.output)
    print(f"\nSaved to: {args.output}")

    # Summary
    print("\n" + "="*60)
    print("IMPROVEMENT SUMMARY")
    print("="*60)

    total_pre = sum(d['accuracy'] * d['count'] for d in pre_results.values())
    total_post = sum(d['accuracy'] * d['count'] for d in post_results.values())
    total_count = sum(d['count'] for d in post_results.values())

    pre_overall = total_pre / total_count if total_count else 0
    post_overall = total_post / total_count if total_count else 0

    print(f"  Overall: {pre_overall:.1%} -> {post_overall:.1%} ({(post_overall-pre_overall)*100:+.1f}%)")


if __name__ == "__main__":
    main()
