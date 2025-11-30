#!/usr/bin/env python3
"""Supervised training on professional games (AlphaGo style)."""
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from config import Config, DEFAULT, QUICK
from model import GoNet, create_model, save_checkpoint, load_checkpoint
from sgf_parser import load_sgf_dataset


class ProGameDataset(Dataset):
    """Dataset of professional game positions."""

    def __init__(self, states: np.ndarray, moves: np.ndarray, board_size: int):
        self.states = states
        self.moves = moves
        self.board_size = board_size

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        move = self.moves[idx]

        # Convert move to action index
        action_idx = move[0] * self.board_size + move[1]

        return torch.FloatTensor(state), torch.LongTensor([action_idx])


def train_step(model, optimizer, states, actions, device, grad_clip: float = 1.0):
    """Single supervised training step with gradient clipping."""
    model.train()

    states = states.to(device)
    actions = actions.to(device)

    # Forward
    log_policies, values = model(states)

    # Policy loss: cross-entropy with pro moves
    policy_loss = F.nll_loss(log_policies, actions.squeeze())

    # No value loss in supervised training (we don't have game outcomes)
    total_loss = policy_loss

    # Backward with gradient clipping for stability
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'total_loss': total_loss.item()
    }


def evaluate_accuracy(model, dataloader, device):
    """Evaluate policy accuracy on dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for states, actions in dataloader:
            states = states.to(device)
            actions = actions.to(device)

            log_policies, _ = model(states)
            predictions = log_policies.argmax(dim=1)

            correct += (predictions == actions.squeeze()).sum().item()
            total += len(actions)

    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Supervised training on pro games')
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--max-games', type=int, default=10000, help='Max games to load')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--quick', action='store_true', help='Use quick config for testing')
    parser.add_argument('--data-dir', type=str, default='data/games')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # Config
    config = QUICK if args.quick else DEFAULT
    config.board_size = args.board_size
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {config.device}")
    print(f"Board size: {config.board_size}")
    print(f"Network: {config.num_blocks} blocks, {config.num_filters} filters")

    # Load dataset
    print(f"\nLoading pro games from {args.data_dir}...")
    states, moves = load_sgf_dataset(args.data_dir, config.board_size, args.max_games)

    if len(states) == 0:
        print("No games found!")
        return

    print(f"Loaded {len(states):,} positions")

    # Split train/val
    val_split = int(len(states) * 0.9)
    train_states, train_moves = states[:val_split], moves[:val_split]
    val_states, val_moves = states[val_split:], moves[val_split:]

    train_dataset = ProGameDataset(train_states, train_moves, config.board_size)
    val_dataset = ProGameDataset(val_states, val_moves, config.board_size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_dataset):,} positions")
    print(f"Val: {len(val_dataset):,} positions")

    # Model
    start_step = 0
    if args.resume:
        model, start_step = load_checkpoint(args.resume, config)
        config = model.config  # Use checkpoint's config
        print(f"Resumed from {args.resume} at step {start_step}")
    else:
        model = create_model(config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Logging
    writer = SummaryWriter('logs')

    global_step = start_step
    best_accuracy = 0.0

    try:
        for epoch in range(args.epochs):
            print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

            # Training
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (states, actions) in enumerate(train_loader):
                losses = train_step(model, optimizer, states, actions, config.device)

                epoch_loss += losses['total_loss']
                num_batches += 1
                global_step += 1

                if batch_idx % 100 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}: loss={losses['total_loss']:.4f}")

                # Clear CUDA cache periodically to prevent fragmentation
                if batch_idx > 0 and batch_idx % 500 == 0 and config.device == 'cuda':
                    torch.cuda.empty_cache()

                # Mid-epoch checkpoint every 1000 batches
                if batch_idx > 0 and batch_idx % 1000 == 0:
                    if config.device == 'cuda':
                        torch.cuda.synchronize()  # Ensure all ops complete before saving
                    save_checkpoint(model, optimizer, global_step, f'checkpoints/supervised_epoch_{epoch + 1}_batch_{batch_idx}.pt')
                    print(f"  [Checkpoint saved: epoch {epoch + 1}, batch {batch_idx}]")

                writer.add_scalar('Loss/train', losses['total_loss'], global_step)

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

            # Validation
            val_accuracy = evaluate_accuracy(model, val_loader, config.device)
            print(f"Validation accuracy: {val_accuracy:.1%}")
            writer.add_scalar('Accuracy/val', val_accuracy, epoch + 1)

            # Save checkpoint
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                save_checkpoint(model, optimizer, global_step, 'checkpoints/supervised_best.pt')
                print(f"New best accuracy: {val_accuracy:.1%}")

            save_checkpoint(model, optimizer, global_step, f'checkpoints/supervised_epoch_{epoch + 1}.pt')

    except (RuntimeError, KeyboardInterrupt) as e:
        print(f"\n[!] Training interrupted: {e}")
        print("[!] Saving emergency checkpoint...")
        save_checkpoint(model, optimizer, global_step, 'checkpoints/supervised_emergency.pt')
        print("[!] Saved to checkpoints/supervised_emergency.pt")
        raise

    # Final save
    save_checkpoint(model, optimizer, global_step, 'checkpoints/supervised_final.pt')
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.1%}")
    print(f"Saved to checkpoints/supervised_best.pt")
    print(f"\nUse this for self-play:")
    print(f"  python train.py --resume checkpoints/supervised_best.pt --iterations 100")


if __name__ == '__main__':
    main()
