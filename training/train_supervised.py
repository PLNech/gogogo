#!/usr/bin/env python3
"""Supervised training on professional games (AlphaGo style)."""
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler  # Mixed precision (torch 2.0+)
import numpy as np
from config import Config, DEFAULT, QUICK
from model import GoNet, create_model, save_checkpoint, load_checkpoint
from sgf_parser import load_sgf_dataset
from visualize import TrainingVisualizer, plot_network_architecture

# Enable cudnn autotuning for faster convolutions
torch.backends.cudnn.benchmark = True


class CurriculumSampler(Sampler):
    """
    Curriculum learning sampler for tactical positions.

    Based on KataGo insight: sparse tactical features need more exposure.
    Schedule (from ARCH.md):
    - Epochs 1-3: 100% tactical positions
    - Epochs 4-6: 70% tactical, 30% normal
    - Epochs 7+: 50% tactical, 50% normal
    """

    def __init__(self, tactical_mask: np.ndarray, epoch_size: int = None):
        """
        Args:
            tactical_mask: Boolean array where True = position has tactical activity
            epoch_size: Number of samples per epoch (default: len(tactical_mask))
        """
        self.tactical_indices = np.where(tactical_mask)[0]
        self.normal_indices = np.where(~tactical_mask)[0]
        self.epoch_size = epoch_size or len(tactical_mask)
        self.tactical_ratio = 1.0
        self._epoch = 0

        print(f"CurriculumSampler: {len(self.tactical_indices):,} tactical, "
              f"{len(self.normal_indices):,} normal positions")

    def set_epoch(self, epoch: int):
        """Update tactical ratio based on epoch."""
        self._epoch = epoch
        if epoch <= 3:
            self.tactical_ratio = 1.0
        elif epoch <= 6:
            self.tactical_ratio = 0.7
        else:
            self.tactical_ratio = 0.5
        print(f"  Curriculum: epoch {epoch}, tactical_ratio={self.tactical_ratio:.0%}")

    def __iter__(self):
        n_tactical = int(self.epoch_size * self.tactical_ratio)
        n_normal = self.epoch_size - n_tactical

        # Sample with replacement to handle imbalanced sizes
        tactical = np.random.choice(self.tactical_indices, n_tactical, replace=True)
        normal = np.random.choice(self.normal_indices, n_normal, replace=True) if n_normal > 0 else np.array([], dtype=np.int64)

        indices = np.concatenate([tactical, normal])
        np.random.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self):
        return self.epoch_size


class ProGameDataset(Dataset):
    """Dataset of professional game positions with optional augmentation."""

    def __init__(self, states: np.ndarray, moves: np.ndarray, board_size: int,
                 values: np.ndarray = None, ownership: np.ndarray = None,
                 opponent_moves: np.ndarray = None, augment: bool = False):
        # Pre-convert to tensors (faster than converting each time)
        self.states = torch.from_numpy(states).float()
        self.moves = moves
        self.board_size = board_size
        self.values = torch.from_numpy(values).float() if values is not None else None
        self.ownership = torch.from_numpy(ownership).float() if ownership is not None else None
        self.opponent_moves = opponent_moves  # Keep as numpy for augmentation
        self.augment = augment

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        move = self.moves[idx]
        value = self.values[idx] if self.values is not None else None
        own = self.ownership[idx] if self.ownership is not None else None
        opp_move = self.opponent_moves[idx] if self.opponent_moves is not None else None

        # Data augmentation: random rotation/reflection (8 symmetries)
        if self.augment:
            k = torch.randint(0, 4, (1,)).item()  # 0-3 rotations
            flip = torch.randint(0, 2, (1,)).item()  # 0-1 flip

            if k > 0:
                state = torch.rot90(state, k, dims=(1, 2))
                if own is not None:
                    own = torch.rot90(own, k, dims=(0, 1))
                # Rotate move coordinates
                r, c = move[0], move[1]
                for _ in range(k):
                    r, c = c, self.board_size - 1 - r
                move = np.array([r, c])
                # Rotate opponent move too
                if opp_move is not None:
                    or_, oc = opp_move[0], opp_move[1]
                    for _ in range(k):
                        or_, oc = oc, self.board_size - 1 - or_
                    opp_move = np.array([or_, oc])

            if flip:
                state = torch.flip(state, dims=(2,))  # Horizontal flip
                if own is not None:
                    own = torch.flip(own, dims=(1,))  # Horizontal flip
                move = np.array([move[0], self.board_size - 1 - move[1]])
                if opp_move is not None:
                    opp_move = np.array([opp_move[0], self.board_size - 1 - opp_move[1]])

        action_idx = move[0] * self.board_size + move[1]

        # Build result tuple based on what data is available
        # Order: state, action, [value], [ownership], [opponent_action]
        result = [state, torch.tensor(action_idx, dtype=torch.long)]
        if value is not None:
            result.append(value)
        if own is not None:
            result.append(own)
        if opp_move is not None:
            opp_action_idx = opp_move[0] * self.board_size + opp_move[1]
            result.append(torch.tensor(opp_action_idx, dtype=torch.long))
        return tuple(result) if len(result) > 2 else tuple(result)


def train_step(model, optimizer, states, actions, device, scaler=None, grad_clip: float = 1.0,
               value_targets=None, value_weight: float = 1.0,
               ownership_targets=None, board_size: int = 19,
               opponent_actions=None, opponent_weight: float = 0.15):
    """Single supervised training step with mixed precision and gradient clipping.

    Args:
        value_targets: Optional value targets for value head training
        value_weight: Weight for value loss (default 1.0, KataGo uses 1.5)
        ownership_targets: Optional (batch, H, W) ownership maps in [-1, 1]
        board_size: Board size for ownership loss weighting (KataGo uses 1.5/b²)
        opponent_actions: Optional opponent move targets (KataGo: 1.30x speedup)
        opponent_weight: Weight for opponent policy loss (KataGo uses 0.15)
    """
    model.train()

    states = states.to(device, non_blocking=True)
    actions = actions.to(device, non_blocking=True)
    if value_targets is not None:
        value_targets = value_targets.to(device, non_blocking=True)
    if ownership_targets is not None:
        ownership_targets = ownership_targets.to(device, non_blocking=True)
    if opponent_actions is not None:
        opponent_actions = opponent_actions.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

    # Determine what outputs we need
    return_ownership = ownership_targets is not None
    return_opponent = opponent_actions is not None

    # Mixed precision forward pass
    if scaler is not None:
        with autocast(device_type='cuda'):
            # Get model outputs based on what we need
            outputs = model(states, return_ownership=return_ownership,
                           return_opponent_policy=return_opponent)
            if return_ownership or return_opponent:
                log_policies, values = outputs[0], outputs[1]
                idx = 2
                ownership_logits = outputs[idx] if return_ownership else None
                if return_ownership:
                    idx += 1
                opp_log_policies = outputs[idx] if return_opponent else None
            else:
                log_policies, values = outputs

            policy_loss = F.nll_loss(log_policies, actions.squeeze())

            # Value loss (MSE) if targets provided
            if value_targets is not None:
                value_loss = F.mse_loss(values.squeeze(), value_targets)
            else:
                value_loss = torch.tensor(0.0, device=device)

            # Ownership loss (BCE) if targets provided
            # KataGo: weight = 1.5/b², multiply by b² to get per-position loss
            if ownership_targets is not None:
                ownership_binary = (ownership_targets + 1) / 2
                ownership_loss = F.binary_cross_entropy_with_logits(
                    ownership_logits.squeeze(1), ownership_binary, reduction='mean'
                )
                w_ownership = 1.5 / (board_size ** 2)
                ownership_loss = w_ownership * ownership_loss * (board_size ** 2)
            else:
                ownership_loss = torch.tensor(0.0, device=device)

            # Opponent policy loss (NLL) if targets provided
            # KataGo: weight = 0.15, helps learn opponent's likely responses
            if opponent_actions is not None:
                opponent_loss = F.nll_loss(opp_log_policies, opponent_actions.squeeze())
            else:
                opponent_loss = torch.tensor(0.0, device=device)

            total_loss = (policy_loss + value_weight * value_loss +
                         ownership_loss + opponent_weight * opponent_loss)

        # Mixed precision backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        # Standard precision
        outputs = model(states, return_ownership=return_ownership,
                       return_opponent_policy=return_opponent)
        if return_ownership or return_opponent:
            log_policies, values = outputs[0], outputs[1]
            idx = 2
            ownership_logits = outputs[idx] if return_ownership else None
            if return_ownership:
                idx += 1
            opp_log_policies = outputs[idx] if return_opponent else None
        else:
            log_policies, values = outputs

        policy_loss = F.nll_loss(log_policies, actions.squeeze())

        if value_targets is not None:
            value_loss = F.mse_loss(values.squeeze(), value_targets)
        else:
            value_loss = torch.tensor(0.0, device=device)

        if ownership_targets is not None:
            ownership_binary = (ownership_targets + 1) / 2
            ownership_loss = F.binary_cross_entropy_with_logits(
                ownership_logits.squeeze(1), ownership_binary, reduction='mean'
            )
            w_ownership = 1.5 / (board_size ** 2)
            ownership_loss = w_ownership * ownership_loss * (board_size ** 2)
        else:
            ownership_loss = torch.tensor(0.0, device=device)

        if opponent_actions is not None:
            opponent_loss = F.nll_loss(opp_log_policies, opponent_actions.squeeze())
        else:
            opponent_loss = torch.tensor(0.0, device=device)

        total_loss = (policy_loss + value_weight * value_loss +
                     ownership_loss + opponent_weight * opponent_loss)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item() if value_targets is not None else 0.0,
        'ownership_loss': ownership_loss.item() if ownership_targets is not None else 0.0,
        'opponent_loss': opponent_loss.item() if opponent_actions is not None else 0.0,
        'total_loss': total_loss.item()
    }


def evaluate_accuracy(model, dataloader, device):
    """Evaluate policy accuracy on dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            # Handle both with and without value targets
            states, actions = batch[0], batch[1]
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
    parser.add_argument('--max-positions', type=int, default=300000,
                        help='Max positions to load (memory safety, default 300k ~5GB with tactical)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (default: 256, try 1024+ for faster training)')
    parser.add_argument('--tactical-features', action='store_true',
                        help='Enable neuro-symbolic tactical features (27 planes instead of 17)')
    parser.add_argument('--train-value', action='store_true',
                        help='Train value head using game results (requires RE[] in SGF)')
    parser.add_argument('--value-weight', type=float, default=1.0,
                        help='Weight for value loss (KataGo uses 1.5)')
    parser.add_argument('--num-blocks', type=int, default=None, help='ResNet blocks (default: 6)')
    parser.add_argument('--num-filters', type=int, default=None, help='Conv filters (default: 128)')
    parser.add_argument('--quick', action='store_true', help='Use quick config for testing')
    parser.add_argument('--data-dir', type=str, default='data/games')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--curriculum', action='store_true',
                        help='Enable curriculum learning (tactical positions first)')
    parser.add_argument('--ownership', action='store_true',
                        help='Train ownership head (KataGo: 1.65x speedup, 361x more signal)')
    parser.add_argument('--opponent-move', action='store_true',
                        help='Train opponent move prediction (KataGo: 1.30x speedup)')
    parser.add_argument('--opponent-weight', type=float, default=0.15,
                        help='Weight for opponent policy loss (KataGo uses 0.15)')
    args = parser.parse_args()

    # Config
    config = QUICK if args.quick else DEFAULT
    config.board_size = args.board_size
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Override batch size if specified (larger = faster training on RTX 4080)
    if args.batch_size:
        config.batch_size = args.batch_size

    # Override network architecture
    if args.num_blocks:
        config.num_blocks = args.num_blocks
    if args.num_filters:
        config.num_filters = args.num_filters

    # Tactical features (neuro-symbolic approach)
    config.tactical_features = args.tactical_features
    if config.tactical_features:
        config.input_planes = 27  # 17 basic + 10 tactical
    else:
        config.input_planes = 17

    print(f"Device: {config.device}")
    print(f"Board size: {config.board_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Input planes: {config.input_planes} ({'tactical' if config.tactical_features else 'basic'})")
    print(f"Network: {config.num_blocks} blocks, {config.num_filters} filters")
    if args.train_value:
        print(f"Value training: ENABLED (weight={args.value_weight})")
    if args.curriculum:
        print(f"Curriculum learning: ENABLED (requires tactical features)")
        if not config.tactical_features:
            print("WARNING: Curriculum requires --tactical-features, enabling automatically")
            config.tactical_features = True
            config.input_planes = 27
    if args.ownership:
        print(f"Ownership training: ENABLED (KataGo's highest-impact technique)")
    if args.opponent_move:
        print(f"Opponent move prediction: ENABLED (weight={args.opponent_weight})")

    # Load dataset
    print(f"\nLoading pro games from {args.data_dir}...")

    # Determine what data to load
    include_tactical_mask = args.curriculum and config.tactical_features
    tactical_mask = None
    ownership = None
    opponent_moves = None

    load_result = load_sgf_dataset(
        args.data_dir, config.board_size, args.max_games,
        tactical_features=config.tactical_features,
        max_positions=args.max_positions,
        include_value=args.train_value,
        include_tactical_mask=include_tactical_mask,
        include_ownership=args.ownership,
        include_opponent_move=args.opponent_move
    )

    # Unpack results based on what was requested
    # Order: states, moves, [values], [tactical_mask], [ownership], [opponent_moves]
    idx = 0
    states = load_result[idx]; idx += 1
    moves = load_result[idx]; idx += 1
    values = None
    if args.train_value:
        values = load_result[idx]; idx += 1
    if include_tactical_mask:
        tactical_mask = load_result[idx]; idx += 1
    if args.ownership:
        ownership = load_result[idx]; idx += 1
    if args.opponent_move:
        opponent_moves = load_result[idx]; idx += 1

    if len(states) == 0:
        print("No games found!")
        return

    print(f"Loaded {len(states):,} positions")

    # Split train/val
    val_split = int(len(states) * 0.9)
    train_states, train_moves = states[:val_split], moves[:val_split]
    val_states, val_moves = states[val_split:], moves[val_split:]

    if values is not None:
        train_values, val_values = values[:val_split], values[val_split:]
    else:
        train_values, val_values = None, None

    # Split ownership if using ownership training
    if ownership is not None:
        train_ownership, val_ownership = ownership[:val_split], ownership[val_split:]
    else:
        train_ownership, val_ownership = None, None

    # Split opponent moves if using opponent prediction
    if opponent_moves is not None:
        train_opponent, val_opponent = opponent_moves[:val_split], opponent_moves[val_split:]
    else:
        train_opponent, val_opponent = None, None

    # Split tactical mask if using curriculum
    train_tactical_mask = None
    if tactical_mask is not None:
        train_tactical_mask = tactical_mask[:val_split]

    train_dataset = ProGameDataset(train_states, train_moves, config.board_size,
                                   values=train_values, ownership=train_ownership,
                                   opponent_moves=train_opponent, augment=True)
    val_dataset = ProGameDataset(val_states, val_moves, config.board_size,
                                 values=val_values, ownership=val_ownership,
                                 opponent_moves=val_opponent, augment=False)

    # Optimize data loading for i9 (20 cores) + RTX 4080 (12GB)
    num_workers = min(8, os.cpu_count() or 4)  # Use 8 workers for parallel loading

    # Create curriculum sampler if enabled
    curriculum_sampler = None
    if args.curriculum and train_tactical_mask is not None:
        curriculum_sampler = CurriculumSampler(train_tactical_mask, epoch_size=len(train_dataset))
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=curriculum_sampler,  # Use curriculum sampler instead of shuffle
            num_workers=num_workers,
            pin_memory=(config.device == 'cuda'),
            persistent_workers=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(config.device == 'cuda'),
            persistent_workers=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(config.device == 'cuda'),
        persistent_workers=True
    )
    print(f"DataLoader: {num_workers} workers, pin_memory={config.device == 'cuda'}")

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

    # Mixed precision scaler
    scaler = GradScaler() if config.device == 'cuda' else None
    if scaler:
        print("Mixed precision training: ENABLED")

    # Learning rate scheduler (cosine annealing)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.learning_rate * 0.01
    )

    # Directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Logging
    writer = SummaryWriter('logs')

    # Visualization
    viz = TrainingVisualizer('training_plots')
    viz.set_config(config, {
        'max_games': args.max_games,
        'max_positions': args.max_positions,
        'total_positions': len(train_dataset) + len(val_dataset),
        'epochs': args.epochs,
    })

    # Save architecture diagram
    plot_network_architecture(config, 'training_plots/architecture.png')

    global_step = start_step
    best_accuracy = 0.0
    import time

    try:
        for epoch in range(args.epochs):
            print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
            epoch_start = time.time()

            # Update curriculum sampler for this epoch
            if curriculum_sampler is not None:
                curriculum_sampler.set_epoch(epoch + 1)

            # Training
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                # Unpack batch based on what data was requested
                # Order from ProGameDataset: (state, action, [value], [ownership])
                idx = 0
                states = batch[idx]; idx += 1
                actions = batch[idx]; idx += 1

                value_targets = None
                ownership_targets = None
                opponent_actions = None

                if args.train_value:
                    value_targets = batch[idx]; idx += 1
                if args.ownership:
                    ownership_targets = batch[idx]; idx += 1
                if args.opponent_move:
                    opponent_actions = batch[idx]; idx += 1

                losses = train_step(model, optimizer, states, actions, config.device,
                                   scaler=scaler, value_targets=value_targets,
                                   value_weight=args.value_weight,
                                   ownership_targets=ownership_targets,
                                   board_size=config.board_size,
                                   opponent_actions=opponent_actions,
                                   opponent_weight=args.opponent_weight)
                scheduler.step()

                epoch_loss += losses['total_loss']
                num_batches += 1
                global_step += 1

                # Log to visualizer
                lr = scheduler.get_last_lr()[0]
                viz.log_batch(losses['total_loss'], lr)

                if batch_idx % 100 == 0:
                    loss_str = f"loss={losses['total_loss']:.4f}"
                    if args.train_value or args.ownership or args.opponent_move:
                        loss_str += f" (p={losses['policy_loss']:.3f}"
                        if args.train_value:
                            loss_str += f" v={losses['value_loss']:.3f}"
                        if args.ownership:
                            loss_str += f" o={losses['ownership_loss']:.3f}"
                        if args.opponent_move:
                            loss_str += f" op={losses['opponent_loss']:.3f}"
                        loss_str += ")"
                    print(f"  Batch {batch_idx}/{len(train_loader)}: {loss_str} lr={lr:.2e}")

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
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f} ({epoch_time:.1f}s)")

            # Validation
            val_accuracy = evaluate_accuracy(model, val_loader, config.device)
            print(f"Validation accuracy: {val_accuracy:.1%}")
            writer.add_scalar('Accuracy/val', val_accuracy, epoch + 1)

            # Log to visualizer
            viz.log_epoch(epoch + 1, avg_loss, val_accuracy, epoch_time)

            # Save checkpoint
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                save_checkpoint(model, optimizer, global_step, 'checkpoints/supervised_best.pt')
                print(f"New best accuracy: {val_accuracy:.1%}")

            save_checkpoint(model, optimizer, global_step, f'checkpoints/supervised_epoch_{epoch + 1}.pt')

            # Generate plots every epoch
            viz.plot_training_curves(save=True)

    except (RuntimeError, KeyboardInterrupt) as e:
        print(f"\n[!] Training interrupted: {e}")
        print("[!] Saving emergency checkpoint...")
        save_checkpoint(model, optimizer, global_step, 'checkpoints/supervised_emergency.pt')
        print("[!] Saved to checkpoints/supervised_emergency.pt")
        # Generate plots even on interrupt
        print("[!] Generating final plots...")
        viz.generate_all_plots()
        raise

    # Final save
    save_checkpoint(model, optimizer, global_step, 'checkpoints/supervised_final.pt')
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.1%}")
    print(f"Saved to checkpoints/supervised_best.pt")

    # Generate all visualizations
    print("\nGenerating final visualizations...")
    viz.generate_all_plots()

    print(f"\nPlots saved to training_plots/")
    print(f"  - architecture.png")
    print(f"  - training_curves_*.png")
    print(f"  - loss_distribution_*.png")
    print(f"  - convergence_*.png")
    print(f"  - metrics_*.json")

    print(f"\nUse this for self-play:")
    print(f"  python train.py --resume checkpoints/supervised_best.pt --iterations 100")


if __name__ == '__main__':
    main()
