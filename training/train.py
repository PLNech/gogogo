#!/usr/bin/env python3
"""Main training loop with adaptive instinct curriculum."""
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from config import Config, DEFAULT, QUICK
from model import GoNet, create_model, save_checkpoint, load_checkpoint
from selfplay import generate_games, ReplayBuffer
from mcts import MCTS
from board import Board
from training_state import TrainingTracker
from instinct_loss import InstinctCurriculum


def train_step(model, optimizer, states, policies, values, device,
                curriculum: InstinctCurriculum = None):
    """Single training step with optional instinct curriculum.

    Loss = L_policy + L_value + λ(t) × L_instinct

    Args:
        model: Neural network model
        optimizer: Optimizer
        states: Batch of board states (numpy array)
        policies: Target policies (numpy array)
        values: Target values (numpy array)
        device: Torch device
        curriculum: Optional instinct curriculum for auxiliary loss

    Returns:
        Dictionary of loss metrics
    """
    model.train()

    states_tensor = torch.FloatTensor(states).to(device)
    target_policies = torch.FloatTensor(policies).to(device)
    target_values = torch.FloatTensor(values).unsqueeze(1).to(device)

    # Forward
    pred_policies, pred_values = model(states_tensor)

    # Losses
    policy_loss = -torch.mean(torch.sum(target_policies * pred_policies, dim=1))
    value_loss = F.mse_loss(pred_values, target_values)
    total_loss = policy_loss + value_loss

    # Instinct auxiliary loss
    instinct_loss = torch.tensor(0.0, device=device)
    instinct_metrics = {}

    if curriculum is not None and curriculum.current_lambda > 0:
        # Reconstruct boards from state tensors
        boards = [Board.from_tensor(s) for s in states]

        # Compute instinct loss
        instinct_loss, instinct_metrics = curriculum.compute_loss(
            boards, pred_policies, reduction='mean'
        )
        total_loss = total_loss + instinct_loss

    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    metrics = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'total_loss': total_loss.item(),
    }

    if curriculum is not None:
        metrics['instinct_loss'] = instinct_metrics.get('instinct_loss', 0.0)
        metrics['instinct_weighted'] = instinct_metrics.get('instinct_weighted_loss', 0.0)
        metrics['instinct_lambda'] = curriculum.current_lambda
        metrics['instinct_count'] = instinct_metrics.get('instinct_count', 0)

    return metrics


def evaluate_instinct_accuracy(model, config: Config, device: str) -> dict:
    """Evaluate model's instinct accuracy on benchmark positions.

    Returns dict with per-category accuracy and overall accuracy.
    """
    from benchmark import BenchmarkRunner, load_benchmarks, aggregate_results

    benchmark_dir = 'benchmarks/instincts'
    positions = load_benchmarks(benchmark_dir, config.board_size)

    if not positions:
        # No benchmark positions available - return default
        return {
            'overall': 0.0,
            'by_category': {},
            'total_tested': 0,
        }

    runner = BenchmarkRunner(model, config)

    # Group by category (instinct)
    by_instinct = {}
    for pos in positions:
        result = runner.evaluate_position(pos)
        result['name'] = pos.name
        result['category'] = pos.category

        if pos.category not in by_instinct:
            by_instinct[pos.category] = []
        by_instinct[pos.category].append(result)

    # Aggregate by category
    results_by_cat = {}
    total_count = 0
    total_correct = 0

    for instinct, instinct_results in by_instinct.items():
        agg = aggregate_results(instinct_results)
        results_by_cat[instinct] = agg['top1_accuracy']
        total_count += agg['count']
        total_correct += int(agg['top1_accuracy'] * agg['count'])

    overall = total_correct / total_count if total_count > 0 else 0.0

    return {
        'overall': overall,
        'by_category': results_by_cat,
        'total_tested': total_count,
    }


def evaluate(model1, model2, config: Config, num_games: int) -> float:
    """Evaluate model1 vs model2, return model1 win rate."""
    wins = 0
    draws = 0

    for game_idx in range(num_games):
        board = Board(config.board_size)
        mcts1 = MCTS(model1, config)
        mcts2 = MCTS(model2, config)

        # Alternate who plays first
        if game_idx % 2 == 0:
            players = {1: (mcts1, model1), -1: (mcts2, model2)}
            model1_color = 1
        else:
            players = {1: (mcts2, model2), -1: (mcts1, model1)}
            model1_color = -1

        move_count = 0
        max_moves = config.board_size ** 2 * 2

        while not board.is_game_over() and move_count < max_moves:
            mcts, _ = players[board.current_player]
            action = mcts.select_action(board, temperature=0)

            if action == (-1, -1):
                board.pass_move()
            else:
                board.play(action[0], action[1])
            move_count += 1

        score = board.score()
        if score > 0:
            winner = 1
        elif score < 0:
            winner = -1
        else:
            winner = 0
            draws += 1

        if winner == model1_color:
            wins += 1

    return wins / num_games


def main():
    parser = argparse.ArgumentParser(description='Train Go AI')
    parser.add_argument('--board-size', type=int, default=9)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--quick', action='store_true', help='Use quick config for testing')
    parser.add_argument('--instincts', action='store_true',
                        help='Enable instinct curriculum (auxiliary loss for tactical patterns)')
    parser.add_argument('--instinct-lambda', type=float, default=1.0,
                        help='Initial instinct loss weight (decays with accuracy)')
    args = parser.parse_args()

    # Config
    config = QUICK if args.quick else DEFAULT
    config.board_size = args.board_size
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {config.device}")
    print(f"Board size: {config.board_size}")
    print(f"Network: {config.num_blocks} blocks, {config.num_filters} filters")
    if args.instincts:
        print(f"Instinct curriculum: ENABLED (λ₀={args.instinct_lambda})")

    # Directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Model
    if args.resume:
        model, start_step = load_checkpoint(args.resume, config)
        # Update config from loaded model
        config = model.config
        print(f"Resumed from {args.resume} at step {start_step}")
        print(f"Using checkpoint config: {config.num_blocks} blocks, {config.num_filters} filters")
    else:
        model = create_model(config)
        start_step = 0
        print("Created new model")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(config.replay_buffer_size)

    # Instinct curriculum (optional)
    curriculum = None
    if args.instincts:
        curriculum = InstinctCurriculum(
            lambda_0=args.instinct_lambda,
            min_lambda=0.1,
            temperature=2.0,
            device=config.device
        )
        print(f"Instinct curriculum initialized")

    # Logging
    writer = SummaryWriter('logs')

    # Best model for evaluation - use same config as loaded model
    best_model = GoNet(config).to(config.device)
    best_model.load_state_dict(model.state_dict())

    global_step = start_step

    with TrainingTracker(args.iterations, config) as tracker:
        for iteration in range(args.iterations):
            tracker.start_iteration(iteration + 1)
            print(f"\n=== Iteration {iteration + 1}/{args.iterations} ===")

            # Generate self-play games
            tracker.start_selfplay(config.games_per_iter)
            print(f"Generating {config.games_per_iter} games...")
            model.eval()
            samples, records = generate_games(
                model, config, config.games_per_iter,
                game_callback=lambda i: tracker.game_complete(i)
            )
            replay_buffer.add(samples)
            print(f"Buffer size: {len(replay_buffer)}")

            # Training
            if len(replay_buffer) >= config.min_replay_size:
                tracker.start_training(config.train_steps_per_iter)
                print(f"Training for {config.train_steps_per_iter} steps...")
                model.train()

                for step in range(config.train_steps_per_iter):
                    states, policies, values = replay_buffer.sample(config.batch_size)
                    losses = train_step(
                        model, optimizer, states, policies, values,
                        config.device, curriculum=curriculum
                    )

                    global_step += 1
                    tracker.training_step(step, losses, len(replay_buffer))

                    if step % 100 == 0:
                        loss_str = f"loss={losses['total_loss']:.4f} (p={losses['policy_loss']:.4f}, v={losses['value_loss']:.4f}"
                        if curriculum:
                            loss_str += f", i={losses.get('instinct_loss', 0):.4f}×{curriculum.current_lambda:.2f}"
                        loss_str += ")"
                        print(f"  Step {step}: {loss_str}")

                    writer.add_scalar('Loss/policy', losses['policy_loss'], global_step)
                    writer.add_scalar('Loss/value', losses['value_loss'], global_step)
                    writer.add_scalar('Loss/total', losses['total_loss'], global_step)

                    if curriculum:
                        writer.add_scalar('Instinct/loss', losses.get('instinct_loss', 0), global_step)
                        writer.add_scalar('Instinct/lambda', curriculum.current_lambda, global_step)
                        writer.add_scalar('Instinct/count', losses.get('instinct_count', 0), global_step)

                    # Checkpoint
                    if global_step % config.checkpoint_interval == 0:
                        path = f'checkpoints/model_{global_step}.pt'
                        save_checkpoint(model, optimizer, global_step, path)
                        print(f"  Saved checkpoint: {path}")

            # Evaluation (every 5 iterations)
            if iteration % 5 == 0 and iteration > 0:
                tracker.start_eval()
                print("Evaluating vs best model...")
                model.eval()
                win_rate = evaluate(model, best_model, config, config.eval_games)
                print(f"Win rate vs best: {win_rate:.1%}")
                writer.add_scalar('Eval/win_rate', win_rate, global_step)

                is_best = win_rate > config.win_threshold
                tracker.eval_complete(win_rate, is_best)

                if is_best:
                    print("New best model!")
                    best_model.load_state_dict(model.state_dict())
                    save_checkpoint(model, optimizer, global_step, 'checkpoints/best.pt')

                # Instinct benchmark evaluation (updates curriculum lambda)
                if curriculum:
                    print("Evaluating instinct accuracy...")
                    try:
                        instinct_results = evaluate_instinct_accuracy(model, config, config.device)
                        accuracy = instinct_results['overall']
                        curriculum.update_lambda(accuracy)

                        print(f"Instinct accuracy: {accuracy:.1%} (λ → {curriculum.current_lambda:.2f})")
                        writer.add_scalar('Instinct/accuracy', accuracy, global_step)

                        # Log per-category accuracy
                        for cat, cat_acc in instinct_results['by_category'].items():
                            writer.add_scalar(f'Instinct/{cat}', cat_acc, global_step)
                    except Exception as e:
                        print(f"Instinct benchmark failed: {e}")

        # Final save
        save_checkpoint(model, optimizer, global_step, 'checkpoints/final.pt')
        print(f"\nTraining complete! Final model saved to checkpoints/final.pt")


if __name__ == '__main__':
    main()
