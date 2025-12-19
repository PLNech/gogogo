#!/usr/bin/env python3
"""Main training loop."""
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


def train_step(model, optimizer, states, policies, values, device):
    """Single training step."""
    model.train()

    states = torch.FloatTensor(states).to(device)
    target_policies = torch.FloatTensor(policies).to(device)
    target_values = torch.FloatTensor(values).unsqueeze(1).to(device)

    # Forward
    pred_policies, pred_values = model(states)

    # Losses
    policy_loss = -torch.mean(torch.sum(target_policies * pred_policies, dim=1))
    value_loss = F.mse_loss(pred_values, target_values)
    total_loss = policy_loss + value_loss

    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'total_loss': total_loss.item()
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
    args = parser.parse_args()

    # Config
    config = QUICK if args.quick else DEFAULT
    config.board_size = args.board_size
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {config.device}")
    print(f"Board size: {config.board_size}")
    print(f"Network: {config.num_blocks} blocks, {config.num_filters} filters")

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
                    losses = train_step(model, optimizer, states, policies, values, config.device)

                    global_step += 1
                    tracker.training_step(step, losses, len(replay_buffer))

                    if step % 100 == 0:
                        print(f"  Step {step}: loss={losses['total_loss']:.4f} (p={losses['policy_loss']:.4f}, v={losses['value_loss']:.4f})")

                    writer.add_scalar('Loss/policy', losses['policy_loss'], global_step)
                    writer.add_scalar('Loss/value', losses['value_loss'], global_step)
                    writer.add_scalar('Loss/total', losses['total_loss'], global_step)

                    # Checkpoint
                    if global_step % config.checkpoint_interval == 0:
                        path = f'checkpoints/model_{global_step}.pt'
                        save_checkpoint(model, optimizer, global_step, path)
                        print(f"  Saved checkpoint: {path}")

            # Evaluation
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

        # Final save
        save_checkpoint(model, optimizer, global_step, 'checkpoints/final.pt')
        print(f"\nTraining complete! Final model saved to checkpoints/final.pt")


if __name__ == '__main__':
    main()
