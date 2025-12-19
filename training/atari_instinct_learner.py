#!/usr/bin/env python3
"""Learn instinct priority weights through Atari Go self-play.

Philosophy:
    The proverbs say "extend from atari" and "connect against peep"
    But which instinct matters MORE in actual games?

    Self-play reveals truth. We track:
    - When each instinct fires
    - Whether following it led to victory
    - Evolve weights based on win contribution

Usage:
    poetry run python atari_instinct_learner.py --games 5000

Output:
    Learned priority weights for each of Sensei's 8 instincts.
"""
import numpy as np
import torch
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path

from board import Board
from atari_go import AtariGoBoard, AtariGoTrainer
from sensei_instincts import SenseiInstinctDetector, InstinctResult
from config import Config


@dataclass
class InstinctEvent:
    """Record of an instinct firing during a game."""
    instinct: str
    move_number: int
    detected_moves: List[Tuple[int, int]]
    actual_move: Tuple[int, int]
    followed: bool  # Did player follow the instinct?


@dataclass
class GameRecord:
    """Full record of instinct events in a game."""
    events: List[InstinctEvent] = field(default_factory=list)
    winner: int = 0  # 1=Black, -1=White, 0=Draw
    length: int = 0


class InstinctWeightLearner:
    """Learn optimal instinct weights through Atari Go self-play."""

    # Initial weights (from Sensei's/intuition)
    INITIAL_WEIGHTS = {
        'extend_from_atari': 3.0,
        'connect_vs_peep': 2.5,
        'block_the_thrust': 2.0,
        'hane_vs_tsuke': 1.5,
        'hane_at_head_of_two': 1.5,
        'stretch_from_kosumi': 1.2,
        'block_the_angle': 1.2,
        'stretch_from_bump': 1.0,
    }

    def __init__(self, board_size: int = 9):
        self.board_size = board_size
        self.detector = SenseiInstinctDetector()

        # Current weights (will be learned)
        self.weights = dict(self.INITIAL_WEIGHTS)

        # Statistics per instinct
        self.stats = {name: {
            'fired': 0,        # Times instinct detected
            'followed': 0,     # Times player followed it
            'followed_won': 0, # Times following led to win
            'followed_lost': 0,
            'ignored_won': 0,  # Times ignoring led to win
            'ignored_lost': 0,
        } for name in self.INITIAL_WEIGHTS}

        # Game records for analysis
        self.game_records: List[GameRecord] = []

    def detect_instincts(self, board: Board) -> List[InstinctResult]:
        """Detect all instincts in position."""
        return self.detector.detect_all(board)

    def record_move(self, board: Board, actual_move: Tuple[int, int],
                    record: GameRecord, move_number: int):
        """Record instinct events for a move."""
        instincts = self.detect_instincts(board)

        for inst in instincts:
            followed = actual_move in inst.moves
            event = InstinctEvent(
                instinct=inst.instinct,
                move_number=move_number,
                detected_moves=inst.moves,
                actual_move=actual_move,
                followed=followed
            )
            record.events.append(event)
            self.stats[inst.instinct]['fired'] += 1
            if followed:
                self.stats[inst.instinct]['followed'] += 1

    def finalize_game(self, record: GameRecord, winner: int):
        """Update statistics based on game outcome."""
        record.winner = winner

        for event in record.events:
            name = event.instinct
            # Determine if the move was by the winner
            # Move 0 = Black, Move 1 = White, etc.
            player = 1 if event.move_number % 2 == 0 else -1
            won = (player == winner)

            if event.followed:
                if won:
                    self.stats[name]['followed_won'] += 1
                else:
                    self.stats[name]['followed_lost'] += 1
            else:
                if won:
                    self.stats[name]['ignored_won'] += 1
                else:
                    self.stats[name]['ignored_lost'] += 1

        self.game_records.append(record)

    def compute_win_rates(self) -> Dict[str, Dict[str, float]]:
        """Compute win rates for following/ignoring each instinct."""
        results = {}

        for name, stat in self.stats.items():
            followed_total = stat['followed_won'] + stat['followed_lost']
            ignored_total = stat['ignored_won'] + stat['ignored_lost']

            follow_winrate = (stat['followed_won'] / followed_total
                              if followed_total > 0 else 0.5)
            ignore_winrate = (stat['ignored_won'] / ignored_total
                              if ignored_total > 0 else 0.5)

            results[name] = {
                'fired': stat['fired'],
                'followed': stat['followed'],
                'follow_rate': stat['followed'] / stat['fired'] if stat['fired'] > 0 else 0,
                'follow_winrate': follow_winrate,
                'ignore_winrate': ignore_winrate,
                'follow_advantage': follow_winrate - ignore_winrate,
            }

        return results

    def update_weights(self, learning_rate: float = 0.1) -> Dict[str, float]:
        """Update weights based on follow advantage.

        Instincts that lead to wins when followed get higher weight.
        """
        win_rates = self.compute_win_rates()

        for name in self.weights:
            if name not in win_rates:
                continue

            wr = win_rates[name]
            if wr['fired'] < 10:  # Not enough data
                continue

            # Follow advantage: how much better is following vs ignoring
            advantage = wr['follow_advantage']

            # Update weight: increase if following helps, decrease if not
            # Scale by follow rate (instincts that are rarely followed need bigger boost)
            follow_rate = wr['follow_rate']

            # If advantage > 0 and follow_rate < 0.5, we should increase weight more
            # (the instinct works but isn't being used enough)
            adjustment = advantage * (1 + (0.5 - follow_rate))

            self.weights[name] = max(0.1, self.weights[name] + learning_rate * adjustment)

        # Normalize to keep sum similar
        total = sum(self.weights.values())
        target_sum = sum(self.INITIAL_WEIGHTS.values())
        scale = target_sum / total if total > 0 else 1.0
        self.weights = {k: v * scale for k, v in self.weights.items()}

        return dict(self.weights)

    def get_summary(self) -> str:
        """Get formatted summary of learned weights."""
        win_rates = self.compute_win_rates()

        lines = [
            "=" * 70,
            "LEARNED INSTINCT WEIGHTS",
            "=" * 70,
            f"{'Instinct':<25} {'Weight':>8} {'Fired':>8} {'Follow%':>8} {'Adv':>8}",
            "-" * 70,
        ]

        # Sort by learned weight
        sorted_instincts = sorted(self.weights.items(), key=lambda x: -x[1])

        for name, weight in sorted_instincts:
            wr = win_rates.get(name, {})
            fired = wr.get('fired', 0)
            follow_rate = wr.get('follow_rate', 0) * 100
            advantage = wr.get('follow_advantage', 0) * 100

            lines.append(
                f"{name:<25} {weight:>8.2f} {fired:>8} {follow_rate:>7.1f}% {advantage:>+7.1f}%"
            )

        lines.extend([
            "-" * 70,
            f"Total games: {len(self.game_records)}",
            "=" * 70,
        ])

        return "\n".join(lines)

    def save(self, path: str):
        """Save learned weights and statistics."""
        data = {
            'weights': self.weights,
            'initial_weights': self.INITIAL_WEIGHTS,
            'stats': self.stats,
            'num_games': len(self.game_records),
            'win_rates': self.compute_win_rates(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {path}")


class AtariInstinctTrainer(AtariGoTrainer):
    """Atari Go trainer that tracks instinct usage."""

    def __init__(self, config: Config, checkpoint_path: Optional[str] = None,
                 num_parallel: int = 16):
        super().__init__(config, checkpoint_path, num_parallel)
        self.instinct_learner = InstinctWeightLearner(config.board_size)

    def play_game_with_tracking(self, temperature: float = 1.0) -> GameRecord:
        """Play one game, tracking instinct usage."""
        board = AtariGoBoard(self.config.board_size)
        record = GameRecord()

        move_count = 0
        max_moves = self.config.board_size ** 2 * 2

        while not board.is_game_over() and move_count < max_moves:
            # Get policy from model
            self.model.eval()
            with torch.no_grad():
                tensor = torch.FloatTensor(board.to_tensor()).unsqueeze(0)
                tensor = tensor.to(self.config.device)
                log_policy, value = self.model(tensor)[:2]
                policy = torch.exp(log_policy).cpu().numpy()[0]

            # Select move
            legal_mask = np.zeros(board.size ** 2 + 1)
            for r, c in board.get_legal_moves():
                legal_mask[r * board.size + c] = 1
            legal_mask[board.size ** 2] = 1

            masked = policy * legal_mask
            if masked.sum() < 1e-8:
                masked = legal_mask
            masked = masked / masked.sum()

            if temperature == 0:
                action_idx = np.argmax(masked)
            else:
                log_p = np.log(masked + 1e-10) / temperature
                log_p = log_p - log_p.max()
                p = np.exp(log_p)
                p = p / p.sum()
                action_idx = np.random.choice(len(p), p=p)

            if action_idx == board.size ** 2:
                move = (-1, -1)
            else:
                move = (action_idx // board.size, action_idx % board.size)

            # Record instinct events BEFORE playing
            if move != (-1, -1):
                self.instinct_learner.record_move(board, move, record, move_count)

            # Play move
            if move == (-1, -1):
                board.pass_move()
            elif board.is_valid_move(move[0], move[1]):
                board.play(move[0], move[1])
            else:
                board.pass_move()

            move_count += 1
            temperature = max(0.3, temperature * 0.95)

        record.length = move_count
        winner = board.get_winner()
        self.instinct_learner.finalize_game(record, winner)

        return record

    def run_instinct_learning(self, num_games: int = 1000,
                               update_every: int = 100,
                               save_path: str = "instinct_weights.json"):
        """Run training and learn instinct weights."""
        print("\n" + "=" * 70)
        print("ATARI GO INSTINCT WEIGHT LEARNING")
        print("=" * 70)
        print(f"Games: {num_games}")
        print(f"Update weights every: {update_every} games")
        print("=" * 70 + "\n")

        start = time.time()

        for i in range(num_games):
            temp = 1.0 if i < num_games // 2 else 0.5
            record = self.play_game_with_tracking(temperature=temp)

            self.games_played += 1
            if record.winner == 1:
                self.black_wins += 1
            elif record.winner == -1:
                self.white_wins += 1
            else:
                self.draws += 1

            # Update weights periodically
            if (i + 1) % update_every == 0:
                self.instinct_learner.update_weights()

                elapsed = time.time() - start
                gps = (i + 1) / elapsed

                print(f"Game {i+1}/{num_games} | {gps:.1f} g/s | "
                      f"B/W/D: {self.black_wins}/{self.white_wins}/{self.draws}")
                print(self.instinct_learner.get_summary())

        # Final update and save
        self.instinct_learner.update_weights()
        self.instinct_learner.save(save_path)

        elapsed = time.time() - start
        print(f"\nCompleted in {elapsed/60:.1f} minutes")
        print(self.instinct_learner.get_summary())

        return self.instinct_learner.weights


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Learn instinct weights through Atari Go')
    parser.add_argument('--games', type=int, default=2000, help='Number of games')
    parser.add_argument('--update-every', type=int, default=200, help='Update weights every N games')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint')
    parser.add_argument('--output', type=str, default='learned_instinct_weights.json')
    args = parser.parse_args()

    config = Config()
    config.board_size = 9
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = AtariInstinctTrainer(config, args.checkpoint)

    weights = trainer.run_instinct_learning(
        num_games=args.games,
        update_every=args.update_every,
        save_path=args.output
    )

    print("\n" + "=" * 70)
    print("FINAL LEARNED WEIGHTS")
    print("=" * 70)
    for name, weight in sorted(weights.items(), key=lambda x: -x[1]):
        initial = InstinctWeightLearner.INITIAL_WEIGHTS[name]
        delta = weight - initial
        print(f"  {name:<25}: {weight:.2f} (was {initial:.1f}, Δ{delta:+.2f})")


if __name__ == '__main__':
    main()
