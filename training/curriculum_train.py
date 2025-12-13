#!/usr/bin/env python3
"""Curriculum Training: Atari Go → 9x9 Go → 13x13 → 19x19.

Progressive training strategy that bootstraps tactical understanding
through simplified game variants before scaling to full Go.

Curriculum:
1. Atari Go 9x9 (first capture wins) - teaches tactics
2. Regular Go 9x9 - teaches territory + tactics transfer
3. Regular Go 13x13 - scale up with transferred weights
4. Regular Go 19x19 - final polish

This approach is inspired by KataGo's success with board size curriculum.
"""
import argparse
import torch
from pathlib import Path
from datetime import datetime

from config import Config
from atari_go import AtariGoTrainer


def phase1_atari_go(config: Config, games: int = 2000, mcts_sims: int = 100):
    """Phase 1: Atari Go on 9x9 to learn tactics."""
    print("\n" + "="*70)
    print("PHASE 1: ATARI GO 9x9 (First Capture Wins)")
    print("="*70)
    print("Goal: Learn tactical patterns - captures, ladders, nets, life/death")
    print(f"Games: {games}, MCTS sims: {mcts_sims}")
    print("="*70 + "\n")

    config.board_size = 9

    trainer = AtariGoTrainer(config)
    trainer.run_training(
        num_games=games,
        mcts_sims=mcts_sims,
        games_per_train=10,
        save_every=200
    )

    return "checkpoints/atari_go_final.pt"


def phase2_go_9x9(config: Config, checkpoint: str, games: int = 2000, mcts_sims: int = 200):
    """Phase 2: Regular Go on 9x9 with transferred weights."""
    print("\n" + "="*70)
    print("PHASE 2: REGULAR GO 9x9")
    print("="*70)
    print("Goal: Learn territory scoring while retaining tactics")
    print(f"Starting from: {checkpoint}")
    print(f"Games: {games}, MCTS sims: {mcts_sims}")
    print("="*70 + "\n")

    # Import self-play trainer
    from self_play import SelfPlayTrainer

    config.board_size = 9

    trainer = SelfPlayTrainer(config, checkpoint)
    trainer.run_training(
        num_games=games,
        mcts_sims=mcts_sims
    )

    return "checkpoints/go_9x9_final.pt"


def phase3_go_13x13(config: Config, checkpoint: str, games: int = 1000, mcts_sims: int = 200):
    """Phase 3: Scale up to 13x13."""
    print("\n" + "="*70)
    print("PHASE 3: REGULAR GO 13x13")
    print("="*70)
    print("Goal: Transfer patterns to larger board")
    print(f"Starting from: {checkpoint}")
    print(f"Games: {games}, MCTS sims: {mcts_sims}")
    print("="*70 + "\n")

    from self_play import SelfPlayTrainer

    config.board_size = 13

    trainer = SelfPlayTrainer(config, checkpoint)
    trainer.run_training(
        num_games=games,
        mcts_sims=mcts_sims
    )

    return "checkpoints/go_13x13_final.pt"


def phase4_go_19x19(config: Config, checkpoint: str, games: int = 1000, mcts_sims: int = 400):
    """Phase 4: Full 19x19 Go."""
    print("\n" + "="*70)
    print("PHASE 4: REGULAR GO 19x19")
    print("="*70)
    print("Goal: Master full-size board")
    print(f"Starting from: {checkpoint}")
    print(f"Games: {games}, MCTS sims: {mcts_sims}")
    print("="*70 + "\n")

    from self_play import SelfPlayTrainer

    config.board_size = 19

    trainer = SelfPlayTrainer(config, checkpoint)
    trainer.run_training(
        num_games=games,
        mcts_sims=mcts_sims
    )

    return "checkpoints/go_19x19_final.pt"


def run_curriculum(start_phase: int = 1,
                   checkpoint: str = None,
                   atari_games: int = 2000,
                   go9_games: int = 2000,
                   go13_games: int = 1000,
                   go19_games: int = 1000):
    """Run the full curriculum training."""

    config = Config()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'#'*70}")
    print(f"# CURRICULUM TRAINING - {timestamp}")
    print(f"# Device: {config.device}")
    print(f"# Starting from phase: {start_phase}")
    print(f"{'#'*70}\n")

    current_checkpoint = checkpoint

    # Phase 1: Atari Go
    if start_phase <= 1:
        current_checkpoint = phase1_atari_go(config, atari_games)
        print(f"\n✓ Phase 1 complete: {current_checkpoint}\n")

    # Phase 2: Go 9x9
    if start_phase <= 2:
        if current_checkpoint is None:
            print("Error: Need checkpoint from Phase 1")
            return
        current_checkpoint = phase2_go_9x9(config, current_checkpoint, go9_games)
        print(f"\n✓ Phase 2 complete: {current_checkpoint}\n")

    # Phase 3: Go 13x13
    if start_phase <= 3:
        if current_checkpoint is None:
            print("Error: Need checkpoint from Phase 2")
            return
        current_checkpoint = phase3_go_13x13(config, current_checkpoint, go13_games)
        print(f"\n✓ Phase 3 complete: {current_checkpoint}\n")

    # Phase 4: Go 19x19
    if start_phase <= 4:
        if current_checkpoint is None:
            print("Error: Need checkpoint from Phase 3")
            return
        current_checkpoint = phase4_go_19x19(config, current_checkpoint, go19_games)
        print(f"\n✓ Phase 4 complete: {current_checkpoint}\n")

    print(f"\n{'#'*70}")
    print("# CURRICULUM TRAINING COMPLETE!")
    print(f"# Final model: {current_checkpoint}")
    print(f"{'#'*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Curriculum training for Go AI')
    parser.add_argument('--start-phase', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Phase to start from (1=Atari Go, 2=9x9, 3=13x13, 4=19x19)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to resume from')
    parser.add_argument('--atari-games', type=int, default=2000,
                        help='Games for Atari Go phase')
    parser.add_argument('--go9-games', type=int, default=2000,
                        help='Games for 9x9 Go phase')
    parser.add_argument('--go13-games', type=int, default=1000,
                        help='Games for 13x13 Go phase')
    parser.add_argument('--go19-games', type=int, default=1000,
                        help='Games for 19x19 Go phase')
    parser.add_argument('--phase1-only', action='store_true',
                        help='Only run Atari Go phase')
    args = parser.parse_args()

    if args.phase1_only:
        config = Config()
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        phase1_atari_go(config, args.atari_games)
    else:
        run_curriculum(
            start_phase=args.start_phase,
            checkpoint=args.checkpoint,
            atari_games=args.atari_games,
            go9_games=args.go9_games,
            go13_games=args.go13_games,
            go19_games=args.go19_games
        )


if __name__ == '__main__':
    main()
