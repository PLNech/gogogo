#!/usr/bin/env python3
"""Atari Go: First capture wins.

A simplified Go variant perfect for teaching tactical patterns:
- First player to capture ANY stones wins
- Forces network to learn ladders, nets, snapbacks, life/death
- Symmetric learning: attack patterns = defense patterns
- Fast games (20-50 moves typically)
- Clear reward signal

This bootstraps tactical understanding before full Go training.
"""
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque
import random
import time

from board import Board
from model import GoNet, load_checkpoint
from config import Config
from mcts import MCTS, NNCache
from tactical_data import (
    generate_tactical_dataset, position_to_tensor, TacticalPosition,
    generate_ladder_position, generate_ladder_continuation,
    generate_net_position, generate_snapback
)


@dataclass
class AtariGoGame:
    """Result of an Atari Go game."""
    moves: List[Tuple[int, int]]
    winner: int  # 1 = Black, -1 = White, 0 = Draw (rare)
    captured_at: Tuple[int, int]  # The winning capture move
    length: int


class AtariGoBoard(Board):
    """Board for Atari Go variant.

    Overrides game-over detection: first capture wins.
    """

    def __init__(self, size: int = 9):
        super().__init__(size)
        self.game_winner = None

    def play(self, row: int, col: int) -> int:
        """Play a move and check for capture win.

        Returns:
            Number of stones captured (winner if > 0)
        """
        player_who_moved = self.current_player
        captured = super().play(row, col)

        if captured > 0:
            # The player who just moved wins!
            self.game_winner = player_who_moved

        return captured

    def is_game_over(self) -> bool:
        """Game ends when someone captures or passes twice."""
        if self.game_winner is not None:
            return True
        return super().is_game_over()

    def get_winner(self) -> int:
        """Return winner: 1=Black, -1=White, 0=Draw."""
        if self.game_winner is not None:
            return self.game_winner
        # If game ended by passes, it's a draw (rare in Atari Go)
        return 0

    def copy(self) -> 'AtariGoBoard':
        """Create a copy of the board."""
        new_board = AtariGoBoard(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.history = [h.copy() for h in self.history]
        new_board.ko_point = self.ko_point
        new_board.passes = self.passes
        new_board.move_count = self.move_count
        new_board.game_winner = self.game_winner
        return new_board


class AtariGoMCTS(MCTS):
    """MCTS adapted for Atari Go.

    Key differences:
    - Terminal value is win/loss based on capture
    - No territory scoring
    """

    def __init__(self, model, config: Config, **kwargs):
        super().__init__(model, config, **kwargs)

    def search(self, board: AtariGoBoard, verbose: bool = False) -> np.ndarray:
        """Run MCTS for Atari Go position."""
        # Use parent's search but with Atari Go board
        return super().search(board, verbose)


@dataclass
class TrainingSample:
    """A single training sample from Atari Go."""
    board_tensor: np.ndarray
    policy_target: np.ndarray
    value_target: float  # +1 if current player wins, -1 if loses


@dataclass
class ParallelGame:
    """State for a game being played in parallel."""
    board: AtariGoBoard
    samples: List[TrainingSample]
    moves: List[Tuple[int, int]]
    done: bool = False
    winner: int = 0


class AtariGoTrainer:
    """Self-play trainer for Atari Go with parallel game execution."""

    def __init__(self, config: Config, checkpoint_path: Optional[str] = None,
                 num_parallel: int = 16):
        self.config = config
        self.config.board_size = 9  # Atari Go on 9x9
        self.num_parallel = num_parallel  # Games to play in parallel

        # Initialize or load model
        if checkpoint_path:
            self.model, self.step = load_checkpoint(checkpoint_path, config)
            print(f"Loaded checkpoint from {checkpoint_path} (step {self.step})")
            # Resize for 9x9 if needed
            if self.model.config.board_size != 9:
                print(f"Warning: Model is {self.model.config.board_size}x{self.model.config.board_size}, creating new 9x9 model")
                self.config.board_size = 9
                self.model = GoNet(self.config).to(config.device)
                self.step = 0
        else:
            self.model = GoNet(self.config).to(config.device)
            self.step = 0

        self.model.to(config.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )

        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=100000)

        # Stats
        self.games_played = 0
        self.black_wins = 0
        self.white_wins = 0
        self.draws = 0
        self.total_game_length = 0

    def _batch_evaluate(self, boards: List[AtariGoBoard]) -> Tuple[np.ndarray, np.ndarray]:
        """Batch evaluate multiple positions at once."""
        if not boards:
            return np.array([]), np.array([])

        self.model.eval()
        with torch.no_grad():
            tensors = np.stack([b.to_tensor() for b in boards])
            x = torch.FloatTensor(tensors).to(self.config.device)
            outputs = self.model(x)
            policies = torch.exp(outputs[0]).cpu().numpy()
            values = outputs[1].cpu().numpy().flatten()
        return policies, values

    def _fast_policy_move(self, board: AtariGoBoard, policy: np.ndarray,
                          temperature: float) -> Tuple[int, int]:
        """Select move from policy without full MCTS (for speed)."""
        size = board.size

        # Mask illegal moves
        legal_mask = np.zeros(size * size + 1)
        for r, c in board.get_legal_moves():
            legal_mask[r * size + c] = 1
        legal_mask[size * size] = 1  # Pass always legal

        masked_policy = policy * legal_mask
        if masked_policy.sum() < 1e-8:
            masked_policy = legal_mask  # Fall back to uniform

        masked_policy = masked_policy / masked_policy.sum()

        if temperature == 0:
            action_idx = np.argmax(masked_policy)
        else:
            # Temperature sampling
            log_p = np.log(masked_policy + 1e-10) / temperature
            log_p = log_p - log_p.max()
            p = np.exp(log_p)
            p = p / p.sum()
            action_idx = np.random.choice(len(p), p=p)

        if action_idx == size * size:
            return (-1, -1)
        return (action_idx // size, action_idx % size)

    def play_games_parallel(self, num_games: int, temperature: float = 1.0,
                            use_mcts: bool = False, mcts_sims: int = 50
                            ) -> List[Tuple[AtariGoGame, List[TrainingSample]]]:
        """Play multiple games in parallel with batched NN evaluation.

        Args:
            num_games: Total games to play
            temperature: Sampling temperature
            use_mcts: If True, use MCTS (slower). If False, use raw policy (faster).
            mcts_sims: MCTS simulations if use_mcts=True

        Returns:
            List of (game_result, samples) tuples
        """
        results = []
        active_games: List[ParallelGame] = []

        # Initialize parallel games
        for _ in range(min(num_games, self.num_parallel)):
            active_games.append(ParallelGame(
                board=AtariGoBoard(self.config.board_size),
                samples=[],
                moves=[]
            ))

        games_started = len(active_games)
        max_moves = self.config.board_size ** 2 * 2

        while active_games:
            # Collect boards that need evaluation
            boards_to_eval = [g.board for g in active_games if not g.done]

            if not boards_to_eval:
                break

            # Batch evaluate all positions
            policies, values = self._batch_evaluate(boards_to_eval)

            # Process each active game
            policy_idx = 0
            for game in active_games:
                if game.done:
                    continue

                policy = policies[policy_idx]
                policy_idx += 1

                # Store training sample
                game.samples.append(TrainingSample(
                    board_tensor=game.board.to_tensor().copy(),
                    policy_target=policy.copy(),
                    value_target=0.0
                ))

                # Select and play move
                if use_mcts:
                    # Full MCTS (slower but stronger)
                    mcts = AtariGoMCTS(self.model, self.config, batch_size=8)
                    mcts.config.mcts_simulations = mcts_sims
                    mcts_policy = mcts.search(game.board)
                    move = self._fast_policy_move(game.board, mcts_policy, temperature)
                else:
                    # Raw policy (faster, good for early training)
                    move = self._fast_policy_move(game.board, policy, temperature)

                game.moves.append(move)

                if move == (-1, -1):
                    game.board.pass_move()
                elif game.board.is_valid_move(move[0], move[1]):
                    game.board.play(move[0], move[1])
                else:
                    game.board.pass_move()

                # Check game end
                if game.board.is_game_over() or len(game.moves) >= max_moves:
                    game.done = True
                    game.winner = game.board.get_winner()

            # Harvest completed games
            still_active = []
            for game in active_games:
                if game.done:
                    # Fill in value targets
                    winner = game.winner
                    current_player = 1
                    for sample in game.samples:
                        if winner == 0:
                            sample.value_target = 0.0
                        elif winner == current_player:
                            sample.value_target = 1.0
                        else:
                            sample.value_target = -1.0
                        current_player = -current_player

                    results.append((
                        AtariGoGame(
                            moves=game.moves,
                            winner=winner,
                            captured_at=game.moves[-1] if winner != 0 else (-1, -1),
                            length=len(game.moves)
                        ),
                        game.samples
                    ))

                    # Start new game if needed
                    if games_started < num_games:
                        still_active.append(ParallelGame(
                            board=AtariGoBoard(self.config.board_size),
                            samples=[],
                            moves=[]
                        ))
                        games_started += 1
                else:
                    still_active.append(game)

            active_games = still_active

            # Decrease temperature over time
            temperature = max(0.3, temperature * 0.999)

        return results

    def play_game(self, temperature: float = 1.0,
                  mcts_sims: int = 100) -> Tuple[AtariGoGame, List[TrainingSample]]:
        """Play one game of Atari Go with self-play.

        Returns:
            Game result and list of training samples
        """
        board = AtariGoBoard(self.config.board_size)
        mcts = AtariGoMCTS(self.model, self.config, batch_size=8)
        mcts.config.mcts_simulations = mcts_sims

        samples = []
        moves = []

        move_count = 0
        max_moves = self.config.board_size ** 2 * 2  # Safety limit

        while not board.is_game_over() and move_count < max_moves:
            # Get MCTS policy
            policy = mcts.search(board)

            # Store sample (value filled in later)
            samples.append(TrainingSample(
                board_tensor=board.to_tensor().copy(),
                policy_target=policy.copy(),
                value_target=0.0  # Placeholder
            ))

            # Select move
            if temperature == 0:
                action_idx = np.argmax(policy)
            else:
                # Apply temperature with better numerical stability
                log_policy = np.log(policy + 1e-10)
                log_policy_temp = log_policy / temperature
                log_policy_temp = log_policy_temp - log_policy_temp.max()  # Stability
                policy_temp = np.exp(log_policy_temp)
                policy_temp = policy_temp / policy_temp.sum()
                action_idx = np.random.choice(len(policy), p=policy_temp)

            # Convert to move
            if action_idx == board.size ** 2:
                move = (-1, -1)  # Pass
                board.pass_move()
            else:
                move = (action_idx // board.size, action_idx % board.size)
                if board.is_valid_move(move[0], move[1]):
                    board.play(move[0], move[1])
                else:
                    # Invalid move - pass instead
                    move = (-1, -1)
                    board.pass_move()

            moves.append(move)
            move_count += 1

            # Decrease temperature as game progresses
            if move_count > 10:
                temperature = max(0.1, temperature * 0.95)

        # Determine winner
        winner = board.get_winner()

        # Fill in value targets
        # Value from perspective of player to move at that position
        current_player = 1  # Black starts
        for sample in samples:
            if winner == 0:
                sample.value_target = 0.0
            elif winner == current_player:
                sample.value_target = 1.0
            else:
                sample.value_target = -1.0
            current_player = -current_player

        # Find capture move
        captured_at = moves[-1] if winner != 0 else (-1, -1)

        game = AtariGoGame(
            moves=moves,
            winner=winner,
            captured_at=captured_at,
            length=len(moves)
        )

        return game, samples

    def inject_tactical_positions(self, num_positions: int = 500) -> int:
        """Inject supervised tactical positions into replay buffer.

        This provides clean learning signal for ladders, nets, etc.
        Returns number of positions injected.
        """
        data = generate_tactical_dataset(num_positions, self.config.board_size)

        injected = 0
        for i in range(len(data['tensors'])):
            self.replay_buffer.append(TrainingSample(
                board_tensor=data['tensors'][i],
                policy_target=data['policies'][i],
                value_target=data['values'][i]
            ))
            injected += 1

        return injected

    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """Train on a batch from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return {}

        # Sample batch
        batch = random.sample(list(self.replay_buffer), batch_size)

        # Prepare tensors
        boards = torch.FloatTensor(np.stack([s.board_tensor for s in batch])).to(self.config.device)
        policy_targets = torch.FloatTensor(np.stack([s.policy_target for s in batch])).to(self.config.device)
        value_targets = torch.FloatTensor([s.value_target for s in batch]).to(self.config.device)

        # Forward pass
        self.model.train()
        log_policy, value = self.model(boards)[:2]

        # Policy loss (cross entropy with MCTS policy)
        policy_loss = -torch.sum(policy_targets * log_policy, dim=1).mean()

        # Value loss (MSE)
        value_loss = F.mse_loss(value.squeeze(), value_targets)

        # Total loss
        loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.step += 1

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }

    def run_training(self, num_games: int = 1000,
                     games_per_train: int = 10,
                     mcts_sims: int = 100,
                     save_every: int = 100,
                     verbose: bool = True,
                     fast_mode: bool = True,
                     tactical_injection: int = 500,
                     inject_every: int = 200):
        """Run self-play training loop.

        Args:
            num_games: Total games to play
            games_per_train: Train after this many games
            mcts_sims: MCTS simulations per move (only if fast_mode=False)
            save_every: Save checkpoint every N games
            verbose: Print progress
            fast_mode: Use raw policy instead of MCTS (10x faster, good for early training)
            tactical_injection: Number of tactical positions to inject each time
            inject_every: Inject tactical positions every N games
        """
        print(f"\n{'='*60}")
        print("ATARI GO SELF-PLAY TRAINING" + (" (FAST MODE)" if fast_mode else ""))
        print(f"{'='*60}")
        print(f"Board size: {self.config.board_size}x{self.config.board_size}")
        print(f"Parallel games: {self.num_parallel}")
        if not fast_mode:
            print(f"MCTS simulations: {mcts_sims}")
        print(f"Games to play: {num_games}")
        print(f"Tactical injection: {tactical_injection} positions every {inject_every} games")
        print(f"Device: {self.config.device}")
        print(f"{'='*60}\n")

        # Initial tactical injection
        if tactical_injection > 0:
            injected = self.inject_tactical_positions(tactical_injection)
            print(f"Injected {injected} tactical positions (including ladders)")
        last_injection = 0

        start_time = time.time()
        batch_size = self.num_parallel
        games_completed = 0

        while games_completed < num_games:
            # Play batch of games in parallel
            batch_start = time.time()
            games_to_play = min(batch_size, num_games - games_completed)

            results = self.play_games_parallel(
                num_games=games_to_play,
                temperature=1.0 if games_completed < num_games // 2 else 0.5,
                use_mcts=not fast_mode,
                mcts_sims=mcts_sims
            )
            batch_time = time.time() - batch_start

            # Process results
            batch_lengths = []
            for game, samples in results:
                self.replay_buffer.extend(samples)
                self.games_played += 1
                self.total_game_length += game.length
                batch_lengths.append(game.length)

                if game.winner == 1:
                    self.black_wins += 1
                elif game.winner == -1:
                    self.white_wins += 1
                else:
                    self.draws += 1

            games_completed += len(results)

            # Train on accumulated samples
            train_metrics = {}
            if len(self.replay_buffer) >= 128:
                num_train_steps = max(1, len(results) // 2)
                for _ in range(num_train_steps):
                    metrics = self.train_step(batch_size=128)
                    if metrics:
                        for k, v in metrics.items():
                            train_metrics[k] = train_metrics.get(k, 0) + v / num_train_steps

            # Print progress
            if verbose:
                avg_len = self.total_game_length / self.games_played
                games_per_sec = len(results) / batch_time

                status = f"Game {games_completed}/{num_games}"
                status += f" | Batch: {len(results)} in {batch_time:.1f}s ({games_per_sec:.1f} g/s)"
                status += f" | Len: {np.mean(batch_lengths):.0f} (avg: {avg_len:.1f})"
                status += f" | B/W/D: {self.black_wins}/{self.white_wins}/{self.draws}"
                status += f" | Buffer: {len(self.replay_buffer)}"

                if train_metrics:
                    status += f" | Loss: {train_metrics.get('loss', 0):.3f}"

                print(status)

            # Periodic tactical injection
            if tactical_injection > 0 and games_completed - last_injection >= inject_every:
                injected = self.inject_tactical_positions(tactical_injection)
                if verbose:
                    print(f"  → Injected {injected} tactical positions")
                last_injection = games_completed

            # Save checkpoint
            if games_completed % save_every == 0:
                self.save_checkpoint(f"checkpoints/atari_go_{games_completed}.pt")

        elapsed = time.time() - start_time
        games_per_sec = num_games / elapsed
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Games: {self.games_played} ({games_per_sec:.1f} games/sec)")
        print(f"Black wins: {self.black_wins} ({self.black_wins/self.games_played*100:.1f}%)")
        print(f"White wins: {self.white_wins} ({self.white_wins/self.games_played*100:.1f}%)")
        print(f"Draws: {self.draws} ({self.draws/self.games_played*100:.1f}%)")
        print(f"Avg game length: {self.total_game_length/self.games_played:.1f}")
        print(f"Time: {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
        print(f"{'='*60}")

        # Final save
        self.save_checkpoint("checkpoints/atari_go_final.pt")

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,
            'games_played': self.games_played,
            'black_wins': self.black_wins,
            'white_wins': self.white_wins,
        }, path)
        print(f"Saved checkpoint to {path}")


def evaluate_tactics(model, config: Config, num_positions: int = 100) -> Dict[str, float]:
    """Evaluate model on tactical positions.

    Tests if model can:
    1. Capture stones in atari
    2. Escape from atari
    3. Spot ladder captures
    """
    model.eval()

    results = {
        'capture_atari': 0,
        'escape_atari': 0,
        'total': 0
    }

    for _ in range(num_positions):
        board = AtariGoBoard(config.board_size)

        # Create random position with atari
        # ... (simplified for now)
        pass

    return results


def main():
    import argparse
    import signal

    parser = argparse.ArgumentParser(description='Atari Go self-play training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint to resume from')
    parser.add_argument('--games', type=int, default=2000,
                        help='Number of games to play')
    parser.add_argument('--mcts-sims', type=int, default=50,
                        help='MCTS simulations per move (only with --no-fast)')
    parser.add_argument('--parallel', type=int, default=32,
                        help='Number of parallel games')
    parser.add_argument('--save-every', type=int, default=500,
                        help='Save checkpoint every N games')
    parser.add_argument('--no-fast', action='store_true',
                        help='Use MCTS instead of raw policy (slower but stronger)')
    parser.add_argument('--tactical-inject', type=int, default=500,
                        help='Number of tactical positions to inject each time')
    parser.add_argument('--inject-every', type=int, default=200,
                        help='Inject tactical positions every N games')
    parser.add_argument('--no-tactical', action='store_true',
                        help='Disable tactical injection')
    args = parser.parse_args()

    config = Config()
    config.board_size = 9
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = AtariGoTrainer(config, args.checkpoint, num_parallel=args.parallel)

    # Track last checkpoint for Ctrl+C reporting
    last_checkpoint = [None]

    def save_on_interrupt(signum, frame):
        print("\n\n*** Interrupted! Saving checkpoint... ***")
        path = f"checkpoints/atari_go_interrupted_{trainer.games_played}.pt"
        trainer.save_checkpoint(path)
        print(f"\nLast checkpoint: {path}")
        print(f"Games played: {trainer.games_played}")
        print(f"Buffer size: {len(trainer.replay_buffer)}")
        exit(0)

    signal.signal(signal.SIGINT, save_on_interrupt)

    trainer.run_training(
        num_games=args.games,
        mcts_sims=args.mcts_sims,
        save_every=args.save_every,
        fast_mode=not args.no_fast,
        tactical_injection=0 if args.no_tactical else args.tactical_inject,
        inject_every=args.inject_every
    )


if __name__ == '__main__':
    main()
