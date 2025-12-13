#!/usr/bin/env python3
"""Regular Go self-play training.

Builds on Atari Go tactical foundation, adding:
- Territory scoring (Tromp-Taylor area scoring)
- Komi (6.5 points for White)
- Game ends on double pass
- Value based on game outcome (win/loss)

Designed to transfer tactical knowledge from Atari Go while
learning territory and endgame concepts.
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


@dataclass
class GoGame:
    """Result of a regular Go game."""
    moves: List[Tuple[int, int]]
    winner: int  # 1 = Black, -1 = White, 0 = Draw (jigo)
    final_score: float  # Positive = Black wins by X
    length: int


@dataclass
class TrainingSample:
    """A single training sample from Go."""
    board_tensor: np.ndarray
    policy_target: np.ndarray
    value_target: float  # +1 if current player wins, -1 if loses


@dataclass
class ParallelGame:
    """State for a game being played in parallel."""
    board: Board
    samples: List[TrainingSample]
    moves: List[Tuple[int, int]]
    done: bool = False
    winner: int = 0
    final_score: float = 0.0


class SelfPlayTrainer:
    """Self-play trainer for regular Go with parallel game execution."""

    def __init__(self, config: Config, checkpoint_path: Optional[str] = None,
                 num_parallel: int = 16):
        self.config = config
        self.num_parallel = num_parallel

        # Initialize or load model
        if checkpoint_path:
            self.model, self.step = load_checkpoint(checkpoint_path, config)
            print(f"Loaded checkpoint from {checkpoint_path} (step {self.step})")
            # Adjust for board size mismatch
            if self.model.config.board_size != config.board_size:
                print(f"Warning: Model is {self.model.config.board_size}x{self.model.config.board_size}, "
                      f"creating new {config.board_size}x{config.board_size} model")
                self.model = GoNet(config).to(config.device)
                self.step = 0
        else:
            self.model = GoNet(config).to(config.device)
            self.step = 0

        self.model.to(config.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )

        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=200000)

        # Stats
        self.games_played = 0
        self.black_wins = 0
        self.white_wins = 0
        self.draws = 0
        self.total_game_length = 0
        self.total_score = 0.0

    def _batch_evaluate(self, boards: List[Board]) -> Tuple[np.ndarray, np.ndarray]:
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

    def _fast_policy_move(self, board: Board, policy: np.ndarray,
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

    def _should_resign(self, board: Board, value: float, move_count: int) -> bool:
        """Decide if the position is hopeless and should resign.

        Conservative resignation policy to avoid giving up winnable positions.
        """
        # Don't resign early in the game
        if move_count < board.size * 2:
            return False

        # Resign if value is extremely low and we've played enough moves
        if value < -0.9 and move_count > board.size * 4:
            return True

        return False

    def _apply_tactical_boosts(self, board: Board, policy: np.ndarray,
                               tactics, weight: float) -> np.ndarray:
        """Apply tactical boosts to neural network policy.

        Blends NN policy with tactical adjustments using additive boosts.
        This helps overcome bad NN priors for tactical moves.

        Args:
            board: Current board state
            policy: Neural network policy (probability distribution)
            tactics: TacticalAnalyzer instance
            weight: Blend weight for tactical adjustments (0-1)

        Returns:
            Adjusted policy with tactical boosts applied
        """
        if weight <= 0:
            return policy

        size = board.size
        adjusted = policy.copy()
        any_adjustment = False

        # Get legal moves
        for move in board.get_legal_moves():
            if move == (-1, -1):
                continue  # Skip pass

            move_idx = move[0] * size + move[1]
            boost = tactics.get_tactical_boost(board, move)

            # Apply significant tactical signals
            if boost > 1.2 or boost < 0.5:
                # Additive boost: add weight * boost to probability
                # This helps tactical moves even with low NN prior
                adjusted[move_idx] = policy[move_idx] + weight * (boost - 1.0)
                adjusted[move_idx] = max(adjusted[move_idx], 1e-8)  # Keep positive
                any_adjustment = True

        if any_adjustment:
            # Re-normalize to valid distribution
            if adjusted.sum() > 1e-8:
                adjusted = adjusted / adjusted.sum()
            else:
                return policy  # Fall back on error

        return adjusted

    def play_games_parallel(self, num_games: int, temperature: float = 1.0,
                            use_mcts: bool = False, mcts_sims: int = 100,
                            resign_threshold: float = -0.95,
                            use_hybrid: bool = False, tactical_weight: float = 0.3
                            ) -> List[Tuple[GoGame, List[TrainingSample]]]:
        """Play multiple games in parallel with batched NN evaluation.

        Args:
            num_games: Total games to play
            temperature: Sampling temperature
            use_mcts: If True, use MCTS (slower). If False, use raw policy (faster).
            mcts_sims: MCTS simulations if use_mcts=True
            resign_threshold: Value below which to resign
            use_hybrid: If True, apply tactical boosts to policy
            tactical_weight: Weight for tactical adjustments (0-1)

        Returns:
            List of (game_result, samples) tuples
        """
        # Initialize tactical analyzer if using hybrid mode
        tactics = None
        if use_hybrid:
            from tactics import TacticalAnalyzer
            tactics = TacticalAnalyzer()

        results = []
        active_games: List[ParallelGame] = []

        # Initialize parallel games
        for _ in range(min(num_games, self.num_parallel)):
            active_games.append(ParallelGame(
                board=Board(self.config.board_size),
                samples=[],
                moves=[]
            ))

        games_started = len(active_games)
        max_moves = self.config.board_size ** 2 * 2  # Allow for endgame

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
                value = values[policy_idx]
                policy_idx += 1

                # Apply tactical boosts if using hybrid mode
                if tactics is not None:
                    policy = self._apply_tactical_boosts(
                        game.board, policy, tactics, tactical_weight
                    )

                # Store training sample (with tactical-enhanced policy)
                game.samples.append(TrainingSample(
                    board_tensor=game.board.to_tensor().copy(),
                    policy_target=policy.copy(),
                    value_target=0.0
                ))

                # Check for resignation
                if self._should_resign(game.board, value, len(game.moves)):
                    game.done = True
                    game.winner = -game.board.current_player  # Opponent wins
                    game.final_score = -20.0 * game.board.current_player  # Resign score
                    continue

                # Select and play move
                if use_mcts:
                    if tactics is not None:
                        from hybrid_mcts import HybridMCTS
                        mcts = HybridMCTS(self.model, self.config, tactics,
                                         batch_size=8, tactical_weight=tactical_weight)
                    else:
                        mcts = MCTS(self.model, self.config, batch_size=8)
                    mcts.config.mcts_simulations = mcts_sims
                    mcts_policy = mcts.search(game.board)
                    move = self._fast_policy_move(game.board, mcts_policy, temperature)
                else:
                    move = self._fast_policy_move(game.board, policy, temperature)

                game.moves.append(move)

                if move == (-1, -1):
                    game.board.pass_move()
                elif game.board.is_valid_move(move[0], move[1]):
                    game.board.play(move[0], move[1])
                else:
                    game.board.pass_move()

                # Check game end (double pass or move limit)
                if game.board.is_game_over() or len(game.moves) >= max_moves:
                    game.done = True
                    game.final_score = game.board.score()
                    if game.final_score > 0:
                        game.winner = 1  # Black wins
                    elif game.final_score < 0:
                        game.winner = -1  # White wins
                    else:
                        game.winner = 0  # Jigo (draw)

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
                        GoGame(
                            moves=game.moves,
                            winner=winner,
                            final_score=game.final_score,
                            length=len(game.moves)
                        ),
                        game.samples
                    ))

                    # Start new game if needed
                    if games_started < num_games:
                        still_active.append(ParallelGame(
                            board=Board(self.config.board_size),
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
                     save_every: int = 500,
                     verbose: bool = True,
                     fast_mode: bool = True,
                     use_hybrid: bool = False,
                     tactical_weight: float = 0.3):
        """Run self-play training loop for regular Go.

        Args:
            num_games: Total games to play
            games_per_train: Train after this many games
            mcts_sims: MCTS simulations per move (only if fast_mode=False)
            save_every: Save checkpoint every N games
            verbose: Print progress
            fast_mode: Use raw policy instead of MCTS (faster, good for early training)
            use_hybrid: If True, apply tactical boosts during play
            tactical_weight: Weight for tactical adjustments (0-1)
        """
        mode_str = " (FAST MODE)" if fast_mode else ""
        mode_str += " [HYBRID]" if use_hybrid else ""
        print(f"\n{'='*60}")
        print(f"REGULAR GO {self.config.board_size}x{self.config.board_size} SELF-PLAY TRAINING{mode_str}")
        print(f"{'='*60}")
        print(f"Board size: {self.config.board_size}x{self.config.board_size}")
        print(f"Parallel games: {self.num_parallel}")
        if not fast_mode:
            print(f"MCTS simulations: {mcts_sims}")
        if use_hybrid:
            print(f"Tactical weight: {tactical_weight}")
        print(f"Games to play: {num_games}")
        print(f"Komi: 6.5")
        print(f"Device: {self.config.device}")
        print(f"{'='*60}\n")

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
                mcts_sims=mcts_sims,
                use_hybrid=use_hybrid,
                tactical_weight=tactical_weight
            )
            batch_time = time.time() - batch_start

            # Process results
            batch_lengths = []
            batch_scores = []
            for game, samples in results:
                self.replay_buffer.extend(samples)
                self.games_played += 1
                self.total_game_length += game.length
                self.total_score += abs(game.final_score)
                batch_lengths.append(game.length)
                batch_scores.append(game.final_score)

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
                avg_score = self.total_score / self.games_played
                games_per_sec = len(results) / batch_time

                status = f"Game {games_completed}/{num_games}"
                status += f" | Batch: {len(results)} in {batch_time:.1f}s ({games_per_sec:.1f} g/s)"
                status += f" | Len: {np.mean(batch_lengths):.0f} (avg: {avg_len:.1f})"
                status += f" | Score: {np.mean(batch_scores):+.1f} (avg: {avg_score:.1f})"
                status += f" | B/W/D: {self.black_wins}/{self.white_wins}/{self.draws}"
                status += f" | Buffer: {len(self.replay_buffer)}"

                if train_metrics:
                    status += f" | Loss: {train_metrics.get('loss', 0):.3f}"

                print(status)

            # Save checkpoint
            if games_completed % save_every == 0:
                self.save_checkpoint(f"checkpoints/go_{self.config.board_size}x{self.config.board_size}_{games_completed}.pt")

        elapsed = time.time() - start_time
        games_per_sec = num_games / elapsed
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Games: {self.games_played} ({games_per_sec:.1f} games/sec)")
        print(f"Black wins: {self.black_wins} ({self.black_wins/self.games_played*100:.1f}%)")
        print(f"White wins: {self.white_wins} ({self.white_wins/self.games_played*100:.1f}%)")
        print(f"Draws (jigo): {self.draws} ({self.draws/self.games_played*100:.1f}%)")
        print(f"Avg game length: {self.total_game_length/self.games_played:.1f}")
        print(f"Avg score margin: {self.total_score/self.games_played:.1f}")
        print(f"Time: {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
        print(f"{'='*60}")

        # Final save
        self.save_checkpoint(f"checkpoints/go_{self.config.board_size}x{self.config.board_size}_final.pt")

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


def main():
    import argparse
    import signal

    parser = argparse.ArgumentParser(description='Regular Go self-play training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint to resume from')
    parser.add_argument('--board-size', type=int, default=9,
                        help='Board size (9, 13, or 19)')
    parser.add_argument('--games', type=int, default=2000,
                        help='Number of games to play')
    parser.add_argument('--mcts-sims', type=int, default=100,
                        help='MCTS simulations per move (only with --no-fast)')
    parser.add_argument('--parallel', type=int, default=32,
                        help='Number of parallel games')
    parser.add_argument('--save-every', type=int, default=500,
                        help='Save checkpoint every N games')
    parser.add_argument('--no-fast', action='store_true',
                        help='Use MCTS instead of raw policy (slower but stronger)')
    parser.add_argument('--hybrid', action='store_true',
                        help='Enable tactical boost integration (neurosymbolic)')
    parser.add_argument('--tactical-weight', type=float, default=0.3,
                        help='Weight for tactical adjustments (0-1, default: 0.3)')
    args = parser.parse_args()

    config = Config()
    config.board_size = args.board_size
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = SelfPlayTrainer(config, args.checkpoint, num_parallel=args.parallel)

    def save_on_interrupt(signum, frame):
        print("\n\n*** Interrupted! Saving checkpoint... ***")
        path = f"checkpoints/go_{config.board_size}x{config.board_size}_interrupted_{trainer.games_played}.pt"
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
        use_hybrid=args.hybrid,
        tactical_weight=args.tactical_weight
    )


if __name__ == '__main__':
    main()
