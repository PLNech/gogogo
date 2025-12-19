"""Training state tracking for dashboard integration."""
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime

STATE_FILE = Path(__file__).parent / ".training_state.json"


@dataclass
class TrainingState:
    """Current training state."""
    active: bool = False
    started_at: str = ""
    iteration: int = 0
    total_iterations: int = 0
    phase: str = "idle"  # "selfplay", "training", "evaluating", "idle"

    # Self-play stats
    games_generated: int = 0
    games_target: int = 0
    current_game: int = 0

    # Training stats
    train_step: int = 0
    train_steps_target: int = 0
    last_loss: float = 0.0
    last_policy_loss: float = 0.0
    last_value_loss: float = 0.0

    # Buffer stats
    buffer_size: int = 0
    buffer_capacity: int = 0

    # Evaluation stats
    eval_win_rate: float = 0.0
    is_best: bool = False

    # Timing
    last_update: str = ""
    eta_seconds: Optional[int] = None


def write_state(state: TrainingState):
    """Write training state to file."""
    state.last_update = datetime.now().isoformat()
    STATE_FILE.write_text(json.dumps(asdict(state), indent=2))


def read_state() -> Optional[TrainingState]:
    """Read training state from file."""
    if not STATE_FILE.exists():
        return None

    try:
        data = json.loads(STATE_FILE.read_text())
        # Check if stale (no update in 60 seconds = probably dead)
        last_update = datetime.fromisoformat(data.get('last_update', '2000-01-01'))
        age = (datetime.now() - last_update).total_seconds()
        if age > 60:
            data['active'] = False
            data['phase'] = 'stale'
        return TrainingState(**data)
    except Exception:
        return None


def clear_state():
    """Clear training state."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()


class TrainingTracker:
    """Context manager for tracking training state."""

    def __init__(self, total_iterations: int, config=None):
        self.state = TrainingState(
            active=True,
            started_at=datetime.now().isoformat(),
            total_iterations=total_iterations,
            buffer_capacity=config.replay_buffer_size if config else 0,
            games_target=config.games_per_iter if config else 0,
            train_steps_target=config.train_steps_per_iter if config else 0,
        )
        self._start_time = time.time()
        self._iter_start = time.time()

    def __enter__(self):
        write_state(self.state)
        return self

    def __exit__(self, *args):
        self.state.active = False
        self.state.phase = "completed"
        write_state(self.state)

    def start_iteration(self, iteration: int):
        """Called at start of each iteration."""
        self._iter_start = time.time()
        self.state.iteration = iteration
        self.state.current_game = 0
        self.state.train_step = 0
        write_state(self.state)

    def start_selfplay(self, num_games: int):
        """Called when starting self-play phase."""
        self.state.phase = "selfplay"
        self.state.games_target = num_games
        self.state.current_game = 0
        write_state(self.state)

    def game_complete(self, game_idx: int):
        """Called after each game."""
        self.state.current_game = game_idx + 1
        self.state.games_generated += 1
        # Update every 5th game to reduce disk writes
        if game_idx % 5 == 0:
            write_state(self.state)

    def start_training(self, steps: int):
        """Called when starting training phase."""
        self.state.phase = "training"
        self.state.train_steps_target = steps
        self.state.train_step = 0
        write_state(self.state)

    def training_step(self, step: int, losses: dict, buffer_size: int):
        """Called after each training step."""
        self.state.train_step = step
        self.state.last_loss = losses.get('total_loss', 0)
        self.state.last_policy_loss = losses.get('policy_loss', 0)
        self.state.last_value_loss = losses.get('value_loss', 0)
        self.state.buffer_size = buffer_size
        # Update every 50th step
        if step % 50 == 0:
            write_state(self.state)

    def start_eval(self):
        """Called when starting evaluation."""
        self.state.phase = "evaluating"
        write_state(self.state)

    def eval_complete(self, win_rate: float, is_best: bool):
        """Called after evaluation."""
        self.state.eval_win_rate = win_rate
        self.state.is_best = is_best
        write_state(self.state)
