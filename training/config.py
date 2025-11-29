"""Training configuration."""
from dataclasses import dataclass

@dataclass
class Config:
    board_size: int = 13  # Sweet spot: 19x19 too big, 9x9 too small
    num_blocks: int = 6
    num_filters: int = 128
    input_planes: int = 17
    mcts_simulations: int = 100
    c_puct: float = 1.5
    temperature: float = 1.0
    temp_threshold: int = 30
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    replay_buffer_size: int = 100_000
    min_replay_size: int = 1000
    games_per_iter: int = 100
    train_steps_per_iter: int = 1000
    checkpoint_interval: int = 500
    eval_games: int = 50
    win_threshold: float = 0.55
    device: str = "cuda"

DEFAULT = Config()
QUICK = Config(num_blocks=4, num_filters=64, mcts_simulations=50, games_per_iter=20, train_steps_per_iter=100)
