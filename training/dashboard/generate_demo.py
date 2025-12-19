#!/usr/bin/env python3
"""Generate demo game data for testing the dashboard."""
import sys
from pathlib import Path
from datetime import datetime
import uuid
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from game_record import GameRecord, MoveStats


def generate_demo_game(board_size: int = 9, num_moves: int = 80) -> GameRecord:
    """Generate a realistic-looking demo game."""
    game_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    record = GameRecord(
        board_size=board_size,
        game_id=game_id,
        timestamp=datetime.now().isoformat(),
        black_name="Demo Black",
        white_name="Demo White",
    )

    # Simulate game progression
    black_stones = 0
    white_stones = 0
    black_captures = 0
    white_captures = 0
    black_groups = 0
    white_groups = 0

    # Win probability starts at 50%, drifts with some randomness
    win_prob = 0.5

    for move_num in range(num_moves):
        player = 1 if move_num % 2 == 0 else -1

        # Random move (not pass for now)
        move = (np.random.randint(0, board_size), np.random.randint(0, board_size))

        # Update stone counts
        if player == 1:
            black_stones += 1
            if move_num > 10 and np.random.random() < 0.15:
                black_groups += 1
        else:
            white_stones += 1
            if move_num > 10 and np.random.random() < 0.15:
                white_groups += 1

        # Occasional captures
        captures = 0
        if move_num > 15 and np.random.random() < 0.1:
            captures = np.random.randint(1, 4)
            if player == 1:
                black_captures += captures
                white_stones = max(0, white_stones - captures)
            else:
                white_captures += captures
                black_stones = max(0, black_stones - captures)

        # Group count evolution
        black_groups = max(1, min(black_stones, black_groups + np.random.randint(-1, 2)))
        white_groups = max(1, min(white_stones, white_groups + np.random.randint(-1, 2)))

        # Win probability drift
        drift = np.random.normal(0, 0.03)
        if player == 1 and captures > 0:
            drift += 0.05
        elif player == -1 and captures > 0:
            drift -= 0.05
        win_prob = np.clip(win_prob + drift, 0.05, 0.95)

        # Score estimate
        score = (black_stones - white_stones) + (black_captures - white_captures) * 0.5 - 6.5

        # Liberties (simulated)
        min_black_libs = max(1, np.random.randint(1, 5))
        min_white_libs = max(1, np.random.randint(1, 5))

        # Atari situations
        black_ataris = 1 if min_black_libs == 1 and np.random.random() < 0.3 else 0
        white_ataris = 1 if min_white_libs == 1 and np.random.random() < 0.3 else 0

        # Policy entropy (higher early, lower late)
        entropy = 3.0 * (1 - move_num / num_moves) + np.random.normal(0, 0.3)
        entropy = max(0.5, entropy)

        stats = MoveStats(
            move_num=move_num,
            move=move,
            player=player,
            black_stones=black_stones,
            white_stones=white_stones,
            black_groups=black_groups,
            white_groups=white_groups,
            empty_points=board_size**2 - black_stones - white_stones,
            captures_this_move=captures,
            total_black_captures=black_captures,
            total_white_captures=white_captures,
            score_estimate=score,
            black_territory=max(0, int(score + board_size**2 / 2)),
            white_territory=max(0, int(-score + board_size**2 / 2)),
            ko_point=(3, 3) if move_num == 45 else None,
            black_ataris=black_ataris,
            white_ataris=white_ataris,
            mcts_value=win_prob,
            mcts_visits=800,
            chosen_move_visits=int(800 * np.random.uniform(0.3, 0.7)),
            chosen_move_prior=np.random.uniform(0.1, 0.5),
            policy_entropy=entropy,
            top5_moves=[
                (move, np.random.uniform(0.2, 0.4)),
                ((move[0]+1, move[1]), np.random.uniform(0.1, 0.2)),
                ((move[0], move[1]+1), np.random.uniform(0.05, 0.15)),
                ((move[0]-1, move[1]), np.random.uniform(0.03, 0.1)),
                ((move[0], move[1]-1), np.random.uniform(0.02, 0.08)),
            ],
            min_black_liberties=min_black_libs,
            min_white_liberties=min_white_libs,
            avg_black_liberties=min_black_libs + np.random.uniform(0.5, 2),
            avg_white_liberties=min_white_libs + np.random.uniform(0.5, 2),
        )

        record.add_move(stats)

    # Determine winner based on final win probability
    winner = 1 if win_prob > 0.5 else -1
    final_score = abs(win_prob - 0.5) * 20 + 0.5  # Scale to reasonable score
    if winner == -1:
        final_score = -final_score

    record.finalize(winner, final_score)

    return record


def main():
    """Generate demo games and save them."""
    games_dir = Path(__file__).parent.parent / "games"
    games_dir.mkdir(exist_ok=True)

    print("Generating demo games...")

    for i in range(5):
        record = generate_demo_game(
            board_size=9,
            num_moves=np.random.randint(60, 120)
        )
        path = games_dir / f"{record.game_id}.json"
        record.save(str(path))
        print(f"  Created: {path.name} ({record.total_moves} moves, {record.result_string})")

    print(f"\nDone! Games saved to {games_dir}")
    print("\nStart the dashboard with:")
    print("  cd training && poetry run python dashboard/app.py")


if __name__ == "__main__":
    main()
