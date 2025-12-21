#!/usr/bin/env python3
"""Live training dashboard for terminal."""
import json
import time
import os
import sys
from datetime import datetime, timedelta

CLEAR = "\033[2J\033[H"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"

def format_time(seconds):
    if seconds is None:
        return "calculating..."
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"

def progress_bar(current, total, width=30):
    if total == 0:
        return "░" * width
    pct = min(current / total, 1.0)
    filled = int(width * pct)
    return "█" * filled + "░" * (width - filled)

def main():
    state_file = ".training_state.json"

    while True:
        try:
            print(CLEAR, end="")

            if not os.path.exists(state_file):
                print(f"{RED}No training running.{RESET}")
                print("Start with: poetry run python train.py --instincts")
                time.sleep(2)
                continue

            with open(state_file) as f:
                d = json.load(f)

            if not d.get('active', False):
                print(f"{YELLOW}Training completed or stopped.{RESET}")
                break

            # Header
            print(f"{BOLD}╔══════════════════════════════════════════════════════════╗{RESET}")
            print(f"{BOLD}║          🎯 INSTINCT CURRICULUM TRAINING                 ║{RESET}")
            print(f"{BOLD}╚══════════════════════════════════════════════════════════╝{RESET}")
            print()

            # Iteration progress
            iter_pct = d['iteration'] / d['total_iterations'] * 100
            print(f"  {BOLD}Iteration:{RESET} {d['iteration']}/{d['total_iterations']} ({iter_pct:.1f}%)")
            print(f"  {progress_bar(d['iteration'], d['total_iterations'], 50)}")
            print()

            # Phase-specific display
            phase = d.get('phase', 'unknown')
            phase_colors = {'selfplay': BLUE, 'training': GREEN, 'eval': YELLOW}
            phase_emoji = {'selfplay': '🎮', 'training': '🧠', 'eval': '📊'}

            color = phase_colors.get(phase, RESET)
            emoji = phase_emoji.get(phase, '⏳')

            print(f"  {BOLD}Phase:{RESET} {color}{emoji} {phase.upper()}{RESET}")
            print()

            if phase == 'selfplay':
                games = d.get('games_generated', 0)
                target = d.get('games_target', 100)
                print(f"  {BOLD}Self-Play Games:{RESET}")
                print(f"  {progress_bar(games, target, 50)} {games}/{target}")
                print()
                print(f"  {YELLOW}Generating games with MCTS (slow but smart){RESET}")

            elif phase == 'training':
                step = d.get('train_step', 0)
                target = d.get('train_steps_target', 1000)
                print(f"  {BOLD}Training Steps:{RESET}")
                print(f"  {progress_bar(step, target, 50)} {step}/{target}")
                print()

                # Losses
                print(f"  {BOLD}Losses:{RESET}")
                print(f"    Policy: {d.get('last_policy_loss', 0):.4f}")
                print(f"    Value:  {d.get('last_value_loss', 0):.4f}")
                print(f"    Total:  {d.get('last_loss', 0):.4f}")

            elif phase == 'eval':
                print(f"  {BOLD}Evaluating model...{RESET}")
                win_rate = d.get('eval_win_rate', 0)
                if win_rate > 0:
                    print(f"  Win rate: {win_rate:.1%}")

            print()

            # Buffer status
            buffer = d.get('buffer_size', 0)
            capacity = d.get('buffer_capacity', 100000)
            print(f"  {BOLD}Replay Buffer:{RESET} {buffer:,} / {capacity:,} positions")
            print(f"  {progress_bar(buffer, capacity, 50)}")
            print()

            # ETA
            eta = d.get('eta_seconds')
            started = d.get('started_at', '')
            if started:
                start_time = datetime.fromisoformat(started)
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"  {BOLD}Elapsed:{RESET} {format_time(elapsed)}")

            if eta:
                print(f"  {BOLD}ETA:{RESET} {format_time(eta)}")

            print()
            print(f"  {BOLD}TensorBoard:{RESET} http://localhost:6006")
            print(f"  {BOLD}Press Ctrl+C to exit{RESET}")

            time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n{YELLOW}Exiting dashboard (training continues){RESET}")
            break
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")
            time.sleep(2)

if __name__ == '__main__':
    main()
