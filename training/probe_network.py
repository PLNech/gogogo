#!/usr/bin/env python3
"""Interactive network probing tool with live visualization.

Probe trained networks by:
- Setting up custom positions
- Loading SGF games and stepping through
- Visualizing policy heatmaps in real-time
- Comparing network predictions vs actual moves
- Testing tactical scenarios interactively
"""
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button, Slider, TextBox
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re

from board import Board
from model import load_checkpoint
from config import Config


class NetworkProbe:
    """Interactive network probing interface."""

    def __init__(self, model, config: Config, use_tactical: bool = True):
        self.model = model
        self.config = config
        self.use_tactical = use_tactical
        self.board = Board(config.board_size)
        self.move_history: List[Tuple[int, int]] = []
        self.sgf_moves: List[Tuple[int, int]] = []
        self.sgf_index = 0

        # Setup figure
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        self.ax_board = self.axes[0]
        self.ax_info = self.axes[1]

        # Add control buttons
        self._setup_controls()

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Initial draw
        self._update_display()

    def _setup_controls(self):
        """Setup control buttons and sliders."""
        # Adjust subplot to make room for controls
        plt.subplots_adjust(bottom=0.15)

        # Reset button
        ax_reset = plt.axes([0.1, 0.02, 0.1, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset (R)')
        self.btn_reset.on_clicked(lambda e: self._reset_board())

        # Undo button
        ax_undo = plt.axes([0.22, 0.02, 0.1, 0.04])
        self.btn_undo = Button(ax_undo, 'Undo (U)')
        self.btn_undo.on_clicked(lambda e: self._undo_move())

        # Pass button
        ax_pass = plt.axes([0.34, 0.02, 0.1, 0.04])
        self.btn_pass = Button(ax_pass, 'Pass (P)')
        self.btn_pass.on_clicked(lambda e: self._pass_move())

        # SGF navigation
        ax_prev = plt.axes([0.55, 0.02, 0.08, 0.04])
        self.btn_prev = Button(ax_prev, '< Prev')
        self.btn_prev.on_clicked(lambda e: self._sgf_prev())

        ax_next = plt.axes([0.65, 0.02, 0.08, 0.04])
        self.btn_next = Button(ax_next, 'Next >')
        self.btn_next.on_clicked(lambda e: self._sgf_next())

        # Top-K slider
        ax_topk = plt.axes([0.78, 0.02, 0.15, 0.04])
        self.slider_topk = Slider(ax_topk, 'Top-K', 1, 10, valinit=5, valstep=1)
        self.slider_topk.on_changed(lambda v: self._update_display())

    def get_policy_and_value(self) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        """Get policy, value, and ownership from model."""
        self.model.eval()
        with torch.no_grad():
            tensor = self.board.to_tensor(use_tactical_features=self.use_tactical)
            x = torch.FloatTensor(tensor).unsqueeze(0).to(next(self.model.parameters()).device)

            outputs = self.model(x)
            log_policy = outputs[0]
            value = outputs[1]

            policy = torch.exp(log_policy).cpu().numpy()[0]
            value_scalar = value.cpu().numpy()[0, 0]

            # Get ownership if available
            ownership = None
            if len(outputs) > 2 and outputs[2] is not None:
                ownership = torch.tanh(outputs[2]).cpu().numpy()[0, 0]

            return policy, value_scalar, ownership

    def _draw_board(self):
        """Draw the Go board with stones and policy overlay."""
        ax = self.ax_board
        ax.clear()
        size = self.board.size

        # Board background
        ax.set_xlim(-0.5, size - 0.5)
        ax.set_ylim(-0.5, size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_facecolor('#DEB887')

        # Grid lines
        for i in range(size):
            ax.axhline(y=i, color='black', linewidth=0.5)
            ax.axvline(x=i, color='black', linewidth=0.5)

        # Star points
        if size == 19:
            star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        elif size == 9:
            star_points = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        else:
            star_points = []

        for r, c in star_points:
            ax.add_patch(Circle((c, r), 0.08, color='black'))

        # Get policy for heatmap
        policy, value, ownership = self.get_policy_and_value()
        board_policy = policy[:-1].reshape(size, size)

        # Draw policy heatmap (only on empty points)
        mask = self.board.board != 0
        board_policy_masked = np.ma.array(board_policy, mask=mask)
        im = ax.imshow(board_policy_masked, cmap='YlOrRd', alpha=0.6,
                       extent=[-0.5, size-0.5, size-0.5, -0.5], zorder=5,
                       vmin=0, vmax=board_policy.max())

        # Draw stones
        for r in range(size):
            for c in range(size):
                if self.board.board[r, c] == 1:  # Black
                    ax.add_patch(Circle((c, r), 0.4, color='black', zorder=10))
                    # Mark last move
                    if self.move_history and self.move_history[-1] == (r, c):
                        ax.add_patch(Circle((c, r), 0.15, color='white', zorder=11))
                elif self.board.board[r, c] == -1:  # White
                    ax.add_patch(Circle((c, r), 0.4, color='white', edgecolor='black', linewidth=1, zorder=10))
                    if self.move_history and self.move_history[-1] == (r, c):
                        ax.add_patch(Circle((c, r), 0.15, color='black', zorder=11))

        # Draw top-K move markers
        top_k = int(self.slider_topk.val)
        flat_indices = np.argsort(board_policy.flatten())[::-1]
        count = 0
        for idx in flat_indices:
            if count >= top_k:
                break
            r, c = idx // size, idx % size
            if self.board.board[r, c] == 0:
                prob = board_policy[r, c]
                # Number marker
                ax.text(c, r, str(count + 1), ha='center', va='center',
                       fontsize=14, fontweight='bold', color='blue', zorder=20)
                count += 1

        # Labels
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        col_labels = 'ABCDEFGHJKLMNOPQRST'[:size]  # Skip 'I'
        ax.set_xticklabels(list(col_labels))
        ax.set_yticklabels([str(size - i) for i in range(size)])

        # Title
        to_play = 'Black' if self.board.current_player == 1 else 'White'
        ax.set_title(f'{to_play} to play | Value: {value:+.3f}', fontsize=14)

        return policy, value, ownership

    def _draw_info_panel(self, policy: np.ndarray, value: float, ownership: Optional[np.ndarray]):
        """Draw information panel with stats and top moves."""
        ax = self.ax_info
        ax.clear()
        ax.axis('off')

        size = self.board.size
        board_policy = policy[:-1].reshape(size, size)

        lines = []
        lines.append("NETWORK PROBE")
        lines.append("=" * 40)
        lines.append("")

        # Value interpretation
        if value > 0.3:
            value_str = f"Black winning ({value:+.3f})"
        elif value < -0.3:
            value_str = f"White winning ({value:+.3f})"
        else:
            value_str = f"Close game ({value:+.3f})"
        lines.append(f"Value: {value_str}")
        lines.append(f"Pass probability: {policy[-1]:.1%}")
        lines.append("")

        # Top moves
        lines.append("Top 10 Moves:")
        lines.append("-" * 30)
        flat_indices = np.argsort(board_policy.flatten())[::-1]
        col_labels = 'ABCDEFGHJKLMNOPQRST'[:size]

        count = 0
        for idx in flat_indices:
            if count >= 10:
                break
            r, c = idx // size, idx % size
            if self.board.board[r, c] == 0:
                prob = board_policy[r, c]
                coord = f"{col_labels[c]}{size - r}"
                lines.append(f"  {count+1:2d}. {coord:4s} {prob:6.2%}")
                count += 1

        lines.append("")
        lines.append("-" * 30)
        lines.append("Controls:")
        lines.append("  Click: Play move")
        lines.append("  R: Reset board")
        lines.append("  U: Undo last move")
        lines.append("  P: Pass")
        lines.append("  Q: Quit")
        lines.append("  </> : SGF navigation")

        if self.sgf_moves:
            lines.append("")
            lines.append(f"SGF: Move {self.sgf_index}/{len(self.sgf_moves)}")

        # Display text
        text = '\n'.join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace')

        # Draw ownership map if available
        if ownership is not None:
            # Small inset for ownership
            inset_ax = ax.inset_axes([0.5, 0.1, 0.45, 0.45])
            inset_ax.set_title('Ownership', fontsize=10)
            im = inset_ax.imshow(ownership, cmap='RdBu', vmin=-1, vmax=1)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            # Colorbar
            cbar = plt.colorbar(im, ax=inset_ax, shrink=0.8)
            cbar.set_ticks([-1, 0, 1])
            cbar.set_ticklabels(['W', '?', 'B'])

    def _update_display(self):
        """Update the full display."""
        policy, value, ownership = self._draw_board()
        self._draw_info_panel(policy, value, ownership)
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        """Handle mouse click on board."""
        if event.inaxes != self.ax_board:
            return

        # Convert click to board coordinates
        c = int(round(event.xdata))
        r = int(round(event.ydata))

        if 0 <= r < self.board.size and 0 <= c < self.board.size:
            if self.board.is_valid_move(r, c):
                self.board.play(r, c)
                self.move_history.append((r, c))
                self._update_display()

    def _on_key(self, event):
        """Handle keyboard input."""
        if event.key == 'r':
            self._reset_board()
        elif event.key == 'u':
            self._undo_move()
        elif event.key == 'p':
            self._pass_move()
        elif event.key == 'q':
            plt.close(self.fig)
        elif event.key == ',':  # < key
            self._sgf_prev()
        elif event.key == '.':  # > key
            self._sgf_next()

    def _reset_board(self):
        """Reset to empty board."""
        self.board = Board(self.config.board_size)
        self.move_history = []
        self.sgf_index = 0
        self._update_display()

    def _undo_move(self):
        """Undo last move (rebuild from history)."""
        if self.move_history:
            self.move_history.pop()
            # Rebuild board from history
            self.board = Board(self.config.board_size)
            for r, c in self.move_history:
                if r == -1 and c == -1:
                    self.board.pass_move()
                else:
                    self.board.play(r, c)
            self._update_display()

    def _pass_move(self):
        """Play a pass."""
        self.board.pass_move()
        self.move_history.append((-1, -1))
        self._update_display()

    def load_sgf(self, sgf_path: str):
        """Load SGF file for playback."""
        with open(sgf_path, 'r') as f:
            content = f.read()

        # Simple SGF parsing for moves
        moves = []
        # Match B[xx] or W[xx] patterns
        pattern = r';([BW])\[([a-s]{2})\]'
        for match in re.finditer(pattern, content):
            color, coord = match.groups()
            if len(coord) == 2:
                c = ord(coord[0]) - ord('a')
                r = ord(coord[1]) - ord('a')
                moves.append((r, c))

        self.sgf_moves = moves
        self.sgf_index = 0
        self._reset_board()
        print(f"Loaded {len(moves)} moves from {sgf_path}")

    def _sgf_next(self):
        """Play next SGF move."""
        if self.sgf_moves and self.sgf_index < len(self.sgf_moves):
            r, c = self.sgf_moves[self.sgf_index]
            if self.board.is_valid_move(r, c):
                self.board.play(r, c)
                self.move_history.append((r, c))
                self.sgf_index += 1
            self._update_display()

    def _sgf_prev(self):
        """Go back one SGF move."""
        if self.sgf_index > 0:
            self.sgf_index -= 1
            self._undo_move()

    def run(self):
        """Start the interactive session."""
        print("\nNetwork Probe Interactive Mode")
        print("=" * 40)
        print("Click on board to play moves")
        print("Press 'Q' to quit")
        print()
        plt.show()


def setup_test_position(board: Board, position: str) -> str:
    """Setup a predefined test position."""
    positions = {
        'capture': {
            'desc': 'Black to capture white stone',
            'black': [(9, 8), (9, 10), (8, 9)],
            'white': [(9, 9)],
        },
        'atari': {
            'desc': 'Black stone in atari - find escape',
            'black': [(9, 9)],
            'white': [(9, 8), (9, 10), (8, 9)],
        },
        'ladder': {
            'desc': 'Ladder setup',
            'black': [(9, 9), (10, 8)],
            'white': [(9, 8)],
        },
        'corner': {
            'desc': '3-3 invasion response',
            'black': [(3, 3)],
            'white': [(2, 2)],
        },
        'approach': {
            'desc': 'Respond to approach',
            'black': [(3, 3)],
            'white': [(2, 5)],
        },
    }

    if position not in positions:
        return f"Unknown position. Available: {list(positions.keys())}"

    pos = positions[position]
    for r, c in pos.get('black', []):
        board.board[r, c] = 1
    for r, c in pos.get('white', []):
        board.board[r, c] = -1

    return pos['desc']


def main():
    parser = argparse.ArgumentParser(description='Interactive network probing tool')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--sgf', type=str, default=None, help='SGF file to load')
    parser.add_argument('--position', type=str, default=None,
                        choices=['capture', 'atari', 'ladder', 'corner', 'approach'],
                        help='Start with test position')
    parser.add_argument('--no-tactical', action='store_true',
                        help='Disable tactical features')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    config = Config()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, step = load_checkpoint(args.checkpoint, config)
    print(f"Loaded model (step {step}, board_size={model.config.board_size})")

    use_tactical = not args.no_tactical and model.config.input_planes == 27
    print(f"Using {'tactical' if use_tactical else 'basic'} features ({model.config.input_planes} planes)")

    # Create probe
    probe = NetworkProbe(model, model.config, use_tactical)

    # Setup initial position if requested
    if args.position:
        desc = setup_test_position(probe.board, args.position)
        print(f"Position: {desc}")

    # Load SGF if provided
    if args.sgf:
        probe.load_sgf(args.sgf)

    # Run interactive session
    probe.run()


if __name__ == '__main__':
    main()
