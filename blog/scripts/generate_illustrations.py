#!/usr/bin/env python3
"""
Blog Illustration Generator for GoGoGo
Creates Go-board diagrams, charts, and technical illustrations.

Usage:
    python generate_illustrations.py          # Generate all illustrations
    python generate_illustrations.py --post X # Generate for specific post
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Paths
BLOG_DIR = Path(__file__).parent.parent
IMAGES_DIR = BLOG_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Style constants
COLORS = {
    'board': '#DEB887',
    'grid': '#4A4036',
    'black': '#1a1a1a',
    'white': '#f5f5f5',
    'white_stroke': '#333333',
    'text': '#2d2a26',
    'highlight_good': '#4a7c59',
    'highlight_bad': '#9e4244',
    'highlight_blue': '#4a6fa5',
    'highlight_yellow': '#d4a017',
    'bg': '#faf8f5',
}


def setup_board_axes(ax, size=9, show_coords=False):
    """Set up axes for a Go board."""
    ax.set_facecolor(COLORS['board'])
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_aspect('equal')

    # Grid lines
    for i in range(size):
        ax.axhline(y=i, color=COLORS['grid'], linewidth=0.8, alpha=0.7)
        ax.axvline(x=i, color=COLORS['grid'], linewidth=0.8, alpha=0.7)

    # Star points
    star_points = {
        9: [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)],
        13: [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)],
        19: [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)],
    }
    if size in star_points:
        for x, y in star_points[size]:
            circle = patches.Circle((x, y), 0.12, color=COLORS['grid'])
            ax.add_patch(circle)

    if show_coords:
        for i in range(size):
            ax.text(i, -0.8, chr(65 + i) if i < 8 else chr(66 + i),
                   ha='center', fontsize=8, color=COLORS['text'])
            ax.text(-0.8, i, str(i + 1), va='center', fontsize=8, color=COLORS['text'])

    ax.axis('off')
    return ax


def draw_stone(ax, x, y, color='black', label=None, alpha=1.0):
    """Draw a Go stone."""
    fill_color = COLORS['black'] if color == 'black' else COLORS['white']
    stroke_color = COLORS['grid'] if color == 'black' else COLORS['white_stroke']

    circle = patches.Circle((x, y), 0.42, facecolor=fill_color,
                            edgecolor=stroke_color, linewidth=1.5, alpha=alpha)
    ax.add_patch(circle)

    if label:
        text_color = COLORS['white'] if color == 'black' else COLORS['black']
        ax.text(x, y, str(label), ha='center', va='center',
               fontsize=10, fontweight='bold', color=text_color)


def draw_marker(ax, x, y, marker_type='x', color='red'):
    """Draw a marker on the board."""
    if marker_type == 'x':
        ax.plot(x, y, 'x', markersize=15, markeredgewidth=3, color=color)
    elif marker_type == 'circle':
        circle = patches.Circle((x, y), 0.3, fill=False,
                                edgecolor=color, linewidth=2)
        ax.add_patch(circle)
    elif marker_type == 'square':
        rect = patches.Rectangle((x - 0.25, y - 0.25), 0.5, 0.5,
                                  fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)


def highlight_area(ax, points, color, alpha=0.3):
    """Highlight an area on the board."""
    for x, y in points:
        rect = patches.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                                  facecolor=color, alpha=alpha)
        ax.add_patch(rect)


# =============================================================================
# ILLUSTRATION GENERATORS
# =============================================================================

def generate_wall_problem():
    """Generate the wall problem illustration."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor=COLORS['bg'])
    setup_board_axes(ax, size=9)

    # The infamous wall pattern
    wall_moves = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]
    for i, (x, y) in enumerate(wall_moves, 1):
        draw_stone(ax, x, y, 'black', label=2*i - 1)

    # Some opponent stones scattered
    white_moves = [(5, 2), (5, 4), (5, 6), (7, 3), (7, 5)]
    for i, (x, y) in enumerate(white_moves, 1):
        draw_stone(ax, x, y, 'white', label=2*i)

    ax.set_title('The Wall Problem\n"Stones placed in straight lines"',
                fontsize=14, color=COLORS['text'], pad=10)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'wall-problem.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: wall-problem.png")


def generate_mcts_tree():
    """Generate MCTS tree diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor=COLORS['bg'])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Tree structure
    nodes = {
        'root': (5, 7),
        'c1': (2, 5), 'c2': (5, 5), 'c3': (8, 5),
        'c11': (1, 3), 'c12': (3, 3),
        'c21': (4, 3), 'c22': (6, 3),
        'c31': (7, 3), 'c32': (9, 3),
        'expand': (2, 1),
    }

    # Draw edges
    edges = [
        ('root', 'c1'), ('root', 'c2'), ('root', 'c3'),
        ('c1', 'c11'), ('c1', 'c12'),
        ('c2', 'c21'), ('c2', 'c22'),
        ('c3', 'c31'), ('c3', 'c32'),
    ]

    for n1, n2 in edges:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        style = '-' if n1 != 'c1' else '-'
        lw = 2.5 if n1 == 'root' and n2 == 'c1' else 1.5
        color = COLORS['highlight_blue'] if (n1 == 'root' and n2 == 'c1') or (n1 == 'c1' and n2 == 'c11') else COLORS['grid']
        ax.plot([x1, x2], [y1, y2], style, color=color, linewidth=lw, zorder=1)

    # Expansion edge (dashed)
    ax.plot([1, 2], [3, 1], '--', color=COLORS['highlight_good'], linewidth=2, zorder=1)

    # Draw nodes
    for name, (x, y) in nodes.items():
        if name == 'expand':
            color = COLORS['highlight_good']
            label = '?'
        elif name == 'root':
            color = COLORS['highlight_blue']
            label = 'R'
        elif name.startswith('c1'):
            color = COLORS['highlight_blue']
            label = name[-1] if len(name) > 2 else '1'
        else:
            color = COLORS['text']
            label = name[-1] if len(name) > 2 else name[1]

        circle = patches.Circle((x, y), 0.4, facecolor=color, edgecolor=COLORS['grid'],
                                linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=10,
               fontweight='bold', color='white', zorder=3)

    # Labels
    ax.annotate('1. SELECT', xy=(2.5, 6), fontsize=11, color=COLORS['highlight_blue'],
               fontweight='bold')
    ax.annotate('2. EXPAND', xy=(0.5, 2), fontsize=11, color=COLORS['highlight_good'],
               fontweight='bold')

    # Simulation line
    ax.annotate('', xy=(2, 0.3), xytext=(2, 0.8),
               arrowprops=dict(arrowstyle='->', color=COLORS['highlight_yellow'], lw=2))
    ax.text(2.5, 0.5, '3. SIMULATE', fontsize=11, color=COLORS['highlight_yellow'],
           fontweight='bold')

    # Backprop arrows - flow UP from expanded node through tree to root
    # expand (2,1) -> c11 (1,3) -> c1 (2,5) -> root (5,7)
    backprop_path = [(2, 1.5), (1, 2.5), (1, 3.5), (2, 4.5), (2, 5.5), (5, 6.5)]
    for i in range(len(backprop_path) - 1):
        ax.annotate('', xy=backprop_path[i+1], xytext=backprop_path[i],
                   arrowprops=dict(arrowstyle='->', color=COLORS['highlight_bad'],
                                  lw=2, ls='--'))
    ax.text(0.2, 4.2, '4. BACKPROP', fontsize=11, color=COLORS['highlight_bad'],
           fontweight='bold', rotation=90)

    ax.set_title('Monte Carlo Tree Search\nSelect → Expand → Simulate → Backpropagate',
                fontsize=14, color=COLORS['text'], pad=10)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'mcts-tree.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: mcts-tree.png")


def generate_snapback():
    """Generate snapback pattern illustration."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor=COLORS['bg'])

    for ax in axes:
        setup_board_axes(ax, size=7)

    # Position 1: Before throw-in
    ax = axes[0]
    # Black tiger mouth
    black_stones = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3), (4, 4)]
    for x, y in black_stones:
        draw_stone(ax, x, y, 'black')
    draw_marker(ax, 3, 3, 'x', COLORS['highlight_bad'])
    ax.set_title('1. White plays X\n(throw-in)', fontsize=11, color=COLORS['text'])

    # Position 2: After capture
    ax = axes[1]
    for x, y in black_stones:
        draw_stone(ax, x, y, 'black')
    draw_stone(ax, 3, 3, 'white', label='1')
    highlight_area(ax, [(3, 3)], COLORS['highlight_bad'], 0.3)
    ax.set_title('2. White captured!\n(0 liberties)', fontsize=11, color=COLORS['text'])

    # Position 3: Snapback
    ax = axes[2]
    for x, y in black_stones:
        draw_stone(ax, x, y, 'black')
    draw_marker(ax, 3, 3, 'circle', COLORS['highlight_good'])
    highlight_area(ax, black_stones, COLORS['highlight_bad'], 0.2)
    ax.set_title('3. Black recaptures!\n(SNAPBACK)', fontsize=11, color=COLORS['text'])

    plt.suptitle('Snapback: Sacrifice to Capture More', fontsize=14,
                color=COLORS['text'], y=1.02)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'snapback-pattern.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: snapback-pattern.png")


def generate_ladder():
    """Generate ladder pattern illustration."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=COLORS['bg'])

    # Working ladder
    ax = axes[0]
    setup_board_axes(ax, size=9, show_coords=True)

    # Initial position
    draw_stone(ax, 2, 6, 'black', label='1')
    draw_stone(ax, 3, 6, 'white')
    draw_stone(ax, 2, 5, 'white')

    # Ladder sequence
    ladder_black = [(1, 5), (0, 4), (1, 3)]
    ladder_white = [(1, 6), (1, 4), (0, 3)]
    for i, (x, y) in enumerate(ladder_black):
        draw_stone(ax, x, y, 'black', label=3 + 2*i)
    for i, (x, y) in enumerate(ladder_white):
        draw_stone(ax, x, y, 'white', label=4 + 2*i)

    # Arrow showing direction
    ax.annotate('', xy=(0, 2), xytext=(2, 5),
               arrowprops=dict(arrowstyle='->', color=COLORS['highlight_bad'],
                              lw=2, ls='--'))
    ax.set_title('Working Ladder\n→ Captured at edge', fontsize=11, color=COLORS['text'])

    # Broken ladder
    ax = axes[1]
    setup_board_axes(ax, size=9, show_coords=True)

    draw_stone(ax, 2, 6, 'black', label='1')
    draw_stone(ax, 3, 6, 'white')
    draw_stone(ax, 2, 5, 'white')

    # Breaker stone!
    draw_stone(ax, 0, 3, 'black')
    highlight_area(ax, [(0, 3)], COLORS['highlight_good'], 0.4)
    ax.text(0, 2.2, 'BREAKER', ha='center', fontsize=9, fontweight='bold',
           color=COLORS['highlight_good'])

    ax.annotate('', xy=(0, 3.5), xytext=(2, 5.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['highlight_good'],
                              lw=2, ls='--'))
    ax.set_title('Broken Ladder\n→ Escapes via breaker', fontsize=11, color=COLORS['text'])

    plt.suptitle('Ladder: Chase to the Edge (or Escape)', fontsize=14,
                color=COLORS['text'], y=1.02)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'ladder-pattern.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: ladder-pattern.png")


def generate_ownership_map():
    """Generate KataGo-style ownership visualization with colorful heatmap."""
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor=COLORS['bg'])

    size = 9

    # Create ownership values: -1 (Black) to +1 (White)
    np.random.seed(42)
    ownership = np.zeros((size, size))

    # Black territory (top-left corner) - strong negative
    ownership[6:9, 0:3] = -0.9 + np.random.random((3, 3)) * 0.15
    ownership[5:7, 0:2] = -0.7 + np.random.random((2, 2)) * 0.2

    # White territory (bottom-right) - strong positive
    ownership[0:3, 6:9] = 0.85 + np.random.random((3, 3)) * 0.1
    ownership[0:2, 5:7] = 0.7 + np.random.random((2, 2)) * 0.2

    # White influence (right side)
    ownership[3:6, 7:9] = 0.5 + np.random.random((3, 2)) * 0.3

    # Black influence (left side)
    ownership[4:7, 0:2] = -0.5 + np.random.random((3, 2)) * 0.3

    # Contested center - near zero with noise
    ownership[3:6, 3:6] = np.random.random((3, 3)) * 0.4 - 0.2

    # Smooth it out
    ownership = gaussian_filter(ownership, sigma=0.8)

    # KataGo-style colormap: Blue (Black) -> White (neutral) -> Yellow/Orange (White)
    colors_list = ['#1a237e', '#3949ab', '#7986cb', '#e8e8e8',
                   '#ffcc80', '#ffa726', '#ef6c00']
    katago_cmap = LinearSegmentedColormap.from_list('katago', colors_list, N=256)

    # Draw board background
    ax.set_facecolor('#dcb35c')  # Wood color

    # Draw ownership heatmap
    im = ax.imshow(ownership, cmap=katago_cmap, vmin=-1, vmax=1,
                   extent=[-0.5, size-0.5, -0.5, size-0.5],
                   origin='lower', alpha=0.75, zorder=1)

    # Grid lines
    for i in range(size):
        ax.axhline(y=i, color='#4a4036', linewidth=0.5, alpha=0.5, zorder=2)
        ax.axvline(x=i, color='#4a4036', linewidth=0.5, alpha=0.5, zorder=2)

    # Star points
    star_points = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
    for x, y in star_points:
        circle = patches.Circle((x, y), 0.08, color='#4a4036', zorder=3)
        ax.add_patch(circle)

    # Place stones
    black_positions = [(0, 7), (1, 7), (0, 8), (1, 8), (2, 6), (3, 5), (4, 4)]
    white_positions = [(7, 0), (7, 1), (8, 0), (8, 1), (6, 2), (5, 3), (5, 4)]

    for x, y in black_positions:
        circle = patches.Circle((x, y), 0.4, facecolor='#1a1a1a',
                                edgecolor='#333', linewidth=1, zorder=4)
        ax.add_patch(circle)
    for x, y in white_positions:
        circle = patches.Circle((x, y), 0.4, facecolor='#f5f5f5',
                                edgecolor='#666', linewidth=1, zorder=4)
        ax.add_patch(circle)

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Black', 'Neutral', 'White'])
    cbar.ax.tick_params(labelsize=10)

    ax.set_title('Ownership Predictions\n361 signals per position (vs 1 bit for win/loss)',
                fontsize=13, color=COLORS['text'], pad=15)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'ownership-map.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: ownership-map.png")


def generate_feature_sparsity():
    """Generate feature sparsity heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor=COLORS['bg'])

    # Simulate feature plane densities
    np.random.seed(42)

    # Dense features (stone positions, history)
    dense = np.random.random((9, 9)) * 0.8 + 0.2
    im1 = axes[0].imshow(dense, cmap='YlOrBr', vmin=0, vmax=1)
    axes[0].set_title('Basic Features\n(stones, history)\n~40% filled', fontsize=11)
    axes[0].axis('off')

    # Sparse features (liberties)
    sparse = np.zeros((9, 9))
    sparse[2:4, 3:5] = np.random.random((2, 2)) * 0.5 + 0.3
    sparse[6:7, 6:8] = np.random.random((1, 2)) * 0.4 + 0.2
    im2 = axes[1].imshow(sparse, cmap='YlOrBr', vmin=0, vmax=1)
    axes[1].set_title('Liberty Features\n(1-3 liberties)\n~5% filled', fontsize=11)
    axes[1].axis('off')

    # Very sparse features (captures)
    very_sparse = np.zeros((9, 9))
    very_sparse[4, 4] = 0.8
    im3 = axes[2].imshow(very_sparse, cmap='YlOrBr', vmin=0, vmax=1)
    axes[2].set_title('Tactical Features\n(captures, atari)\n~1% filled', fontsize=11)
    axes[2].axis('off')

    plt.suptitle('Feature Sparsity: The Network Sees Mostly Zeros',
                fontsize=14, color=COLORS['text'], y=1.02)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'feature-sparsity.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: feature-sparsity.png")


def generate_accuracy_comparison():
    """Generate accuracy comparison chart."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor=COLORS['bg'])

    categories = ['Basic\n(17 planes)', 'Tactical\n(27 planes)', 'With Ownership\n(+ aux targets)']
    values = [40, 21, 38]
    colors = [COLORS['highlight_blue'], COLORS['highlight_bad'], COLORS['highlight_good']]

    bars = ax.bar(categories, values, color=colors, edgecolor=COLORS['grid'], linewidth=2)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val}%', ha='center', fontsize=14, fontweight='bold',
               color=COLORS['text'])

    # Arrow showing the problem
    ax.annotate('', xy=(1, 25), xytext=(0, 36),
               arrowprops=dict(arrowstyle='->', color=COLORS['highlight_bad'],
                              lw=2))
    ax.text(0.5, 32, 'More features,\nless learning!', fontsize=10,
           color=COLORS['highlight_bad'], ha='center')

    # Arrow showing the solution
    ax.annotate('', xy=(2, 35), xytext=(1, 25),
               arrowprops=dict(arrowstyle='->', color=COLORS['highlight_good'],
                              lw=2))
    ax.text(1.7, 28, 'Ownership\nfixes it!', fontsize=10,
           color=COLORS['highlight_good'], ha='center')

    ax.set_ylabel('Policy Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 50)
    ax.set_facecolor(COLORS['bg'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title('Policy Accuracy: The Sparse Feature Problem',
                fontsize=14, color=COLORS['text'], pad=10)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'accuracy-comparison.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: accuracy-comparison.png")


def generate_tactical_results():
    """Generate tactical results comparison chart."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor=COLORS['bg'])

    categories = ['Capture', 'Escape', 'Ladder', 'Snapback', 'Connect', 'Cut', 'TOTAL']
    before = [2, 1, 0, 0, 1, 0, 5]
    after = [3, 2, 1, 1, 1, 0, 8]
    max_vals = [3, 2, 2, 1, 1, 1, 10]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, before, width, label='Before (Neural only)',
                   color=COLORS['highlight_bad'], alpha=0.7,
                   edgecolor=COLORS['grid'], linewidth=1.5)
    bars2 = ax.bar(x + width/2, after, width, label='After (Neurosymbolic)',
                   color=COLORS['highlight_good'], alpha=0.9,
                   edgecolor=COLORS['grid'], linewidth=1.5)

    # Add value labels
    for bar, val, maxv in zip(bars1, before, max_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val}/{maxv}', ha='center', fontsize=9, color=COLORS['text'])
    for bar, val, maxv in zip(bars2, after, max_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val}/{maxv}', ha='center', fontsize=9, fontweight='bold',
               color=COLORS['text'])

    ax.set_ylabel('Correct Decisions', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 12)
    ax.legend(loc='upper left')
    ax.set_facecolor(COLORS['bg'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Improvement annotation
    ax.annotate('+60%', xy=(6.3, 8.5), fontsize=20, fontweight='bold',
               color=COLORS['highlight_good'])

    ax.set_title('Tactical Test Results: Before & After Hybrid System',
                fontsize=14, color=COLORS['text'], pad=10)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'tactical-results.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: tactical-results.png")


def generate_hybrid_architecture():
    """Generate hybrid architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor=COLORS['bg'])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Neural Network box
    nn_box = FancyBboxPatch((1, 5), 3.5, 2, boxstyle="round,pad=0.1",
                            facecolor=COLORS['highlight_blue'], alpha=0.3,
                            edgecolor=COLORS['highlight_blue'], linewidth=2)
    ax.add_patch(nn_box)
    ax.text(2.75, 6.3, 'Neural Network', ha='center', fontsize=12, fontweight='bold',
           color=COLORS['text'])
    ax.text(2.75, 5.7, 'Pattern Recognition', ha='center', fontsize=10,
           color=COLORS['text'], style='italic')
    ax.text(2.75, 5.2, '"This looks like a ladder"', ha='center', fontsize=9,
           color=COLORS['highlight_blue'])

    # Tactical Analyzer box
    ta_box = FancyBboxPatch((5.5, 5), 3.5, 2, boxstyle="round,pad=0.1",
                            facecolor=COLORS['highlight_good'], alpha=0.3,
                            edgecolor=COLORS['highlight_good'], linewidth=2)
    ax.add_patch(ta_box)
    ax.text(7.25, 6.3, 'Tactical Analyzer', ha='center', fontsize=12, fontweight='bold',
           color=COLORS['text'])
    ax.text(7.25, 5.7, 'Symbolic Verification', ha='center', fontsize=10,
           color=COLORS['text'], style='italic')
    ax.text(7.25, 5.2, '"Let me verify: B1, W2..."', ha='center', fontsize=9,
           color=COLORS['highlight_good'])

    # Arrow between boxes
    ax.annotate('', xy=(5.3, 6), xytext=(4.7, 6),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))

    # Input
    ax.text(2.75, 4.3, 'Board State', ha='center', fontsize=10, color=COLORS['text'])
    ax.annotate('', xy=(2.75, 4.8), xytext=(2.75, 4.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=1.5))

    # Outputs
    output_box = FancyBboxPatch((3, 1.5), 4, 1.5, boxstyle="round,pad=0.1",
                                facecolor=COLORS['highlight_yellow'], alpha=0.2,
                                edgecolor=COLORS['highlight_yellow'], linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 2.5, 'Verified Move', ha='center', fontsize=12, fontweight='bold',
           color=COLORS['text'])
    ax.text(5, 1.9, 'Policy + Tactical Confidence', ha='center', fontsize=10,
           color=COLORS['text'])

    ax.annotate('', xy=(5, 3.2), xytext=(7.25, 4.8),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=1.5))

    ax.set_title('Neurosymbolic Hybrid: Intuition + Calculation',
                fontsize=14, color=COLORS['text'], pad=10)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'hybrid-architecture.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: hybrid-architecture.png")


def generate_capture_rule():
    """Generate simple capture rule illustration."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), facecolor=COLORS['bg'])

    for ax in axes:
        setup_board_axes(ax, size=5)

    # Step 1: Surrounding
    ax = axes[0]
    draw_stone(ax, 2, 2, 'black')
    draw_stone(ax, 1, 2, 'white')
    draw_stone(ax, 2, 1, 'white')
    draw_stone(ax, 3, 2, 'white')
    draw_marker(ax, 2, 3, 'x', COLORS['highlight_bad'])
    ax.set_title('1. One liberty left', fontsize=11, color=COLORS['text'])

    # Step 2: Surrounded
    ax = axes[1]
    draw_stone(ax, 2, 2, 'black')
    draw_stone(ax, 1, 2, 'white')
    draw_stone(ax, 2, 1, 'white')
    draw_stone(ax, 3, 2, 'white')
    draw_stone(ax, 2, 3, 'white', label='1')
    highlight_area(ax, [(2, 2)], COLORS['highlight_bad'], 0.4)
    ax.set_title('2. Surrounded!', fontsize=11, color=COLORS['text'])

    # Step 3: Captured
    ax = axes[2]
    draw_stone(ax, 1, 2, 'white')
    draw_stone(ax, 2, 1, 'white')
    draw_stone(ax, 3, 2, 'white')
    draw_stone(ax, 2, 3, 'white')
    draw_marker(ax, 2, 2, 'circle', COLORS['highlight_good'])
    ax.set_title('3. Captured!', fontsize=11, color=COLORS['text'])

    plt.suptitle('The One Rule: Surround to Capture', fontsize=14,
                color=COLORS['text'], y=1.02)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'capture-rule.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: capture-rule.png")


def generate_all():
    """Generate all illustrations."""
    print("\n📸 Generating blog illustrations...\n")

    generate_capture_rule()
    generate_wall_problem()
    generate_mcts_tree()
    generate_feature_sparsity()
    generate_accuracy_comparison()
    generate_ownership_map()
    generate_snapback()
    generate_ladder()
    generate_tactical_results()
    generate_hybrid_architecture()

    print("\n✅ All illustrations generated!")
    print(f"   Location: {IMAGES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate blog illustrations")
    parser.add_argument("--post", help="Generate for specific post")
    args = parser.parse_args()

    generate_all()
