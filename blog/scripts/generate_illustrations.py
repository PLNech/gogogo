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
HERO_DIR = BLOG_DIR / "assets" / "hero"
HERO_DIR.mkdir(parents=True, exist_ok=True)

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


def generate_eight_basic_instincts():
    """Generate Sensei's 8 Basic Instincts board diagrams."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=COLORS['bg'])
    axes = axes.flatten()

    instincts = [
        ("1. Extend from Atari", "アタリから伸びよ", "extend_from_atari"),
        ("2. Hane vs Tsuke", "ツケにはハネ", "hane_tsuke"),
        ("3. Hane at Head of Two", "二子の頭にハネ", "hane_head_two"),
        ("4. Stretch from Kosumi", "コスミから伸びよ", "stretch_kosumi"),
        ("5. Block the Angle", "カケにはオサエ", "block_angle"),
        ("6. Connect vs Peep", "ノゾキにはツギ", "connect_peep"),
        ("7. Block the Thrust", "ツキアタリには", "block_thrust"),
        ("8. Stretch from Bump", "ブツカリから伸びよ", "stretch_bump"),
    ]

    for i, (title, japanese, pattern) in enumerate(instincts):
        ax = axes[i]
        setup_board_axes(ax, size=7)

        if pattern == "extend_from_atari":
            # Black stone in atari (one liberty)
            draw_stone(ax, 3, 3, 'black')
            draw_stone(ax, 2, 3, 'white')
            draw_stone(ax, 4, 3, 'white')
            draw_stone(ax, 3, 2, 'white')
            # Correct response: extend at 3,4
            draw_marker(ax, 3, 4, 'square', COLORS['highlight_good'])
            ax.text(3, 4.7, 'EXTEND', ha='center', fontsize=8,
                   color=COLORS['highlight_good'], fontweight='bold')

        elif pattern == "hane_tsuke":
            # Black stone, white attaches (tsuke)
            draw_stone(ax, 3, 3, 'black')
            draw_stone(ax, 4, 3, 'white', label='1')  # Tsuke
            # Correct response: hane at 4,4
            draw_marker(ax, 4, 4, 'square', COLORS['highlight_good'])
            ax.text(4, 4.7, 'HANE', ha='center', fontsize=8,
                   color=COLORS['highlight_good'], fontweight='bold')

        elif pattern == "hane_head_two":
            # Two white stones in a row
            draw_stone(ax, 3, 3, 'white')
            draw_stone(ax, 4, 3, 'white')
            # Black should play above the head
            draw_marker(ax, 5, 3, 'square', COLORS['highlight_good'])
            ax.text(5, 3.7, 'HANE', ha='center', fontsize=8,
                   color=COLORS['highlight_good'], fontweight='bold')

        elif pattern == "stretch_kosumi":
            # Black stone, white plays kosumi-tsuke (diagonal attach)
            draw_stone(ax, 3, 3, 'black')
            draw_stone(ax, 4, 4, 'white', label='1')  # Kosumi-tsuke
            # Correct response: stretch away
            draw_marker(ax, 2, 3, 'square', COLORS['highlight_good'])
            ax.text(2, 3.7, 'STRETCH', ha='center', fontsize=8,
                   color=COLORS['highlight_good'], fontweight='bold')

        elif pattern == "block_angle":
            # White plays angle attack (kake)
            draw_stone(ax, 3, 3, 'black')
            draw_stone(ax, 4, 4, 'white', label='1')  # Angle play
            # Block diagonally
            draw_marker(ax, 4, 3, 'square', COLORS['highlight_good'])
            ax.text(4.7, 3, 'BLOCK', ha='center', fontsize=8,
                   color=COLORS['highlight_good'], fontweight='bold')

        elif pattern == "connect_peep":
            # Two black stones with cutting point, white peeps
            draw_stone(ax, 2, 3, 'black')
            draw_stone(ax, 4, 3, 'black')
            draw_stone(ax, 3, 4, 'white', label='1')  # Peep
            # Must connect!
            draw_marker(ax, 3, 3, 'square', COLORS['highlight_good'])
            ax.text(3, 2.3, 'CONNECT', ha='center', fontsize=8,
                   color=COLORS['highlight_good'], fontweight='bold')

        elif pattern == "block_thrust":
            # Two black stones, white thrusts between
            draw_stone(ax, 2, 3, 'black')
            draw_stone(ax, 4, 3, 'black')
            draw_stone(ax, 3, 2, 'white', label='1')  # Thrust
            # Block the thrust
            draw_marker(ax, 3, 3, 'square', COLORS['highlight_good'])
            ax.text(3, 3.7, 'BLOCK', ha='center', fontsize=8,
                   color=COLORS['highlight_good'], fontweight='bold')

        elif pattern == "stretch_bump":
            # Black stone, white bumps (supported attachment)
            draw_stone(ax, 3, 3, 'black')
            draw_stone(ax, 4, 3, 'white', label='1')  # Bump
            draw_stone(ax, 5, 3, 'white')  # Support stone
            # Stretch, don't hane
            draw_marker(ax, 2, 3, 'square', COLORS['highlight_good'])
            ax.text(2, 3.7, 'STRETCH', ha='center', fontsize=8,
                   color=COLORS['highlight_good'], fontweight='bold')

        ax.set_title(f'{title}\n{japanese}', fontsize=10, color=COLORS['text'])

    plt.suptitle("Sensei's 8 Basic Instincts\nPatterns masters play without thinking",
                fontsize=14, color=COLORS['text'], y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'eight-basic-instincts.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: eight-basic-instincts.png")


def generate_journey_begins():
    """Generate illustration for 'A Journey Begins' post."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor=COLORS['bg'])
    setup_board_axes(ax, size=9)

    # Empty board with just one black stone - the first move
    draw_stone(ax, 2, 6, 'black', label='1')

    ax.set_title('The First Stone\n\n一石投じる — To cast the first stone',
                fontsize=14, color=COLORS['text'], pad=15)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'journey-begins.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: journey-begins.png")


def generate_from_python_to_browser():
    """Generate illustration for Python to Browser export."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor=COLORS['bg'])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # PyTorch box
    pytorch_box = FancyBboxPatch((0.5, 3.5), 2, 1.5, boxstyle="round,pad=0.1",
                                  facecolor='#ee4c2c', alpha=0.3,
                                  edgecolor='#ee4c2c', linewidth=2)
    ax.add_patch(pytorch_box)
    ax.text(1.5, 4.5, 'PyTorch', ha='center', fontsize=12, fontweight='bold')
    ax.text(1.5, 3.9, 'Training', ha='center', fontsize=9)

    # Arrow
    ax.annotate('', xy=(3, 4.25), xytext=(2.7, 4.25),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))

    # ONNX box
    onnx_box = FancyBboxPatch((3.2, 3.5), 2, 1.5, boxstyle="round,pad=0.1",
                               facecolor='#005CED', alpha=0.3,
                               edgecolor='#005CED', linewidth=2)
    ax.add_patch(onnx_box)
    ax.text(4.2, 4.5, 'ONNX', ha='center', fontsize=12, fontweight='bold')
    ax.text(4.2, 3.9, 'Portable', ha='center', fontsize=9)

    # Arrow
    ax.annotate('', xy=(5.7, 4.25), xytext=(5.4, 4.25),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))

    # TensorFlow.js box
    tfjs_box = FancyBboxPatch((5.9, 3.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                               facecolor='#ff6f00', alpha=0.3,
                               edgecolor='#ff6f00', linewidth=2)
    ax.add_patch(tfjs_box)
    ax.text(7.15, 4.5, 'TensorFlow.js', ha='center', fontsize=12, fontweight='bold')
    ax.text(7.15, 3.9, 'Browser', ha='center', fontsize=9)

    # Browser window below
    browser_box = FancyBboxPatch((2.5, 0.5), 5, 2.2, boxstyle="round,pad=0.1",
                                  facecolor=COLORS['highlight_good'], alpha=0.2,
                                  edgecolor=COLORS['highlight_good'], linewidth=2)
    ax.add_patch(browser_box)
    ax.text(5, 2.2, '🌐 Browser', ha='center', fontsize=14, fontweight='bold')
    ax.text(5, 1.5, 'model.predict(boardState)', ha='center', fontsize=11,
           fontfamily='monospace')
    ax.text(5, 0.9, 'No server needed!', ha='center', fontsize=10,
           color=COLORS['highlight_good'], style='italic')

    # Arrow down
    ax.annotate('', xy=(7.15, 2.9), xytext=(7.15, 3.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=2))

    ax.set_title('From Python to Browser\nTrain once, deploy anywhere',
                fontsize=14, color=COLORS['text'], pad=10)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'python-to-browser.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: python-to-browser.png")


def generate_footsteps_research():
    """Generate illustration for research survey post."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor=COLORS['bg'])

    # AlphaGo/AlphaZero
    ax = axes[0]
    setup_board_axes(ax, size=5)
    # Iconic move 37 style position
    draw_stone(ax, 2, 2, 'black')
    draw_stone(ax, 1, 3, 'white')
    draw_stone(ax, 3, 1, 'white')
    draw_stone(ax, 2, 4, 'black', label='37')
    ax.set_title('AlphaGo (2016)\nDeep Learning + MCTS', fontsize=11, color=COLORS['text'])

    # KataGo multi-size
    ax = axes[1]
    setup_board_axes(ax, size=5)
    # Show different size concept
    for i in range(5):
        ax.axhline(y=i, color=COLORS['grid'], linewidth=0.5, alpha=0.3)
        ax.axvline(x=i, color=COLORS['grid'], linewidth=0.5, alpha=0.3)
    ax.text(2, 2, '5×5\n9×9\n19×19', ha='center', va='center', fontsize=10,
           color=COLORS['highlight_blue'], fontweight='bold')
    ax.set_title('KataGo (2019)\nMulti-size Training', fontsize=11, color=COLORS['text'])

    # GNN approach
    ax = axes[2]
    ax.set_facecolor(COLORS['bg'])
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.axis('off')
    # Draw graph nodes
    nodes = [(1, 2), (2, 1), (2, 3), (3, 2), (2.5, 4), (4, 3)]
    for x, y in nodes:
        circle = patches.Circle((x, y), 0.3, facecolor=COLORS['highlight_good'],
                                edgecolor=COLORS['grid'], linewidth=1.5)
        ax.add_patch(circle)
    # Draw edges
    edges = [((1, 2), (2, 1)), ((1, 2), (2, 3)), ((2, 1), (3, 2)),
             ((2, 3), (3, 2)), ((2, 3), (2.5, 4)), ((3, 2), (4, 3))]
    for (x1, y1), (x2, y2) in edges:
        ax.plot([x1, x2], [y1, y2], '-', color=COLORS['grid'], linewidth=1.5)
    ax.set_title('AlphaGateau (2024)\nGraph Neural Networks', fontsize=11, color=COLORS['text'])

    plt.suptitle('Standing on Shoulders\nFrom AlphaGo to Graph Networks',
                fontsize=14, color=COLORS['text'], y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'research-evolution.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: research-evolution.png")


def generate_tester_cest_douter():
    """Generate illustration for testing reflection post."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=COLORS['bg'])

    # Left: Untested code (sand castle)
    ax = axes[0]
    ax.set_facecolor(COLORS['bg'])
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Stack of blocks (unstable)
    for i, (x, y, w, h) in enumerate([
        (1.5, 0.5, 2, 0.8), (1.7, 1.3, 1.6, 0.7), (1.9, 2, 1.2, 0.6),
        (2.1, 2.6, 0.8, 0.5)
    ]):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                               facecolor=COLORS['highlight_bad'], alpha=0.3 + i*0.15,
                               edgecolor=COLORS['highlight_bad'], linewidth=1.5)
        ax.add_patch(rect)

    # Question marks
    ax.text(3.5, 3, '?', fontsize=40, color=COLORS['highlight_bad'], alpha=0.5)
    ax.text(0.8, 2.5, '?', fontsize=30, color=COLORS['highlight_bad'], alpha=0.5)

    ax.set_title('Without Tests\n"Castle of Sand"', fontsize=12, color=COLORS['text'])

    # Right: Tested code (solid foundation)
    ax = axes[1]
    ax.set_facecolor(COLORS['bg'])
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Solid foundation with checkmarks
    for i, (x, y, w, h) in enumerate([
        (1, 0.5, 3, 0.8), (1.2, 1.4, 2.6, 0.7), (1.4, 2.2, 2.2, 0.6),
        (1.6, 2.9, 1.8, 0.5)
    ]):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                               facecolor=COLORS['highlight_good'], alpha=0.3 + i*0.15,
                               edgecolor=COLORS['highlight_good'], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w + 0.2, y + h/2, '✓', fontsize=16,
               color=COLORS['highlight_good'], va='center')

    ax.set_title('With Tests\n"Concrete Foundation"', fontsize=12, color=COLORS['text'])

    plt.suptitle('Tester c\'est Douter\nTo test is to doubt — and doubt is wisdom',
                fontsize=14, color=COLORS['text'], y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'tester-cest-douter.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: tester-cest-douter.png")


def generate_learning_to_see():
    """Generate illustration for 'Learning to See' post - benchmark results."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor=COLORS['bg'])

    instincts = [
        'Extend\nfrom Atari', 'Hane vs\nTsuke', 'Hane at\nHead of 2',
        'Stretch from\nKosumi', 'Block the\nAngle', 'Connect vs\nPeep',
        'Block the\nThrust', 'Stretch from\nBump'
    ]

    # Benchmark results (model vs random)
    model_scores = [0, 0, 5.6, 0, 5.6, 0, 0, 0]  # ~1.3% overall
    random_baseline = [12.5] * 8  # 1/8 random chance

    x = np.arange(len(instincts))
    width = 0.35

    bars1 = ax.bar(x - width/2, model_scores, width, label='Our Model',
                   color=COLORS['highlight_bad'], alpha=0.8,
                   edgecolor=COLORS['grid'], linewidth=1.5)
    bars2 = ax.bar(x + width/2, random_baseline, width, label='Random Baseline',
                   color=COLORS['text'], alpha=0.3,
                   edgecolor=COLORS['grid'], linewidth=1.5)

    ax.axhline(y=100, color=COLORS['highlight_good'], linestyle='--',
              linewidth=2, alpha=0.5, label='Perfect')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(instincts, fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right')
    ax.set_facecolor(COLORS['bg'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Overall score annotation
    ax.text(3.5, 90, 'Overall: 1.3%', fontsize=16, fontweight='bold',
           color=COLORS['highlight_bad'], ha='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title("Learning to See: 8 Basic Instincts Benchmark\n"
                "Before we run, we crawl",
                fontsize=14, color=COLORS['text'], pad=10)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'learning-to-see-benchmark.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print("Generated: learning-to-see-benchmark.png")


def generate_hero_image(slug, board_pattern='random', title_overlay=None):
    """Generate a hero image for a post with Go board aesthetic."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), facecolor=COLORS['board'])

    # Create wide board view (cropped top portion of 19x19)
    size = 19
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 7.5, size - 0.5)  # Show top 7 rows
    ax.set_facecolor(COLORS['board'])

    # Grid lines
    for i in range(size):
        ax.axhline(y=i, color=COLORS['grid'], linewidth=0.5, alpha=0.4)
        ax.axvline(x=i, color=COLORS['grid'], linewidth=0.5, alpha=0.4)

    # Star points visible in top portion
    for x, y in [(3, 15), (9, 15), (15, 15)]:
        circle = patches.Circle((x, y), 0.12, color=COLORS['grid'], alpha=0.6)
        ax.add_patch(circle)

    np.random.seed(hash(slug) % 2**32)

    if board_pattern == 'opening':
        # Classic opening pattern
        stones = [
            ('black', 3, 16), ('white', 15, 16), ('black', 16, 3),
            ('white', 3, 3), ('black', 9, 16), ('white', 16, 9),
        ]
    elif board_pattern == 'battle':
        # Mid-game fighting
        stones = []
        for _ in range(15):
            x, y = np.random.randint(2, 17), np.random.randint(12, 18)
            color = 'black' if np.random.random() > 0.5 else 'white'
            stones.append((color, x, y))
    elif board_pattern == 'instincts':
        # Show some instinct patterns scattered
        stones = [
            ('black', 4, 15), ('white', 5, 15), ('white', 4, 14), ('white', 3, 15),  # Atari
            ('black', 10, 16), ('white', 11, 16),  # Tsuke
            ('black', 14, 15), ('black', 16, 15), ('white', 15, 16),  # Peep
        ]
    elif board_pattern == 'research':
        # Sparse, contemplative
        stones = [
            ('black', 3, 16), ('white', 15, 16), ('black', 16, 14),
        ]
    elif board_pattern == 'code':
        # Grid-like pattern suggesting code/structure
        stones = []
        for i in range(3, 16, 4):
            for j in range(13, 18, 2):
                if np.random.random() > 0.3:
                    color = 'black' if (i + j) % 2 == 0 else 'white'
                    stones.append((color, i, j))
    else:  # random
        stones = []
        for _ in range(8):
            x, y = np.random.randint(1, 18), np.random.randint(12, 18)
            color = 'black' if np.random.random() > 0.5 else 'white'
            stones.append((color, x, y))

    for color, x, y in stones:
        if 12 <= y <= 18:  # Only draw if visible
            draw_stone(ax, x, y, color)

    ax.axis('off')

    plt.tight_layout(pad=0)
    plt.savefig(HERO_DIR / f'{slug}.png', dpi=100, facecolor=COLORS['board'],
                bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Generated hero: {slug}.png")


def generate_all_heroes():
    """Generate hero images for all posts."""
    print("\n🖼️ Generating hero images...\n")

    heroes = [
        ('a-journey-begins', 'opening'),
        ('first-steps', 'opening'),
        ('the-wall', 'battle'),
        ('standing-on-shoulders', 'research'),
        ('neurosymbolic-harmony', 'battle'),
        ('eight-instincts', 'instincts'),
        ('from-python-to-browser', 'code'),
        ('footsteps-of-giants', 'research'),
        ('learning-to-see', 'instincts'),
        ('tester-cest-douter', 'code'),
        ('adaptive-curriculum', 'battle'),
        ('the-gift-of-sight', 'battle'),
    ]

    for slug, pattern in heroes:
        generate_hero_image(slug, pattern)

    print(f"\n✅ Hero images generated in {HERO_DIR}")


def generate_all():
    """Generate all illustrations."""
    print("\n📸 Generating blog illustrations...\n")

    # Core illustrations
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

    # Sensei's 8 Basic Instincts
    generate_eight_basic_instincts()
    generate_learning_to_see()

    # Post-specific illustrations
    generate_journey_begins()
    generate_from_python_to_browser()
    generate_footsteps_research()
    generate_tester_cest_douter()

    # Hero images for all posts
    generate_all_heroes()

    print("\n✅ All illustrations generated!")
    print(f"   Location: {IMAGES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate blog illustrations")
    parser.add_argument("--post", help="Generate for specific post")
    args = parser.parse_args()

    generate_all()
