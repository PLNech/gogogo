#!/usr/bin/env python3
"""
Generate visualizations for GoGoGo blog posts
Creates heatmaps, policy distributions, and board states
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "blog" / "images"
OUTPUT_DIR.mkdir(exist_ok=True)

def draw_goban(ax, size=5, title=""):
    """Draw empty Go board"""
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Draw grid
    for i in range(size):
        ax.plot([i, i], [0, size-1], 'k-', linewidth=0.5)
        ax.plot([0, size-1], [i, i], 'k-', linewidth=0.5)

    # Mark star points (corners)
    if size >= 5:
        ax.plot([0, 0, size-1, size-1], [0, size-1, 0, size-1],
                'o', markersize=4, color='black')

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.invert_yaxis()
    ax.grid(False)

def place_stones(ax, blacks, whites):
    """Place stones on the board"""
    for (row, col) in blacks:
        circle = plt.Circle((col, row), 0.4, color='black', zorder=10)
        ax.add_patch(circle)

    for (row, col) in whites:
        circle = plt.Circle((col, row), 0.4, color='white',
                          edgecolor='black', linewidth=1.5, zorder=10)
        ax.add_patch(circle)

def visualize_policy_heatmap():
    """Generate policy prior heatmap visualization"""
    # Simulate policy priors (higher in corners, lower in center)
    size = 5
    priors = np.zeros((size, size))

    # Corner bonus
    priors[0, 0] = 0.25
    priors[0, 4] = 0.22
    priors[4, 0] = 0.23
    priors[4, 4] = 0.20

    # Edges
    priors[0, 2] = 0.03
    priors[2, 0] = 0.03
    priors[2, 4] = 0.02
    priors[4, 2] = 0.02

    # Center
    priors[2, 2] = 0.00

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    im = ax.imshow(priors, cmap='YlOrRd', vmin=0, vmax=0.25)
    ax.set_title('Policy Network Move Priors', fontsize=16, fontweight='bold')

    # Add grid
    for i in range(size):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
        ax.axvline(i - 0.5, color='black', linewidth=0.5)

    # Add text values
    for i in range(size):
        for j in range(size):
            if priors[i, j] > 0:
                text = ax.text(j, i, f'{priors[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=12,
                             fontweight='bold')

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Prior Probability', fontsize=12)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'policy_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Generated {output_path}")
    plt.close()

def visualize_liberty_planes():
    """Generate liberty planes visualization"""
    size = 5

    # Create 4 example planes
    plane_1lib = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],  # Stone in atari
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    plane_2lib = np.array([
        [1, 0, 0, 0, 0],  # Corner stone with 2 liberties
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    plane_3lib = np.array([
        [0, 1, 0, 0, 0],  # Edge stone with 3 liberties
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    plane_4lib = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],  # Center stone with 4 liberties
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    planes = [plane_1lib, plane_2lib, plane_3lib, plane_4lib]
    titles = ['1 Liberty (Atari)', '2 Liberties', '3 Liberties', '4 Liberties']

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for idx, (plane, title, ax) in enumerate(zip(planes, titles, axes)):
        im = ax.imshow(plane, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add grid
        for i in range(size):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
            ax.axvline(i - 0.5, color='black', linewidth=0.5)

        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_xlabel('Column', fontsize=10)
        ax.set_ylabel('Row', fontsize=10)

    plt.suptitle('Liberty Planes: Multi-layer Board Representation',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'liberty_planes.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Generated {output_path}")
    plt.close()

def visualize_capture_scenario():
    """Generate capture scenario visualization"""
    size = 5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Before capture
    draw_goban(ax1, size, "Before: Black in Atari")
    place_stones(ax1, [(1, 1)], [(0, 1), (1, 0), (1, 2)])

    # Mark the capture move
    ax1.plot([1], [2], 'r*', markersize=20, label='White captures here')
    ax1.legend(loc='upper right')

    # After capture
    draw_goban(ax2, size, "After: Black Captured")
    place_stones(ax2, [], [(0, 1), (1, 0), (1, 2), (2, 1)])

    # Show captured stone location
    circle = plt.Circle((1, 1), 0.4, color='lightgray',
                       linestyle='--', fill=False, linewidth=2)
    ax2.add_patch(circle)
    ax2.text(1, 1, 'X', ha='center', va='center',
            fontsize=20, color='red', fontweight='bold')

    plt.suptitle('Capture Detection in Policy Network',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'capture_scenario.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Generated {output_path}")
    plt.close()

def visualize_group_strength():
    """Generate group strength comparison"""
    size = 5

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Weak group (1 stone, center)
    draw_goban(axes[0], size, "Weak: 21% Strength")
    place_stones(axes[0], [(2, 2)], [])
    axes[0].text(2, 4, 'Single stone\n4 liberties\nNo position bonus',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

    # Medium group (corner, 3 stones)
    draw_goban(axes[1], size, "Strong: 48% Strength")
    place_stones(axes[1], [(0, 0), (0, 1), (1, 0)], [])
    axes[1].text(2, 4, 'Corner group\n6 liberties\nGood shape',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen'))

    # Atari group (very weak)
    draw_goban(axes[2], size, "Critical: <30% Strength")
    place_stones(axes[2], [(1, 1)], [(0, 0), (0, 1), (1, 0)])
    axes[2].plot([1], [2], 'r*', markersize=20, label='Last liberty')
    axes[2].text(2, 4, 'In atari\n1 liberty\nMust escape!',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral'))
    axes[2].legend(loc='upper right')

    plt.suptitle('Group Strength Assessment', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'group_strength.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Generated {output_path}")
    plt.close()

def main():
    print("\n🎨 Generating visualizations for blog posts...\n")

    visualize_policy_heatmap()
    visualize_liberty_planes()
    visualize_capture_scenario()
    visualize_group_strength()

    print(f"\n✓ All visualizations saved to {OUTPUT_DIR}\n")

if __name__ == "__main__":
    main()
