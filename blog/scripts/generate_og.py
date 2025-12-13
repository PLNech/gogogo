#!/usr/bin/env python3
"""
OG Image Generator for GoGoGo Blog
Creates Go-board inspired Open Graph images for blog posts.

Usage:
    python generate_og.py                    # Generate all missing OG images
    python generate_og.py --post "title"     # Generate for specific post
    python generate_og.py --main             # Generate main site OG image
"""

import argparse
import hashlib
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# Paths
BLOG_DIR = Path(__file__).parent.parent
POSTS_DIR = BLOG_DIR / "_posts"
OG_DIR = BLOG_DIR / "assets" / "og"

# OG image dimensions (1200x630 is optimal for most platforms)
OG_WIDTH = 1200
OG_HEIGHT = 630
DPI = 100

# Colors - warm paper aesthetic
COLORS = {
    'board': '#DEB887',      # Burlywood - traditional goban
    'board_dark': '#C4A574', # Darker wood grain
    'grid': '#4A4036',       # Dark brown grid
    'black': '#1a1a1a',      # Black stones
    'white': '#f5f5f5',      # White stones
    'text': '#2d2a26',       # Dark text
    'text_light': '#6b6560', # Light text
    'accent': '#8B4513',     # Saddle brown accent
    'overlay': '#00000020',  # Subtle overlay
}

# Phase colors for visual distinction
PHASE_COLORS = {
    'Genesis': '#4a7c59',    # Forest green
    'Foundation': '#5c6b73', # Slate
    'Struggle': '#9e4244',   # Burgundy
    'Research': '#4a6fa5',   # Steel blue
    'Synthesis': '#7d5a50',  # Warm brown
}


def create_board_pattern(ax, size=19, density=0.3):
    """Create a Go board background with some stones."""
    # Board background
    ax.set_facecolor(COLORS['board'])

    # Draw grid lines (partial, faded at edges for aesthetic)
    for i in range(size):
        alpha = 0.4 * (1 - abs(i - size/2) / (size/2) * 0.5)
        ax.axhline(y=i, color=COLORS['grid'], linewidth=0.5, alpha=alpha)
        ax.axvline(x=i, color=COLORS['grid'], linewidth=0.5, alpha=alpha)

    # Star points (hoshi)
    if size >= 9:
        star_points = []
        if size == 9:
            star_points = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        elif size == 13:
            star_points = [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
        elif size == 19:
            star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]

        for x, y in star_points:
            circle = patches.Circle((x, y), 0.15, color=COLORS['grid'], alpha=0.6)
            ax.add_patch(circle)

    # Add some random stones for visual interest
    np.random.seed(42)  # Consistent pattern
    n_stones = int(size * size * density)
    positions = np.random.choice(size * size, n_stones, replace=False)

    for pos in positions:
        x, y = pos % size, pos // size
        color = COLORS['black'] if np.random.random() > 0.5 else COLORS['white']
        alpha = 0.3 + 0.4 * np.random.random()

        circle = patches.Circle((x, y), 0.4, color=color, alpha=alpha)
        ax.add_patch(circle)

    ax.set_xlim(-1, size)
    ax.set_ylim(-1, size)
    ax.set_aspect('equal')
    ax.axis('off')


def add_analysis_overlay(ax, size=19):
    """Add neural network analysis aesthetic overlay."""
    # Heat map effect in corner (like policy output)
    np.random.seed(123)

    # Create a subtle gradient overlay in one corner
    for i in range(6):
        for j in range(6):
            intensity = (6 - i) * (6 - j) / 36
            alpha = intensity * 0.15
            rect = patches.Rectangle(
                (i - 0.5, j - 0.5), 1, 1,
                facecolor='#4a6fa5',
                alpha=alpha
            )
            ax.add_patch(rect)

    # Add some "analysis lines" connecting stones
    for _ in range(3):
        x1, y1 = np.random.randint(0, 8), np.random.randint(0, 8)
        x2, y2 = x1 + np.random.randint(1, 4), y1 + np.random.randint(1, 4)
        ax.plot([x1, x2], [y1, y2], color=COLORS['accent'],
                linewidth=1.5, alpha=0.4, linestyle='--')


def generate_og_image(title: str, phase: str = None, output_path: Path = None, is_main: bool = False):
    """Generate an OG image for a blog post or main site."""

    fig = plt.figure(figsize=(OG_WIDTH/DPI, OG_HEIGHT/DPI), dpi=DPI)

    # Create axes for the board (left side)
    ax_board = fig.add_axes([0, 0, 0.45, 1])
    create_board_pattern(ax_board, size=13, density=0.25)
    add_analysis_overlay(ax_board, size=13)

    # Create axes for text (right side)
    ax_text = fig.add_axes([0.42, 0, 0.58, 1])
    ax_text.set_facecolor(COLORS['board'])
    ax_text.axis('off')

    # Gradient overlay on text side
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    ax_text.imshow(gradient, extent=[0, 1, 0, 1], aspect='auto',
                   cmap='Oranges', alpha=0.1, zorder=0)

    # Title text
    if is_main:
        # Main site OG
        ax_text.text(0.5, 0.65, 'GoGoGo', fontsize=48, fontweight='bold',
                    color=COLORS['text'], ha='center', va='center',
                    fontfamily='serif')
        ax_text.text(0.5, 0.45, 'Chronicle', fontsize=36,
                    color=COLORS['text'], ha='center', va='center',
                    fontfamily='serif', style='italic')
        ax_text.text(0.5, 0.25, '一石万局', fontsize=24,
                    color=COLORS['text_light'], ha='center', va='center')
        ax_text.text(0.5, 0.15, 'One stone, ten thousand games', fontsize=14,
                    color=COLORS['text_light'], ha='center', va='center',
                    style='italic')
    else:
        # Post-specific OG
        # Wrap title if too long
        words = title.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 20:
                if len(current_line) > 1:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(' '.join(current_line))
                    current_line = []
        if current_line:
            lines.append(' '.join(current_line))

        title_text = '\n'.join(lines)

        ax_text.text(0.5, 0.55, title_text, fontsize=36, fontweight='bold',
                    color=COLORS['text'], ha='center', va='center',
                    fontfamily='serif', linespacing=1.3)

        # Phase badge
        if phase:
            phase_color = PHASE_COLORS.get(phase, COLORS['accent'])
            badge = patches.FancyBboxPatch(
                (0.3, 0.75), 0.4, 0.12,
                boxstyle="round,pad=0.02",
                facecolor=phase_color,
                alpha=0.9
            )
            ax_text.add_patch(badge)
            ax_text.text(0.5, 0.81, phase.upper(), fontsize=14,
                        color='white', ha='center', va='center',
                        fontweight='bold', fontfamily='sans-serif')

        # Site name
        ax_text.text(0.5, 0.12, 'GoGoGo Chronicle', fontsize=16,
                    color=COLORS['text_light'], ha='center', va='center',
                    fontfamily='serif', style='italic')

    # Add subtle border
    for spine in ax_text.spines.values():
        spine.set_visible(False)

    # Save
    if output_path is None:
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        output_path = OG_DIR / f"{slug}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0,
                facecolor=COLORS['board'])
    plt.close()

    # Optimize with PIL
    optimize_image(output_path)

    print(f"Generated: {output_path}")
    return output_path


def optimize_image(path: Path, max_size_kb: int = 150):
    """Optimize image for web - target under 150KB while maintaining quality."""
    img = Image.open(path)

    # Convert to RGB if necessary
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')

    # Resize if needed (maintain 1200x630 aspect)
    target_size = (1200, 630)
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)

    # Save with optimization, adjusting quality to hit target size
    quality = 85
    while quality > 30:
        img.save(path, 'PNG', optimize=True)
        size_kb = path.stat().st_size / 1024

        if size_kb <= max_size_kb:
            break

        # Try JPEG if PNG is too large
        jpeg_path = path.with_suffix('.jpg')
        img.save(jpeg_path, 'JPEG', quality=quality, optimize=True)
        jpeg_size = jpeg_path.stat().st_size / 1024

        if jpeg_size < size_kb:
            path.unlink()
            jpeg_path.rename(path.with_suffix('.jpg'))
            print(f"  Converted to JPEG ({jpeg_size:.1f}KB)")
            return
        else:
            jpeg_path.unlink()

        quality -= 10

    size_kb = path.stat().st_size / 1024
    print(f"  Size: {size_kb:.1f}KB")


def parse_post_frontmatter(post_path: Path) -> dict:
    """Extract title and phase from post frontmatter."""
    content = post_path.read_text()

    # Extract frontmatter
    match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return {}

    frontmatter = match.group(1)
    data = {}

    for line in frontmatter.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip().strip('"\'')

    return data


def generate_all_og_images():
    """Generate OG images for all posts and main site."""

    # Main site OG
    print("\n📸 Generating main site OG image...")
    generate_og_image("GoGoGo Chronicle", is_main=True,
                     output_path=OG_DIR / "main.png")

    # Post OG images
    print("\n📸 Generating post OG images...")
    for post_path in sorted(POSTS_DIR.glob("*.md")):
        data = parse_post_frontmatter(post_path)
        title = data.get('title', post_path.stem)
        phase = data.get('phase')

        # Generate slug from filename
        slug = post_path.stem.split('-', 3)[-1] if '-' in post_path.stem else post_path.stem
        output_path = OG_DIR / f"{slug}.png"

        # Skip if exists and post hasn't changed
        if output_path.exists():
            post_mtime = post_path.stat().st_mtime
            og_mtime = output_path.stat().st_mtime
            if og_mtime > post_mtime:
                print(f"Skipping (up to date): {title}")
                continue

        generate_og_image(title, phase, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OG images for GoGoGo blog")
    parser.add_argument("--post", help="Generate for specific post title")
    parser.add_argument("--main", action="store_true", help="Generate main site OG only")
    parser.add_argument("--all", action="store_true", help="Regenerate all images")

    args = parser.parse_args()

    if args.main:
        generate_og_image("GoGoGo Chronicle", is_main=True,
                         output_path=OG_DIR / "main.png")
    elif args.post:
        generate_og_image(args.post)
    else:
        generate_all_og_images()

    print("\n✅ OG image generation complete!")
