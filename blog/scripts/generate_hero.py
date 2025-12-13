#!/usr/bin/env python3
"""
Hero Image Generator for GoGoGo Blog
Creates beautiful Go board renders with gradient backgrounds.
No text - pure board aesthetics. Stones on proper intersections.
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageEnhance

BLOG_DIR = Path(__file__).parent.parent
POSTS_DIR = BLOG_DIR / "_posts"
HERO_DIR = BLOG_DIR / "assets" / "hero"
HERO_DIR.mkdir(parents=True, exist_ok=True)

# Dimensions
HERO_WIDTH = 1200
HERO_HEIGHT = 400
DPI = 100
BOARD_SIZE = 19  # Full 19x19 grid

# Theme palettes - KataGo-inspired ownership colors
THEMES = {
    'Genesis': {
        'gradient': ['#0d1b2a', '#1b263b', '#415a77'],
        'grid_color': '#778da9',
        'grid_alpha': 0.3,
    },
    'Foundation': {
        'gradient': ['#1a237e', '#3949ab', '#5c6bc0'],
        'grid_color': '#9fa8da',
        'grid_alpha': 0.35,
    },
    'Struggle': {
        'gradient': ['#4a148c', '#7b1fa2', '#ab47bc'],
        'grid_color': '#ce93d8',
        'grid_alpha': 0.3,
    },
    'Research': {
        'gradient': ['#e65100', '#f57c00', '#ffb74d'],
        'grid_color': '#4e342e',
        'grid_alpha': 0.25,
    },
    'Synthesis': {
        'gradient': ['#1a237e', '#5c6bc0', '#ffcc80'],
        'grid_color': '#90a4ae',
        'grid_alpha': 0.3,
    },
}

DEFAULT_THEME = {
    'gradient': ['#2d2a26', '#4a4540', '#6b6560'],
    'grid_color': '#1a1a1a',
    'grid_alpha': 0.2,
}


def create_gradient_background(ax, colors):
    """Create smooth diagonal gradient."""
    n = 256
    gradient = np.zeros((n, n, 3))

    for i in range(n):
        for j in range(n):
            t = (i + j) / (2 * n)
            if t < 0.5:
                t2 = t * 2
                c1 = np.array([int(colors[0][k:k+2], 16) for k in (1, 3, 5)]) / 255
                c2 = np.array([int(colors[1][k:k+2], 16) for k in (1, 3, 5)]) / 255
            else:
                t2 = (t - 0.5) * 2
                c1 = np.array([int(colors[1][k:k+2], 16) for k in (1, 3, 5)]) / 255
                c2 = np.array([int(colors[2][k:k+2], 16) for k in (1, 3, 5)]) / 255
            gradient[i, j] = c1 * (1 - t2) + c2 * t2

    ax.imshow(gradient, aspect='auto', extent=[0, 1, 0, 1], zorder=0)


def draw_grid(ax, grid_color, grid_alpha, n_lines=13):
    """Draw subtle Go board grid lines with correct aspect ratio."""
    aspect = HERO_WIDTH / HERO_HEIGHT  # 3.0
    margin = 0.05

    # Grid spacing should be the same visually (square cells)
    # Height determines spacing, width gets more lines
    cell_size = (1 - 2*margin) / (n_lines - 1)

    # Horizontal lines (based on height)
    for i in range(n_lines):
        y = margin + cell_size * i
        ax.axhline(y, color=grid_color, alpha=grid_alpha, linewidth=0.5, zorder=1)

    # Vertical lines - need more due to wider image
    # Adjust spacing to make square cells visually
    n_vertical = int((n_lines - 1) * aspect) + 1
    for i in range(n_vertical):
        x = margin + (1 - 2*margin) * i / (n_vertical - 1)
        ax.axvline(x, color=grid_color, alpha=grid_alpha, linewidth=0.5, zorder=1)


def draw_stone(ax, grid_x, grid_y, color, n_vertical=37, n_horizontal=13):
    """Draw a realistic Go stone at grid intersection."""
    margin = 0.05
    aspect = HERO_WIDTH / HERO_HEIGHT  # 3.0

    # Convert grid coordinates to plot coordinates
    # grid_x is on the vertical lines (0 to n_vertical-1)
    # grid_y is on the horizontal lines (0 to n_horizontal-1)
    x = margin + (1 - 2*margin) * grid_x / (n_vertical - 1)
    y = margin + (1 - 2*margin) * grid_y / (n_horizontal - 1)

    # Stone size - to appear round in 3:1 image, height must be width/aspect
    # Using width in normalized coords, height must be smaller
    stone_width = 0.025  # Width in normalized x coords
    stone_height = stone_width / aspect  # Height in normalized y coords (smaller to appear round)

    # Main stone body
    if color == 'black':
        base_color = '#1a1a1a'
        edge_color = '#0a0a0a'
        highlight_color = '#3a3a3a'
    else:
        base_color = '#f0f0f0'
        edge_color = '#cccccc'
        highlight_color = '#ffffff'

    # Stone shadow (subtle)
    shadow = patches.Ellipse(
        (x + 0.002, y - 0.003),
        stone_width * 1.05, stone_height * 1.05,
        facecolor='#000000', alpha=0.2, zorder=2
    )
    ax.add_patch(shadow)

    # Main stone
    stone = patches.Ellipse(
        (x, y), stone_width, stone_height,
        facecolor=base_color, edgecolor=edge_color,
        linewidth=0.5, zorder=3
    )
    ax.add_patch(stone)

    # Highlight for 3D effect
    highlight = patches.Ellipse(
        (x - stone_width * 0.15, y + stone_height * 0.2),
        stone_width * 0.3, stone_height * 0.3,
        facecolor=highlight_color, alpha=0.6, zorder=4
    )
    ax.add_patch(highlight)


def generate_stone_positions(pattern_type, seed=42):
    """Generate realistic stone positions on grid intersections.

    Grid is 37 vertical x 13 horizontal lines.
    x: 0-36, y: 0-12
    """
    np.random.seed(seed)
    positions = []

    if pattern_type == 'sparse':
        # Few stones, corner-focused (opening feel)
        positions = [
            (5, 2, 'black'), (5, 10, 'white'), (31, 2, 'black'),
            (8, 2, 'white'), (5, 5, 'black'),
        ]

    elif pattern_type == 'structured':
        # Orderly fuseki pattern across the board
        positions = [
            (5, 2, 'black'), (31, 10, 'white'), (31, 2, 'black'), (5, 10, 'white'),
            (8, 2, 'white'), (5, 5, 'black'), (8, 10, 'black'), (31, 5, 'white'),
            (18, 2, 'black'), (18, 10, 'white'), (12, 6, 'white'), (24, 6, 'black'),
        ]

    elif pattern_type == 'chaotic':
        # Fighting pattern - groups in conflict in center
        positions = [
            # Black group
            (15, 5), (15, 6), (16, 5), (16, 7), (17, 6),
        ]
        positions = [(x, y, 'black') for x, y in positions]
        # White attacking
        white = [(14, 5), (14, 6), (15, 4), (17, 5), (18, 6)]
        positions.extend([(x, y, 'white') for x, y in white])
        # Corner stones
        positions.extend([(5, 2, 'black'), (31, 10, 'white'), (5, 10, 'black')])

    elif pattern_type == 'analytical':
        # Corner joseki patterns
        positions = [
            # Left corner joseki
            (5, 2, 'black'), (8, 2, 'white'), (5, 5, 'black'), (7, 4, 'white'),
            (8, 5, 'black'), (10, 2, 'white'), (5, 7, 'black'),
            # Right corner joseki
            (31, 10, 'white'), (28, 10, 'black'), (31, 7, 'white'), (29, 8, 'black'),
        ]

    elif pattern_type == 'balanced':
        # Territory on both sides
        positions = [
            # Black territory (left)
            (4, 3, 'black'), (4, 4, 'black'), (4, 5, 'black'), (4, 6, 'black'),
            (5, 4, 'black'), (5, 6, 'black'), (6, 5, 'black'),
            # White territory (right)
            (32, 6, 'white'), (32, 7, 'white'), (32, 8, 'white'), (32, 9, 'white'),
            (31, 7, 'white'), (31, 9, 'white'), (30, 8, 'white'),
            # Center
            (18, 5, 'black'), (19, 5, 'white'), (18, 7, 'white'), (19, 7, 'black'),
        ]

    return positions


def generate_hero_image(phase: str, slug: str, seed: int = 42):
    """Generate a hero image for a blog post."""
    theme = THEMES.get(phase, DEFAULT_THEME)

    fig, ax = plt.subplots(figsize=(HERO_WIDTH/DPI, HERO_HEIGHT/DPI), dpi=DPI)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Gradient background
    create_gradient_background(ax, theme['gradient'])

    # Subtle grid
    draw_grid(ax, theme['grid_color'], theme['grid_alpha'])

    # Pattern based on phase
    patterns = {
        'Genesis': 'sparse',
        'Foundation': 'structured',
        'Struggle': 'chaotic',
        'Research': 'analytical',
        'Synthesis': 'balanced',
    }
    pattern = patterns.get(phase, 'sparse')

    # Draw stones on proper grid intersections
    positions = generate_stone_positions(pattern, seed=seed)
    for gx, gy, color in positions:
        draw_stone(ax, gx, gy, color)

    # Save
    output_path = HERO_DIR / f"{slug}.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0,
                facecolor=fig.get_facecolor())
    plt.close()

    # Optimize
    final_path = optimize_image(output_path)

    # B&W thumbnail
    generate_bw_thumbnail(final_path)

    print(f"Generated hero: {slug}")
    return final_path


def optimize_image(path: Path, max_kb=150) -> Path:
    """Optimize image for web. Returns final path."""
    img = Image.open(path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    img = img.resize((HERO_WIDTH, HERO_HEIGHT), Image.Resampling.LANCZOS)
    img.save(path, 'PNG', optimize=True)

    size_kb = path.stat().st_size / 1024
    if size_kb > max_kb:
        jpeg_path = path.with_suffix('.jpg')
        img.save(jpeg_path, 'JPEG', quality=88, optimize=True)
        if jpeg_path.stat().st_size < path.stat().st_size:
            path.unlink()
            print(f"  → JPEG {jpeg_path.stat().st_size/1024:.0f}KB")
            return jpeg_path

    print(f"  → PNG {size_kb:.0f}KB")
    return path


def generate_bw_thumbnail(color_path: Path):
    """Generate grayscale thumbnail for hover effect."""
    img = Image.open(color_path)
    bw = img.convert('L').convert('RGB')
    enhancer = ImageEnhance.Contrast(bw)
    bw = enhancer.enhance(1.15)

    bw_path = color_path.parent / f"{color_path.stem}-bw{color_path.suffix}"
    bw.save(bw_path, optimize=True)


def parse_frontmatter(post_path: Path) -> dict:
    """Extract frontmatter from post."""
    content = post_path.read_text()
    match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return {}

    data = {}
    for line in match.group(1).split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip().strip('"\'')
    return data


def generate_all_heroes():
    """Generate hero images for all posts."""
    print("\n🎨 Generating hero images...\n")

    for i, post_path in enumerate(sorted(POSTS_DIR.glob("*.md"))):
        data = parse_frontmatter(post_path)
        phase = data.get('phase', 'Genesis')
        slug = post_path.stem.split('-', 3)[-1] if '-' in post_path.stem else post_path.stem

        generate_hero_image(phase, slug, seed=42 + i)

    print("\n🏠 Generating main hero...")
    generate_main_hero()

    print("\n✅ Done!")


def generate_main_hero():
    """Generate main blog landing page hero."""
    theme = {
        'gradient': ['#1a1a1a', '#2d2a26', '#4a4540'],
        'grid_color': '#6b6560',
        'grid_alpha': 0.25,
    }

    fig, ax = plt.subplots(figsize=(HERO_WIDTH/DPI, HERO_HEIGHT/DPI), dpi=DPI)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    create_gradient_background(ax, theme['gradient'])
    draw_grid(ax, theme['grid_color'], theme['grid_alpha'])

    # Elegant balanced position
    positions = generate_stone_positions('balanced', seed=123)
    for gx, gy, color in positions:
        draw_stone(ax, gx, gy, color)

    output_path = HERO_DIR / "main.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
    plt.close()

    final_path = optimize_image(output_path)
    generate_bw_thumbnail(final_path)


if __name__ == "__main__":
    generate_all_heroes()
