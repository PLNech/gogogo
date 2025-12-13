#!/usr/bin/env python3
"""Download 13x13 games from OGS (Online Go Server) API."""

import requests
import time
from pathlib import Path
import argparse

OGS_API = "https://online-go.com/api/v1"


def download_games(board_size: int = 13, max_games: int = 1000, min_rank: int = 1):
    """Download games from OGS.

    Args:
        board_size: Board size (9, 13, or 19)
        max_games: Maximum number of games to download
        min_rank: Minimum rank (1=30k, 31=1d, 38=8d)
    """
    output_dir = Path(f"data/ogs_{board_size}x{board_size}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {board_size}x{board_size} games from OGS...")
    print(f"Output directory: {output_dir}")

    page = 1
    games_downloaded = 0

    while games_downloaded < max_games:
        # OGS game search API
        params = {
            'width': board_size,
            'height': board_size,
            'ended__isnull': False,  # Only completed games
            'page_size': 100,
            'page': page,
            'ordering': '-ended',  # Most recent first
        }

        try:
            response = requests.get(f"{OGS_API}/games/", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break

        if 'results' not in data or len(data['results']) == 0:
            print("No more games found")
            break

        for game in data['results']:
            if games_downloaded >= max_games:
                break

            game_id = game['id']

            # Check if already downloaded
            sgf_path = output_dir / f"{game_id}.sgf"
            if sgf_path.exists():
                continue

            # Filter by rank (optional)
            black_rank = game.get('players', {}).get('black', {}).get('rank', 0)
            white_rank = game.get('players', {}).get('white', {}).get('rank', 0)

            if black_rank < min_rank or white_rank < min_rank:
                continue

            # Download SGF
            try:
                sgf_url = f"{OGS_API}/games/{game_id}/sgf"
                sgf_response = requests.get(sgf_url, timeout=30)
                sgf_response.raise_for_status()

                # Save SGF
                with open(sgf_path, 'w', encoding='utf-8') as f:
                    f.write(sgf_response.text)

                games_downloaded += 1
                if games_downloaded % 10 == 0:
                    print(f"Downloaded {games_downloaded}/{max_games} games")

                # Rate limit
                time.sleep(0.5)

            except Exception as e:
                print(f"Error downloading game {game_id}: {e}")
                continue

        page += 1
        time.sleep(1)  # Be nice to OGS API

    print(f"\nDownload complete! {games_downloaded} games saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Download games from OGS')
    parser.add_argument('--size', type=int, default=13, choices=[9, 13, 19],
                        help='Board size')
    parser.add_argument('--max-games', type=int, default=1000,
                        help='Maximum number of games')
    parser.add_argument('--min-rank', type=int, default=20,
                        help='Minimum rank (1=30k, 20=10k, 31=1d, 38=8d)')
    args = parser.parse_args()

    download_games(args.size, args.max_games, args.min_rank)


if __name__ == '__main__':
    main()
