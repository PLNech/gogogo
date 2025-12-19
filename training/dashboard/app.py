#!/usr/bin/env python3
"""Game analysis dashboard - FastAPI + HTMX + Tailwind.

A contemplative space to study self-play games.
"""
import json
from pathlib import Path
from typing import List, Optional
import sys

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_record import GameRecord
from dashboard.charts import generate_all_charts, overlay_victory_charts
from training_state import read_state, TrainingState
from board import Board

app = FastAPI(title="GoGoGo Game Dashboard", description="Contemplate the games")

# Setup templates and static files
DASHBOARD_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=DASHBOARD_DIR / "templates")
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR / "static"), name="static")

# Game storage
GAMES_DIR = DASHBOARD_DIR.parent / "games"
GAMES_DIR.mkdir(exist_ok=True)

# Cache loaded games
_game_cache: dict[str, GameRecord] = {}


def get_available_games() -> List[str]:
    """List available game IDs."""
    games = []
    for f in GAMES_DIR.glob("*.json"):
        games.append(f.stem)
    return sorted(games, reverse=True)  # Newest first


def load_game(game_id: str) -> Optional[GameRecord]:
    """Load a game by ID."""
    if game_id in _game_cache:
        return _game_cache[game_id]

    path = GAMES_DIR / f"{game_id}.json"
    if not path.exists():
        return None

    record = GameRecord.load(str(path))
    _game_cache[game_id] = record
    return record


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, bypass: str = None, winner: str = None, min_moves: int = None, max_moves: int = None):
    """Dashboard home - redirect to training if active, else list games."""
    # Check if training is active (unless bypass)
    if not bypass:
        state = read_state()
        if state and state.active:
            return RedirectResponse(url="/training", status_code=302)

    games = get_available_games()
    game_summaries = []

    for game_id in games[:50]:  # Show more games
        record = load_game(game_id)
        if record:
            # Apply filters
            if winner:
                if winner == "black" and record.winner != 1:
                    continue
                elif winner == "white" and record.winner != -1:
                    continue
            if min_moves and record.total_moves < min_moves:
                continue
            if max_moves and record.total_moves > max_moves:
                continue

            # Get final stats
            final_stats = record.move_stats[-1] if record.move_stats else None

            game_summaries.append({
                "id": game_id,
                "result": record.result_string,
                "winner": record.winner,
                "score": record.final_score,
                "moves": record.total_moves,
                "board_size": record.board_size,
                "black_captures": final_stats.total_black_captures if final_stats else 0,
                "white_captures": final_stats.total_white_captures if final_stats else 0,
                "black_territory": final_stats.black_territory if final_stats else 0,
                "white_territory": final_stats.white_territory if final_stats else 0,
                "final_black_groups": final_stats.black_groups if final_stats else 0,
                "final_white_groups": final_stats.white_groups if final_stats else 0,
                "avg_entropy": sum(s.policy_entropy for s in record.move_stats) / len(record.move_stats) if record.move_stats else 0,
            })

    # Stats
    total_black_wins = sum(1 for g in game_summaries if g["winner"] == 1)
    total_white_wins = sum(1 for g in game_summaries if g["winner"] == -1)
    avg_moves = sum(g["moves"] for g in game_summaries) / len(game_summaries) if game_summaries else 0

    return templates.TemplateResponse("index.html", {
        "request": request,
        "games": game_summaries,
        "total_games": len(games),
        "filtered_count": len(game_summaries),
        "black_wins": total_black_wins,
        "white_wins": total_white_wins,
        "avg_moves": avg_moves,
        "filters": {"winner": winner, "min_moves": min_moves, "max_moves": max_moves},
    })


@app.get("/training", response_class=HTMLResponse)
async def training_status(request: Request):
    """Live training status page."""
    state = read_state()
    games = get_available_games()[:5]  # Recent games

    return templates.TemplateResponse("training.html", {
        "request": request,
        "state": state,
        "recent_games": games,
    })


@app.get("/training/status", response_class=JSONResponse)
async def training_status_api():
    """API endpoint for training state (for HTMX polling)."""
    state = read_state()
    if not state:
        return {"active": False, "phase": "idle"}

    return {
        "active": state.active,
        "phase": state.phase,
        "iteration": state.iteration,
        "total_iterations": state.total_iterations,
        "current_game": state.current_game,
        "games_target": state.games_target,
        "train_step": state.train_step,
        "train_steps_target": state.train_steps_target,
        "last_loss": round(state.last_loss, 4),
        "last_policy_loss": round(state.last_policy_loss, 4),
        "last_value_loss": round(state.last_value_loss, 4),
        "buffer_size": state.buffer_size,
        "eval_win_rate": round(state.eval_win_rate, 2),
        "games_generated": state.games_generated,
        "last_update": state.last_update,
    }


@app.get("/game/{game_id}", response_class=HTMLResponse)
async def game_detail(request: Request, game_id: str):
    """Single game analysis page."""
    record = load_game(game_id)
    if not record:
        raise HTTPException(status_code=404, detail="Game not found")

    charts = generate_all_charts(record)

    # Get prev/next game IDs for navigation
    games = get_available_games()
    try:
        idx = games.index(game_id)
        prev_game = games[idx + 1] if idx + 1 < len(games) else None
        next_game = games[idx - 1] if idx > 0 else None
    except ValueError:
        prev_game = next_game = None

    return templates.TemplateResponse("game.html", {
        "request": request,
        "game": record,
        "game_id": game_id,
        "charts": charts,
        "charts_json": json.dumps(charts),
        "prev_game": prev_game,
        "next_game": next_game,
    })


@app.get("/game/{game_id}/chart/{chart_name}", response_class=JSONResponse)
async def get_chart(game_id: str, chart_name: str):
    """Get a specific chart as JSON (for HTMX updates)."""
    record = load_game(game_id)
    if not record:
        raise HTTPException(status_code=404, detail="Game not found")

    charts = generate_all_charts(record)
    if chart_name not in charts:
        raise HTTPException(status_code=404, detail="Chart not found")

    return charts[chart_name]


@app.get("/game/{game_id}/board/{move_num}", response_class=JSONResponse)
async def get_board_at_move(game_id: str, move_num: int):
    """Get board state at a specific move for visualization."""
    record = load_game(game_id)
    if not record:
        raise HTTPException(status_code=404, detail="Game not found")

    if move_num < 0 or move_num >= len(record.move_stats):
        raise HTTPException(status_code=400, detail="Invalid move number")

    # Use actual Board class to properly handle captures
    size = record.board_size
    go_board = Board(size)

    for i in range(move_num + 1):
        stats = record.move_stats[i]
        move = stats.move
        if move == (-1, -1):  # Pass
            go_board.pass_move()
        else:
            r, c = move
            if 0 <= r < size and 0 <= c < size:
                # Board handles captures internally
                go_board.play(r, c)

    # Convert numpy board to list
    board = go_board.board.tolist()

    # Get last move for highlighting
    last_move = record.move_stats[move_num].move
    if last_move == (-1, -1):
        last_move = None

    # Simple territory estimate (empty points surrounded by one color)
    territory = [[0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            if board[r][c] == 0:
                # Check neighbors
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and board[nr][nc] != 0:
                        neighbors.append(board[nr][nc])
                if neighbors and all(n == neighbors[0] for n in neighbors):
                    territory[r][c] = neighbors[0]

    # Get atari points from Board's group_stats (single pass)
    atari_dict = go_board.atari_points
    atari = [[0] * size for _ in range(size)]
    for r, c in atari_dict[1]:  # Black in atari
        atari[r][c] = 1
    for r, c in atari_dict[-1]:  # White in atari
        atari[r][c] = -1

    return {
        "board": board,
        "last_move": last_move,
        "territory": territory,
        "atari": atari,
        "move_num": move_num,
    }


@app.get("/game/{game_id}/move/{move_num}", response_class=HTMLResponse)
async def game_at_move(request: Request, game_id: str, move_num: int):
    """Get game state at a specific move (for slider)."""
    record = load_game(game_id)
    if not record:
        raise HTTPException(status_code=404, detail="Game not found")

    if move_num < 0 or move_num >= len(record.move_stats):
        raise HTTPException(status_code=400, detail="Invalid move number")

    stats = record.move_stats[move_num]

    return templates.TemplateResponse("partials/move_info.html", {
        "request": request,
        "stats": stats,
        "move_num": move_num,
        "total_moves": len(record.move_stats),
    })


@app.get("/compare", response_class=HTMLResponse)
async def compare_games(request: Request, ids: str = ""):
    """Compare multiple games overlaid."""
    game_ids = [g.strip() for g in ids.split(",") if g.strip()]

    if not game_ids:
        # Show selection UI
        games = get_available_games()[:20]
        return templates.TemplateResponse("compare_select.html", {
            "request": request,
            "games": games,
        })

    records = []
    labels = []
    for game_id in game_ids[:5]:  # Limit to 5
        record = load_game(game_id)
        if record:
            records.append(record)
            labels.append(f"{game_id} ({record.result_string})")

    if not records:
        raise HTTPException(status_code=404, detail="No valid games found")

    overlay_chart = overlay_victory_charts(records, labels)

    return templates.TemplateResponse("compare.html", {
        "request": request,
        "game_ids": game_ids,
        "labels": labels,
        "overlay_chart": json.dumps(overlay_chart),
    })


@app.get("/api/games", response_class=JSONResponse)
async def api_list_games():
    """API: List all games."""
    games = get_available_games()
    return {"games": games, "count": len(games)}


@app.get("/api/game/{game_id}", response_class=JSONResponse)
async def api_get_game(game_id: str):
    """API: Get full game data."""
    record = load_game(game_id)
    if not record:
        raise HTTPException(status_code=404, detail="Game not found")

    return json.loads(record.to_json())


# Development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
