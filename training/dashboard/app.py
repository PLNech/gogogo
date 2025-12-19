#!/usr/bin/env python3
"""Game analysis dashboard - FastAPI + HTMX + Tailwind.

A contemplative space to study self-play games.
"""
import json
from pathlib import Path
from typing import List, Optional
import sys

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_record import GameRecord
from dashboard.charts import generate_all_charts, overlay_victory_charts

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
async def index(request: Request):
    """Dashboard home - list of games."""
    games = get_available_games()
    game_summaries = []

    for game_id in games[:20]:  # Limit to recent 20
        record = load_game(game_id)
        if record:
            game_summaries.append({
                "id": game_id,
                "result": record.result_string,
                "moves": record.total_moves,
                "board_size": record.board_size,
            })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "games": game_summaries,
        "total_games": len(games),
    })


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
