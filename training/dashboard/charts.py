"""Chart generation for game analysis dashboard.

Generates Plotly.js JSON for client-side rendering with HTMX swaps.
"""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_record import GameRecord


def victory_probability_chart(record: GameRecord) -> Dict[str, Any]:
    """Generate win probability area chart.

    Black probability fills from bottom, white from top.
    The meeting line shows the game flow.
    """
    moves = list(range(len(record.move_stats)))
    black_probs = record.black_win_probs
    white_probs = record.white_win_probs

    # Smooth the probabilities slightly for visual appeal
    def smooth(values, window=3):
        if len(values) < window:
            return values
        result = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            result.append(sum(values[start:end]) / (end - start))
        return result

    black_smooth = smooth(black_probs)

    return {
        "data": [
            {
                "x": moves,
                "y": black_smooth,
                "fill": "tozeroy",
                "fillcolor": "rgba(30, 30, 30, 0.8)",
                "line": {"color": "rgb(30, 30, 30)", "width": 2},
                "name": "Black",
                "hovertemplate": "Move %{x}<br>Black: %{y:.1%}<extra></extra>"
            },
            {
                "x": moves,
                "y": [1.0] * len(moves),
                "fill": "tonexty",
                "fillcolor": "rgba(245, 245, 220, 0.8)",
                "line": {"width": 0},
                "name": "White",
                "hoverinfo": "skip"
            }
        ],
        "layout": {
            "title": {"text": "Victory Probability", "font": {"family": "Georgia, serif"}},
            "xaxis": {"title": "Move", "showgrid": False},
            "yaxis": {"title": "", "showgrid": False, "range": [0, 1], "tickformat": ".0%"},
            "showlegend": False,
            "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(250, 245, 235, 0.5)",
            "hovermode": "x unified"
        }
    }


def groups_chart(record: GameRecord) -> Dict[str, Any]:
    """Generate live groups count chart."""
    moves = list(range(len(record.move_stats)))
    black_groups = record.black_group_counts
    white_groups = record.white_group_counts

    return {
        "data": [
            {
                "x": moves,
                "y": black_groups,
                "mode": "lines",
                "line": {"color": "rgb(30, 30, 30)", "width": 2},
                "name": "Black groups",
                "hovertemplate": "Move %{x}<br>Black: %{y} groups<extra></extra>"
            },
            {
                "x": moves,
                "y": white_groups,
                "mode": "lines",
                "line": {"color": "rgb(180, 180, 180)", "width": 2, "dash": "dot"},
                "name": "White groups",
                "hovertemplate": "Move %{x}<br>White: %{y} groups<extra></extra>"
            }
        ],
        "layout": {
            "title": {"text": "Live Groups", "font": {"family": "Georgia, serif"}},
            "xaxis": {"title": "Move", "showgrid": False},
            "yaxis": {"title": "Groups", "showgrid": True, "gridcolor": "rgba(0,0,0,0.1)"},
            "showlegend": True,
            "legend": {"x": 0.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.8)"},
            "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(250, 245, 235, 0.5)",
            "hovermode": "x unified"
        }
    }


def territory_chart(record: GameRecord) -> Dict[str, Any]:
    """Generate territory/score trajectory chart."""
    moves = list(range(len(record.move_stats)))
    scores = record.score_trajectory

    # Color gradient based on who's ahead
    colors = ["rgb(30, 30, 30)" if s > 0 else "rgb(180, 180, 180)" for s in scores]

    return {
        "data": [
            {
                "x": moves,
                "y": scores,
                "mode": "lines",
                "line": {"color": "rgb(100, 100, 100)", "width": 2},
                "fill": "tozeroy",
                "fillcolor": "rgba(30, 30, 30, 0.3)",
                "name": "Score",
                "hovertemplate": "Move %{x}<br>Score: %{y:+.1f}<extra></extra>"
            },
            {
                "x": moves,
                "y": [0] * len(moves),
                "mode": "lines",
                "line": {"color": "rgba(0,0,0,0.3)", "width": 1, "dash": "dash"},
                "hoverinfo": "skip"
            }
        ],
        "layout": {
            "title": {"text": "Score Estimate", "font": {"family": "Georgia, serif"}},
            "xaxis": {"title": "Move", "showgrid": False},
            "yaxis": {"title": "Black advantage", "showgrid": True, "gridcolor": "rgba(0,0,0,0.1)", "zeroline": True},
            "showlegend": False,
            "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(250, 245, 235, 0.5)",
            "hovermode": "x unified"
        }
    }


def captures_chart(record: GameRecord) -> Dict[str, Any]:
    """Generate cumulative captures chart with events."""
    moves = list(range(len(record.move_stats)))
    black_caps = [s.total_black_captures for s in record.move_stats]
    white_caps = [s.total_white_captures for s in record.move_stats]

    # Capture events for annotations
    events = record.capture_events
    annotations = []
    for move_num, player, count in events:
        if count >= 2:  # Only annotate significant captures
            annotations.append({
                "x": move_num,
                "y": black_caps[move_num] if player == 1 else white_caps[move_num],
                "text": f"+{count}",
                "showarrow": False,
                "font": {"size": 10, "color": "rgb(30,30,30)" if player == 1 else "rgb(150,150,150)"}
            })

    return {
        "data": [
            {
                "x": moves,
                "y": black_caps,
                "mode": "lines",
                "line": {"color": "rgb(30, 30, 30)", "width": 2},
                "name": "Black captures",
                "hovertemplate": "Move %{x}<br>Black captured: %{y}<extra></extra>"
            },
            {
                "x": moves,
                "y": white_caps,
                "mode": "lines",
                "line": {"color": "rgb(180, 180, 180)", "width": 2},
                "name": "White captures",
                "hovertemplate": "Move %{x}<br>White captured: %{y}<extra></extra>"
            }
        ],
        "layout": {
            "title": {"text": "Captures", "font": {"family": "Georgia, serif"}},
            "xaxis": {"title": "Move", "showgrid": False},
            "yaxis": {"title": "Stones captured", "showgrid": True, "gridcolor": "rgba(0,0,0,0.1)"},
            "showlegend": True,
            "legend": {"x": 0.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.8)"},
            "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(250, 245, 235, 0.5)",
            "annotations": annotations[:10],  # Limit annotations
            "hovermode": "x unified"
        }
    }


def liberties_chart(record: GameRecord) -> Dict[str, Any]:
    """Generate minimum liberties chart (danger indicator)."""
    moves = list(range(len(record.move_stats)))
    black_min = [s.min_black_liberties for s in record.move_stats]
    white_min = [s.min_white_liberties for s in record.move_stats]

    # Highlight atari situations (lib = 1)
    black_atari_moves = [m for m, lib in enumerate(black_min) if lib == 1]
    white_atari_moves = [m for m, lib in enumerate(white_min) if lib == 1]

    return {
        "data": [
            {
                "x": moves,
                "y": black_min,
                "mode": "lines",
                "line": {"color": "rgb(30, 30, 30)", "width": 2},
                "name": "Black min libs",
            },
            {
                "x": moves,
                "y": white_min,
                "mode": "lines",
                "line": {"color": "rgb(180, 180, 180)", "width": 2},
                "name": "White min libs",
            },
            {
                "x": black_atari_moves,
                "y": [1] * len(black_atari_moves),
                "mode": "markers",
                "marker": {"color": "red", "size": 8, "symbol": "x"},
                "name": "Black in atari",
                "hovertemplate": "Move %{x}: Black in atari!<extra></extra>"
            },
            {
                "x": white_atari_moves,
                "y": [1] * len(white_atari_moves),
                "mode": "markers",
                "marker": {"color": "orange", "size": 8, "symbol": "x"},
                "name": "White in atari",
                "hovertemplate": "Move %{x}: White in atari!<extra></extra>"
            }
        ],
        "layout": {
            "title": {"text": "Minimum Liberties (Danger)", "font": {"family": "Georgia, serif"}},
            "xaxis": {"title": "Move", "showgrid": False},
            "yaxis": {"title": "Liberties", "showgrid": True, "gridcolor": "rgba(0,0,0,0.1)"},
            "showlegend": True,
            "legend": {"x": 0.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.8)"},
            "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(250, 245, 235, 0.5)",
            "hovermode": "x unified"
        }
    }


def policy_entropy_chart(record: GameRecord) -> Dict[str, Any]:
    """Generate policy entropy chart (model uncertainty)."""
    moves = list(range(len(record.move_stats)))
    entropy = [s.policy_entropy for s in record.move_stats]

    return {
        "data": [
            {
                "x": moves,
                "y": entropy,
                "mode": "lines",
                "fill": "tozeroy",
                "fillcolor": "rgba(100, 100, 200, 0.3)",
                "line": {"color": "rgb(70, 70, 150)", "width": 2},
                "name": "Policy entropy",
                "hovertemplate": "Move %{x}<br>Entropy: %{y:.2f}<extra></extra>"
            }
        ],
        "layout": {
            "title": {"text": "Policy Entropy (Uncertainty)", "font": {"family": "Georgia, serif"}},
            "xaxis": {"title": "Move", "showgrid": False},
            "yaxis": {"title": "Entropy", "showgrid": True, "gridcolor": "rgba(0,0,0,0.1)"},
            "showlegend": False,
            "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(250, 245, 235, 0.5)",
            "hovermode": "x unified"
        }
    }


def generate_all_charts(record: GameRecord) -> Dict[str, Dict[str, Any]]:
    """Generate all charts for a game."""
    return {
        "victory": victory_probability_chart(record),
        "groups": groups_chart(record),
        "territory": territory_chart(record),
        "captures": captures_chart(record),
        "liberties": liberties_chart(record),
        "entropy": policy_entropy_chart(record),
    }


def charts_to_json(record: GameRecord) -> str:
    """Export all charts as JSON for embedding."""
    return json.dumps(generate_all_charts(record))


# Comparison charts for multiple games
def overlay_victory_charts(records: List[GameRecord], labels: List[str] = None) -> Dict[str, Any]:
    """Overlay win probability from multiple games."""
    if labels is None:
        labels = [f"Game {i+1}" for i in range(len(records))]

    colors = [
        "rgba(30, 30, 30, 0.8)",
        "rgba(70, 130, 180, 0.8)",
        "rgba(180, 70, 70, 0.8)",
        "rgba(70, 180, 70, 0.8)",
        "rgba(180, 130, 70, 0.8)",
    ]

    data = []
    for i, (record, label) in enumerate(zip(records, labels)):
        moves = list(range(len(record.move_stats)))
        black_probs = record.black_win_probs
        color = colors[i % len(colors)]

        data.append({
            "x": moves,
            "y": black_probs,
            "mode": "lines",
            "line": {"color": color, "width": 2},
            "name": label,
            "hovertemplate": f"{label}<br>Move %{{x}}<br>Black: %{{y:.1%}}<extra></extra>"
        })

    return {
        "data": data,
        "layout": {
            "title": {"text": "Win Probability Comparison", "font": {"family": "Georgia, serif"}},
            "xaxis": {"title": "Move", "showgrid": False},
            "yaxis": {"title": "P(Black wins)", "range": [0, 1], "tickformat": ".0%"},
            "showlegend": True,
            "legend": {"x": 0.02, "y": 0.98},
            "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(250, 245, 235, 0.5)",
            "hovermode": "x unified"
        }
    }
