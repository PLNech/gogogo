#!/usr/bin/env python3
"""
Neural Network Move Server

Simple HTTP server that provides neural network move predictions.
Used by watchGame.js to display the trained AI playing.

Usage:
    poetry run python serve.py --checkpoint checkpoints/supervised_best.pt
    poetry run python serve.py --checkpoint checkpoints/supervised_epoch_3.pt --port 8765
"""

import argparse
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
import numpy as np
from config import Config, DEFAULT
from model import load_checkpoint
from board import Board


class NeuralServer:
    """Singleton to hold model state."""
    model = None
    config = None
    device = None


def get_moves(board_state: dict) -> dict:
    """Get neural network move predictions for a board state.

    Args:
        board_state: {
            "size": int,
            "board": [[int, ...], ...],  # 0=empty, 1=black, -1=white
            "current_player": int  # 1=black, -1=white
        }

    Returns:
        {
            "moves": [{"move": "D4", "prob": 0.234, "row": 15, "col": 3}, ...],
            "value": 0.123,  # Position evaluation (-1 to 1)
            "pass_prob": 0.001
        }
    """
    model = NeuralServer.model
    config = NeuralServer.config

    size = board_state["size"]
    board_array = np.array(board_state["board"])
    current_player = board_state.get("current_player", 1)

    # Create board object
    board = Board(size)
    board.board = board_array.copy()
    board.current_player = current_player

    # Get model predictions
    model.eval()
    with torch.no_grad():
        tensor = torch.FloatTensor(board.to_tensor()).unsqueeze(0)
        tensor = tensor.to(NeuralServer.device)
        log_policy, value = model(tensor)
        policy = torch.exp(log_policy).cpu().numpy()[0]
        value = value.cpu().item()

    # Convert to moves
    cols = "ABCDEFGHJKLMNOPQRST"  # Skip I
    moves = []

    for idx in policy.argsort()[::-1][:10]:  # Top 10 moves
        if idx == size * size:
            continue  # Skip pass for now

        r, c = idx // size, idx % size
        if board_array[r, c] == 0:  # Empty position
            move_str = f"{cols[c]}{size - r}"
            moves.append({
                "move": move_str,
                "prob": float(policy[idx]),
                "row": int(r),
                "col": int(c)
            })

    pass_prob = float(policy[size * size]) if size * size < len(policy) else 0.0

    return {
        "moves": moves,
        "value": float(value),
        "pass_prob": pass_prob
    }


class MoveHandler(BaseHTTPRequestHandler):
    """HTTP request handler for move predictions."""

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """Handle move request."""
        if self.path != '/move':
            self.send_error(404)
            return

        # Read request body
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)

        try:
            board_state = json.loads(body)
            result = get_moves(board_state)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_GET(self):
        """Handle status check."""
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            status = {
                "status": "ready",
                "model": {
                    "blocks": NeuralServer.config.num_blocks,
                    "filters": NeuralServer.config.num_filters,
                    "board_size": NeuralServer.config.board_size
                },
                "device": str(NeuralServer.device)
            }
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    parser = argparse.ArgumentParser(description='Neural network move server')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/supervised_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--port', type=int, default=8765,
                        help='Server port')
    args = parser.parse_args()

    # Load model
    config = DEFAULT
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading checkpoint: {args.checkpoint}")
    model, step = load_checkpoint(args.checkpoint, config)
    NeuralServer.model = model
    NeuralServer.config = model.config
    NeuralServer.device = config.device

    print(f"Model: {model.config.num_blocks} blocks, {model.config.num_filters} filters")
    print(f"Board size: {model.config.board_size}x{model.config.board_size}")
    print(f"Device: {config.device}")
    print(f"Trained for {step} steps")

    # Start server
    server = HTTPServer(('localhost', args.port), MoveHandler)
    print(f"\n🧠 Neural server running on http://localhost:{args.port}")
    print(f"   POST /move - Get move predictions")
    print(f"   GET /status - Check server status")
    print(f"\nPress Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
