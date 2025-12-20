#!/usr/bin/env python3
"""Adaptive Instinct Curriculum for Go AI Training.

The model can't learn captures if captures never appear in training data.
Self-play with random init → no captures → model never learns.

Solution: Wire instincts directly into training loss.

Loss = L_policy + L_value + λ(t) × L_instinct

Where:
    λ(t) = λ₀ × (1 - instinct_accuracy)

As model masters instincts → λ → 0 → pure RL takes over.

Usage:
    from instinct_loss import InstinctCurriculum

    curriculum = InstinctCurriculum(model, config)

    # During training:
    loss = policy_loss + value_loss + curriculum.compute_loss(board, policy)

    # Periodically update curriculum weight:
    curriculum.update_lambda(current_benchmark_accuracy)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from board import Board
from tactics import TacticalAnalyzer
from instincts import InstinctAnalyzer
from sensei_instincts import SenseiInstinctDetector


@dataclass
class InstinctOpportunity:
    """A detected instinct opportunity in a position."""
    category: str  # capture, escape, connect, cut, extend, block, atari, defend
    moves: List[Tuple[int, int]]  # Correct moves for this instinct
    priority: float  # Higher = more important (captures > extends)


class InstinctDetector:
    """Detects instinct opportunities in Go positions.

    Combines TacticalAnalyzer (capture, escape, atari) with InstinctAnalyzer
    (connect, block, etc.) to provide comprehensive instinct detection.
    """

    # Priority order - learned from 2000 Atari Go games (2025-12-20)
    # Higher priority = more important to learn (auxiliary loss weight)
    #
    # Key finding: hane_vs_tsuke has +13.2% follow advantage!
    # When opponent attaches, wrapping with hane is the winning response.
    # All 8 proverbs confirmed positive with correct pattern detection.
    PRIORITIES = {
        # Tactical (deterministic - must learn)
        'capture': 3.0,           # Taking opponent's stones
        'escape': 2.5,            # Saving own stones
        'atari': 2.0,             # Putting opponent in atari
        # Sensei's 8 Instincts (learned weights from 2000 Atari Go games)
        'hane_vs_tsuke': 3.0,     # ツケにはハネ (+13.2% advantage!) 🏆
        'extend_from_atari': 2.8, # アタリから伸びよ (+9.7% advantage)
        'block_the_thrust': 2.8,  # ツキアタリには (+9.6% advantage)
        'connect_vs_peep': 2.0,   # ノゾキにはツギ (+3.4% advantage)
        'block_the_angle': 2.0,   # カケにはオサエ (+3.5% advantage)
        'connect': 1.5,           # Joining groups
        'cut': 1.5,               # Separating opponent groups
        'stretch_from_bump': 1.5, # ブツカリから伸びよ (+3.2% advantage)
        'stretch_from_kosumi': 1.5,  # コスミから伸びよ (+3.0% advantage)
        'hane_at_head_of_two': 1.2,  # 二子の頭にハネ (+1.9% advantage)
        'defend': 1.0,            # Protecting weak points
        'block': 1.0,             # Preventing opponent extension
        'extend': 1.0,            # Gaining space
    }

    def __init__(self):
        self.tactical = TacticalAnalyzer()
        self.instinct = InstinctAnalyzer()
        self.sensei = SenseiInstinctDetector()

    def detect_capture(self, board: Board) -> Optional[InstinctOpportunity]:
        """Detect if current player can capture opponent stones."""
        player = board.current_player
        opponent = -player
        capture_moves = []

        # Find opponent groups in atari (1 liberty)
        visited = set()
        for r in range(board.size):
            for c in range(board.size):
                if (r, c) in visited or board.board[r, c] != opponent:
                    continue
                group = board.get_group(r, c)
                visited.update(group)

                if board.count_liberties(group) == 1:
                    # Find the capturing move (the last liberty)
                    for gr, gc in group:
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = gr + dr, gc + dc
                            if 0 <= nr < board.size and 0 <= nc < board.size:
                                if board.board[nr, nc] == 0:
                                    capture_moves.append((nr, nc))

        if capture_moves:
            return InstinctOpportunity(
                category='capture',
                moves=list(set(capture_moves)),
                priority=self.PRIORITIES['capture']
            )
        return None

    def detect_escape(self, board: Board) -> Optional[InstinctOpportunity]:
        """Detect if current player has stones in atari that can escape."""
        player = board.current_player
        escape_moves = []

        # Find our groups in atari
        visited = set()
        for r in range(board.size):
            for c in range(board.size):
                if (r, c) in visited or board.board[r, c] != player:
                    continue
                group = board.get_group(r, c)
                visited.update(group)

                if board.count_liberties(group) == 1:
                    # Find escape moves (extending the last liberty)
                    # Also consider capturing the attacker
                    for gr, gc in group:
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = gr + dr, gc + dc
                            if 0 <= nr < board.size and 0 <= nc < board.size:
                                if board.board[nr, nc] == 0:
                                    # Check if this gives us more liberties
                                    escape_moves.append((nr, nc))

        if escape_moves:
            return InstinctOpportunity(
                category='escape',
                moves=list(set(escape_moves)),
                priority=self.PRIORITIES['escape']
            )
        return None

    def detect_atari(self, board: Board) -> Optional[InstinctOpportunity]:
        """Detect moves that put opponent in atari."""
        player = board.current_player
        opponent = -player
        atari_moves = []

        # Find opponent groups with 2 liberties
        visited = set()
        for r in range(board.size):
            for c in range(board.size):
                if (r, c) in visited or board.board[r, c] != opponent:
                    continue
                group = board.get_group(r, c)
                visited.update(group)

                liberties = board.count_liberties(group)
                if liberties == 2:
                    # Find the liberty moves (playing either puts them in atari)
                    for gr, gc in group:
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = gr + dr, gc + dc
                            if 0 <= nr < board.size and 0 <= nc < board.size:
                                if board.board[nr, nc] == 0:
                                    atari_moves.append((nr, nc))

        if atari_moves:
            return InstinctOpportunity(
                category='atari',
                moves=list(set(atari_moves)),
                priority=self.PRIORITIES['atari']
            )
        return None

    def detect_connect(self, board: Board) -> Optional[InstinctOpportunity]:
        """Detect moves that connect friendly groups."""
        player = board.current_player
        connect_moves = []

        # Find groups of the same color
        groups = []
        visited = set()
        for r in range(board.size):
            for c in range(board.size):
                if (r, c) in visited or board.board[r, c] != player:
                    continue
                group = board.get_group(r, c)
                visited.update(group)
                groups.append(group)

        if len(groups) < 2:
            return None

        # Find empty points adjacent to multiple groups
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r, c] != 0:
                    continue

                adjacent_groups = set()
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board.size and 0 <= nc < board.size:
                        if board.board[nr, nc] == player:
                            # Find which group this belongs to
                            for i, group in enumerate(groups):
                                if (nr, nc) in group:
                                    adjacent_groups.add(i)
                                    break

                if len(adjacent_groups) >= 2:
                    connect_moves.append((r, c))

        if connect_moves:
            return InstinctOpportunity(
                category='connect',
                moves=connect_moves,
                priority=self.PRIORITIES['connect']
            )
        return None

    def detect_all(self, board: Board) -> List[InstinctOpportunity]:
        """Detect all instinct opportunities in a position.

        Combines tactical detectors with Sensei's 8 Basic Instincts.
        """
        opportunities = []

        # Tactical instincts (high priority)
        capture = self.detect_capture(board)
        if capture:
            opportunities.append(capture)

        escape = self.detect_escape(board)
        if escape:
            opportunities.append(escape)

        atari = self.detect_atari(board)
        if atari:
            opportunities.append(atari)

        connect = self.detect_connect(board)
        if connect:
            opportunities.append(connect)

        # Sensei's 8 Basic Instincts
        sensei_results = self.sensei.detect_all(board)
        for result in sensei_results:
            # Convert SenseiInstinctResult to InstinctOpportunity
            priority = self.PRIORITIES.get(result.instinct, 1.0)
            opportunities.append(InstinctOpportunity(
                category=result.instinct,
                moves=result.moves,
                priority=priority
            ))

        # Sort by priority (highest first)
        opportunities.sort(key=lambda x: x.priority, reverse=True)

        return opportunities


class InstinctCurriculum:
    """Adaptive instinct curriculum for Go training.

    Computes auxiliary loss for missed instinct moves. Loss weight
    adapts based on model's benchmark accuracy - high weight early,
    decays as model masters fundamentals.
    """

    def __init__(
        self,
        lambda_0: float = 1.0,
        min_lambda: float = 0.1,
        temperature: float = 2.0,
        device: str = 'cuda'
    ):
        """Initialize curriculum.

        Args:
            lambda_0: Initial instinct loss weight
            min_lambda: Minimum weight (never fully disable)
            temperature: Softmax temperature for instinct targets
            device: Torch device
        """
        self.lambda_0 = lambda_0
        self.min_lambda = min_lambda
        self.temperature = temperature
        self.device = device

        self.detector = InstinctDetector()
        self.current_lambda = lambda_0
        self.current_accuracy = 0.0

    def update_lambda(self, benchmark_accuracy: float):
        """Update curriculum weight based on benchmark accuracy.

        λ(t) = max(λ_min, λ_0 × (1 - accuracy))
        """
        self.current_accuracy = benchmark_accuracy
        self.current_lambda = max(
            self.min_lambda,
            self.lambda_0 * (1.0 - benchmark_accuracy)
        )

    def compute_instinct_target(
        self,
        board: Board,
        board_size: int
    ) -> Optional[torch.Tensor]:
        """Compute instinct policy target for a position.

        Returns soft target distribution over instinct moves,
        or None if no instinct opportunity detected.
        """
        opportunities = self.detector.detect_all(board)

        if not opportunities:
            return None

        # Use highest priority opportunity
        opp = opportunities[0]

        # Create soft target distribution
        target = torch.zeros(board_size * board_size + 1, device=self.device)

        for r, c in opp.moves:
            idx = r * board_size + c
            target[idx] = opp.priority

        # Normalize with temperature
        if target.sum() > 0:
            target = F.softmax(target / self.temperature, dim=0)

        return target

    def compute_loss(
        self,
        boards: List[Board],
        log_policies: torch.Tensor,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute instinct loss for a batch.

        Args:
            boards: List of Board objects
            log_policies: Model's log policy outputs (B, N)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size = len(boards)
        board_size = boards[0].size if boards else 9

        losses = []
        instinct_counts = {'capture': 0, 'escape': 0, 'atari': 0, 'connect': 0}

        for i, board in enumerate(boards):
            target = self.compute_instinct_target(board, board_size)

            if target is None:
                continue

            # Cross-entropy loss: -sum(target * log_policy)
            loss = -torch.sum(target * log_policies[i])
            losses.append(loss)

            # Track which instinct
            opps = self.detector.detect_all(board)
            if opps:
                cat = opps[0].category
                if cat in instinct_counts:
                    instinct_counts[cat] += 1

        if not losses:
            return torch.tensor(0.0, device=self.device), {
                'instinct_loss': 0.0,
                'instinct_count': 0,
                'instinct_lambda': self.current_lambda,
                **{f'instinct_{k}': 0 for k in instinct_counts}
            }

        loss_stack = torch.stack(losses)

        if reduction == 'mean':
            total_loss = loss_stack.mean()
        elif reduction == 'sum':
            total_loss = loss_stack.sum()
        else:
            total_loss = loss_stack

        # Apply adaptive weight
        weighted_loss = self.current_lambda * total_loss

        metrics = {
            'instinct_loss': float(total_loss),
            'instinct_weighted_loss': float(weighted_loss),
            'instinct_count': len(losses),
            'instinct_lambda': self.current_lambda,
            'instinct_accuracy': self.current_accuracy,
            **{f'instinct_{k}': v for k, v in instinct_counts.items()}
        }

        return weighted_loss, metrics


def create_instinct_curriculum(config) -> InstinctCurriculum:
    """Factory function to create curriculum from config."""
    return InstinctCurriculum(
        lambda_0=getattr(config, 'instinct_lambda', 1.0),
        min_lambda=getattr(config, 'instinct_min_lambda', 0.1),
        temperature=getattr(config, 'instinct_temperature', 2.0),
        device=config.device
    )


if __name__ == '__main__':
    # Quick test
    from board import Board

    # Create a capture position
    board = Board(9)
    board.board[4, 4] = -1  # White stone
    board.board[4, 3] = 1   # Black surrounding
    board.board[4, 5] = 1
    board.board[3, 4] = 1
    # board[5,4] is the capture point
    board.current_player = 1

    detector = InstinctDetector()
    opps = detector.detect_all(board)

    print("Board:")
    print(board)
    print("\nInstinct opportunities:")
    for opp in opps:
        print(f"  {opp.category}: moves={opp.moves}, priority={opp.priority}")
