from __future__ import annotations

import math
import random
from dataclasses import dataclass

from five.ai.interfaces import AIEngine, AnalysisResult, CandidateMove
from five.core.board import Board
from five.core.move import Move
from five.core.rules import DIRECTIONS, in_bounds
from five.core.state import GameState


@dataclass(slots=True)
class RandomPlayer:
    name: str = "random"

    def load_checkpoint(self, path: str) -> None:
        pass

    def select_move(self, state: GameState, temperature: float = 0.0) -> AnalysisResult:
        moves = state.legal_moves()
        choice = random.choice(moves)
        probability = 1.0 / len(moves)
        candidates = [CandidateMove(move=move, score=probability) for move in moves[:5]]
        return AnalysisResult(
            action=choice,
            action_probability=probability,
            value_estimate=0.0,
            candidates=candidates,
        )

    def analyze(self, state: GameState, top_k: int = 5) -> list[CandidateMove]:
        moves = state.legal_moves()
        probability = 1.0 / max(len(moves), 1)
        return [CandidateMove(move=m, score=probability) for m in moves[:top_k]]


def _count_line(board: Board, move: Move, player: int, dr: int, dc: int) -> tuple[int, int]:
    """Count consecutive stones and open ends in one direction pair."""
    count = 1
    open_ends = 0
    for sign in (1, -1):
        r, c = move.row + sign * dr, move.col + sign * dc
        while in_bounds(board.size, r, c) and board.grid[r, c] == player:
            count += 1
            r += sign * dr
            c += sign * dc
        if in_bounds(board.size, r, c) and board.grid[r, c] == 0:
            open_ends += 1
    return count, open_ends


def _score_move_for_heuristic(board: Board, move: Move, player: int) -> float:
    """Evaluate a single move with tactical shape analysis."""
    score = 0.0
    opponent = -player
    center = (board.size - 1) / 2.0
    distance = math.dist((move.row, move.col), (center, center))
    score += max(0, 5.0 - distance) * 0.5

    for target, multiplier in [(player, 1.0), (opponent, 0.9)]:
        board.grid[move.row, move.col] = target
        for dr, dc in DIRECTIONS:
            count, open_ends = _count_line(board, move, target, dr, dc)
            if count >= 5:
                score += 100000 * multiplier
            elif count == 4 and open_ends == 2:
                score += 10000 * multiplier
            elif count == 4 and open_ends == 1:
                score += 5000 * multiplier
            elif count == 3 and open_ends == 2:
                score += 1000 * multiplier
            elif count == 3 and open_ends == 1:
                score += 100 * multiplier
            elif count == 2 and open_ends == 2:
                score += 50 * multiplier
        board.grid[move.row, move.col] = 0

    return score


@dataclass(slots=True)
class HeuristicPlayer:
    name: str = "heuristic"

    def load_checkpoint(self, path: str) -> None:
        pass

    def select_move(self, state: GameState, temperature: float = 0.0) -> AnalysisResult:
        board = state.board
        player = state.current_player
        legal = state.legal_moves()

        scored = [(_score_move_for_heuristic(board, m, player), m) for m in legal]
        scored.sort(key=lambda item: item[0], reverse=True)

        best_score, best_move = scored[0]
        candidates = [CandidateMove(move=m, score=float(s)) for s, m in scored[:5]]
        return AnalysisResult(
            action=best_move,
            action_probability=1.0,
            value_estimate=best_score,
            candidates=candidates,
        )

    def analyze(self, state: GameState, top_k: int = 5) -> list[CandidateMove]:
        board = state.board
        player = state.current_player
        legal = state.legal_moves()
        scored = [(_score_move_for_heuristic(board, m, player), m) for m in legal]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [CandidateMove(move=m, score=float(s)) for s, m in scored[:top_k]]


@dataclass(slots=True)
class EnginePlayer:
    engine: AIEngine
    name: str = "model"

    def select_move(self, state: GameState, temperature: float = 0.0) -> AnalysisResult:
        return self.engine.select_move(state, temperature=temperature)
