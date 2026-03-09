from __future__ import annotations

import math
import random
from dataclasses import dataclass

from five.ai.interfaces import AIEngine, AnalysisResult, CandidateMove
from five.core.move import Move
from five.core.state import GameState


@dataclass(slots=True)
class RandomPlayer:
    name: str = "random"

    def select_move(self, state: GameState) -> AnalysisResult:
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


@dataclass(slots=True)
class HeuristicPlayer:
    name: str = "heuristic"

    def select_move(self, state: GameState) -> AnalysisResult:
        scored = []
        center = (state.board.size - 1) / 2.0
        for move in state.legal_moves():
            distance = math.dist((move.row, move.col), (center, center))
            score = -distance
            scored.append((score, move))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_move = scored[0]
        candidates = [CandidateMove(move=move, score=float(score)) for score, move in scored[:5]]
        return AnalysisResult(
            action=best_move,
            action_probability=1.0,
            value_estimate=best_score,
            candidates=candidates,
        )


@dataclass(slots=True)
class EnginePlayer:
    engine: AIEngine
    name: str = "model"

    def select_move(self, state: GameState, temperature: float = 0.0) -> AnalysisResult:
        return self.engine.select_move(state, temperature=temperature)
