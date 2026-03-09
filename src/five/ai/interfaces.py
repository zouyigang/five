from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from five.core.move import Move
from five.core.state import GameState


@dataclass(slots=True)
class CandidateMove:
    move: Move
    score: float
    visits: int | None = None
    value: float | None = None


@dataclass(slots=True)
class AnalysisResult:
    action: Move
    action_probability: float
    value_estimate: float
    candidates: list[CandidateMove]


class AIEngine(Protocol):
    def load_checkpoint(self, path: str) -> None:
        ...

    def select_move(self, state: GameState, temperature: float = 0.0) -> AnalysisResult:
        ...

    def analyze(self, state: GameState, top_k: int = 5) -> list[CandidateMove]:
        ...
