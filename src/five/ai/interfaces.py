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


def select_moves_batched(
    engine: AIEngine,
    states: list[GameState],
    temperature: float = 0.0,
) -> list[AnalysisResult]:
    """对一批局面求走子，能批处理的引擎走批处理，否则逐个回退。

    神经网络引擎单条前向严重浪费 GPU（batch=1 每步 3.45ms，batch=256 每局面
    0.076ms）；启发式/随机引擎是纯 CPU 逐点打分，批处理没有意义，回退即可。
    """
    batched = getattr(engine, "select_moves", None)
    if callable(batched):
        return batched(states, temperature=temperature)
    return [engine.select_move(state, temperature=temperature) for state in states]
