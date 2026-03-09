from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MCTSConfig:
    simulations: int = 64
    c_puct: float = 1.25


class MCTSPlayer:
    """Reserved interface for the future AlphaZero-lite upgrade."""

    def __init__(self, config: MCTSConfig | None = None) -> None:
        self.config = config or MCTSConfig()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("MCTS is reserved for the future AlphaZero-lite upgrade.")
