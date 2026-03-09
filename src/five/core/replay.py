from __future__ import annotations

from dataclasses import dataclass

from five.core.move import Move
from five.core.state import GameState


@dataclass(slots=True)
class ReplayFrame:
    ply: int
    state: GameState
    move: Move | None


def reconstruct_states(moves: list[Move], board_size: int, win_length: int) -> list[ReplayFrame]:
    state = GameState.new(board_size=board_size, win_length=win_length)
    frames = [ReplayFrame(ply=0, state=state.copy(), move=None)]
    for ply, move in enumerate(moves, start=1):
        state.apply_move(move)
        frames.append(ReplayFrame(ply=ply, state=state.copy(), move=move))
    return frames
