from __future__ import annotations

from dataclasses import dataclass

from five.core.move import Move
from five.core.state import GameState


@dataclass(slots=True)
class GomokuGame:
    board_size: int = 9
    win_length: int = 5

    def new_game(self) -> GameState:
        return GameState.new(board_size=self.board_size, win_length=self.win_length)

    def step(self, state: GameState, action: int) -> GameState:
        next_state = state.copy()
        next_state.apply_move(Move.from_index(action, state.board.size))
        return next_state
