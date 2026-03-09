from __future__ import annotations

from dataclasses import dataclass, field

from five.core.board import Board
from five.core.move import Move


@dataclass(slots=True)
class GameState:
    board: Board
    current_player: int = 1
    last_move: Move | None = None
    winner: int = 0
    history: list[Move] = field(default_factory=list)

    @classmethod
    def new(cls, board_size: int = 9, win_length: int = 5) -> "GameState":
        return cls(board=Board(size=board_size, win_length=win_length))

    def copy(self) -> "GameState":
        return GameState(
            board=self.board.copy(),
            current_player=self.current_player,
            last_move=self.last_move,
            winner=self.winner,
            history=list(self.history),
        )

    @property
    def is_terminal(self) -> bool:
        return self.winner != 0 or self.board.is_full()

    def legal_moves(self) -> list[Move]:
        return self.board.legal_moves()

    def legal_mask(self):
        return self.board.legal_mask()

    def apply_move(self, move: Move) -> None:
        if self.is_terminal:
            raise ValueError("Cannot apply a move to a terminal game state.")
        self.board.apply_move(move, self.current_player)
        self.history.append(move)
        self.last_move = move
        self.winner = self.board.check_winner(move)
        if not self.is_terminal:
            self.current_player *= -1
