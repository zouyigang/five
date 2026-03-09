from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from five.core.move import Move
from five.core.rules import DIRECTIONS, in_bounds


@dataclass(slots=True)
class Board:
    size: int
    win_length: int = 5
    grid: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)

    def copy(self) -> "Board":
        copied = Board(size=self.size, win_length=self.win_length)
        copied.grid = self.grid.copy()
        return copied

    def is_legal(self, move: Move) -> bool:
        return in_bounds(self.size, move.row, move.col) and self.grid[move.row, move.col] == 0

    def apply_move(self, move: Move, player: int) -> None:
        if not self.is_legal(move):
            raise ValueError(f"Illegal move: {move}")
        self.grid[move.row, move.col] = player

    def legal_moves(self) -> list[Move]:
        rows, cols = np.where(self.grid == 0)
        return [Move(int(row), int(col)) for row, col in zip(rows.tolist(), cols.tolist())]

    def legal_mask(self) -> np.ndarray:
        return (self.grid.reshape(-1) == 0).astype(np.float32)

    def move_count(self) -> int:
        return int(np.count_nonzero(self.grid))

    def is_full(self) -> bool:
        return self.move_count() == self.size * self.size

    def check_winner(self, move: Move) -> int:
        player = int(self.grid[move.row, move.col])
        if player == 0:
            return 0
        for delta_row, delta_col in DIRECTIONS:
            total = 1
            total += self._count_direction(move, player, delta_row, delta_col)
            total += self._count_direction(move, player, -delta_row, -delta_col)
            if total >= self.win_length:
                return player
        return 0

    def _count_direction(self, move: Move, player: int, delta_row: int, delta_col: int) -> int:
        count = 0
        row, col = move.row + delta_row, move.col + delta_col
        while in_bounds(self.size, row, col) and self.grid[row, col] == player:
            count += 1
            row += delta_row
            col += delta_col
        return count
