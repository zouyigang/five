from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Move:
    row: int
    col: int

    def to_index(self, board_size: int) -> int:
        return self.row * board_size + self.col

    @classmethod
    def from_index(cls, index: int, board_size: int) -> "Move":
        return cls(row=index // board_size, col=index % board_size)
