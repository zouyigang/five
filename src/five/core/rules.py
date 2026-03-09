from __future__ import annotations

from typing import Iterable


DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (1, 1),
    (1, -1),
)


def in_bounds(size: int, row: int, col: int) -> bool:
    return 0 <= row < size and 0 <= col < size


def iter_line(
    size: int,
    row: int,
    col: int,
    delta_row: int,
    delta_col: int,
) -> Iterable[tuple[int, int]]:
    current_row, current_col = row + delta_row, col + delta_col
    while in_bounds(size, current_row, current_col):
        yield current_row, current_col
        current_row += delta_row
        current_col += delta_col
