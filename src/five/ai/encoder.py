from __future__ import annotations

import numpy as np
import torch

from five.core.state import GameState


def encode_state(state: GameState) -> torch.Tensor:
    board = state.board.grid
    current = (board == state.current_player).astype(np.float32)
    opponent = (board == -state.current_player).astype(np.float32)
    last_move = np.zeros_like(board, dtype=np.float32)
    if state.last_move is not None:
        last_move[state.last_move.row, state.last_move.col] = 1.0
    turn_plane = np.full_like(board, 1.0 if state.current_player == 1 else 0.0, dtype=np.float32)
    encoded = np.stack([current, opponent, last_move, turn_plane], axis=0)
    return torch.from_numpy(encoded)
