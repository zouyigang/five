from five.core.board import Board
from five.core.move import Move
from five.core.replay import reconstruct_states
from five.core.state import GameState


def test_horizontal_win_detection():
    state = GameState.new(board_size=9, win_length=5)
    black_moves = [Move(4, col) for col in range(5)]
    white_moves = [Move(3, col) for col in range(4)]
    for black_move, white_move in zip(black_moves[:-1], white_moves):
        state.apply_move(black_move)
        state.apply_move(white_move)
    state.apply_move(black_moves[-1])
    assert state.winner == 1


def test_board_legal_mask_counts_empty_cells():
    board = Board(size=5, win_length=5)
    board.apply_move(Move(0, 0), 1)
    board.apply_move(Move(1, 1), -1)
    assert int(board.legal_mask().sum()) == 23


def test_replay_reconstructs_last_frame():
    moves = [Move(0, 0), Move(1, 1), Move(0, 1)]
    frames = reconstruct_states(moves, board_size=5, win_length=3)
    assert len(frames) == 4
    assert frames[-1].state.board.grid[0, 1] == 1
