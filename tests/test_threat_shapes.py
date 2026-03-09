"""
棋形检测自测：验证 analyze_line / get_threat_info 对所有棋形的分类是否正确。

棋形定义（以落子点为中心）：
- 成五：5 子连，任意端
- 活四：4 子连，两端空 .XXXX.
- 冲四：4 子连，一端堵 OXXXX. 或 .XXXXO
- 跳活四：4 子连且中间有跳，两端空 .XX.XX.
- 跳冲四：4 子连且中间有跳，一端堵
- 活三：3 子连，两端空 .XXX.
- 眠三：3 子连，一端堵 OXXX. 或 .XXXO
- 跳活三：3 子连且中间有跳，两端空 .XX.X.
"""
from __future__ import annotations

from five.core.board import Board
from five.core.move import Move
from five.train.reward import analyze_line, get_threat_info


def _place(board: Board, stones: list[tuple[int, int, int]]) -> None:
    for row, col, player in stones:
        board.grid[row, col] = player


def test_five_winning():
    """成五：5 子连成一线"""
    board = Board(size=9, win_length=5)
    _place(board, [(4, 2, 1), (4, 3, 1), (4, 4, 1), (4, 5, 1)])
    move = Move(4, 6)
    board.grid[4, 6] = 1
    info = get_threat_info(board, move, 1)
    assert info.winning_moves >= 1, "成五应被识别"
    assert info.living_fours == 0 and info.blocked_fours == 0


def test_living_four():
    """活四：.XXXX. 两端空"""
    board = Board(size=9, win_length=5)
    _place(board, [(4, 4, 1), (4, 5, 1), (4, 6, 1)])
    move = Move(4, 3)
    board.grid[4, 3] = 1
    info = get_threat_info(board, move, 1)
    assert info.living_fours >= 1, "活四应被识别"
    assert info.blocked_fours == 0
    assert info.winning_moves == 0


def test_blocked_four_one_end():
    """冲四：OXXXX. 一端被对手堵"""
    board = Board(size=9, win_length=5)
    _place(board, [(4, 1, -1), (4, 2, 1), (4, 3, 1), (4, 4, 1)])
    move = Move(4, 5)
    board.grid[4, 5] = 1
    info = get_threat_info(board, move, 1)
    assert info.blocked_fours >= 1, "冲四应被识别"
    assert info.living_fours == 0


def test_blocked_four_other_end():
    """冲四：.XXXXO 另一端被堵"""
    board = Board(size=9, win_length=5)
    _place(board, [(4, 4, 1), (4, 5, 1), (4, 6, 1), (4, 7, -1)])
    move = Move(4, 3)
    board.grid[4, 3] = 1
    info = get_threat_info(board, move, 1)
    assert info.blocked_fours >= 1, "冲四应被识别"
    assert info.living_fours == 0


def test_living_three():
    """活三：.XXX. 两端空"""
    board = Board(size=9, win_length=5)
    _place(board, [(4, 4, 1), (4, 5, 1)])
    move = Move(4, 3)
    board.grid[4, 3] = 1
    info = get_threat_info(board, move, 1)
    assert info.living_threes >= 1, "活三应被识别"
    assert info.blocked_threes == 0


def test_blocked_three_one_end():
    """眠三：OXXX. 一端被对手堵"""
    board = Board(size=9, win_length=5)
    _place(board, [(4, 1, -1), (4, 2, 1), (4, 3, 1)])
    move = Move(4, 4)
    board.grid[4, 4] = 1
    info = get_threat_info(board, move, 1)
    assert info.blocked_threes >= 1, "眠三应被识别"
    assert info.living_threes == 0


def test_blocked_three_other_end():
    """眠三：.XXXO 另一端被堵（用户反馈的 bug 场景）"""
    board = Board(size=9, win_length=5)
    _place(board, [(4, 4, 1), (4, 5, 1), (4, 6, -1)])
    move = Move(4, 3)
    board.grid[4, 3] = 1
    info = get_threat_info(board, move, 1)
    assert info.blocked_threes >= 1, "眠三应被识别"
    assert info.living_threes == 0


def test_blocked_three_board_edge():
    """眠三：一端抵棋盘边，另一端被对手堵"""
    board = Board(size=9, win_length=5)
    _place(board, [(0, 4, 1), (1, 4, 1), (3, 4, -1)])
    move = Move(2, 4)
    board.grid[2, 4] = 1
    info = get_threat_info(board, move, 1)
    assert info.blocked_threes >= 1, "抵边且另一端被堵应为眠三"
    assert info.living_threes == 0


def test_jump_living_three():
    """跳活三：.XX.X. 中间有跳，两端空"""
    board = Board(size=9, win_length=5)
    _place(board, [(4, 3, 1), (4, 4, 1)])
    move = Move(4, 6)
    board.grid[4, 6] = 1
    info = get_threat_info(board, move, 1)
    assert info.jump_living_threes >= 1, "跳活三应被识别"
    assert info.living_threes == 0
    assert info.blocked_threes == 0


def test_jump_living_four():
    """跳活四：.XX.XX. 中间有跳，两端空"""
    board = Board(size=9, win_length=5)
    _place(board, [(4, 3, 1), (4, 4, 1), (4, 6, 1)])
    move = Move(4, 7)
    board.grid[4, 7] = 1
    info = get_threat_info(board, move, 1)
    assert info.jump_living_fours >= 1, "跳活四应被识别"
    assert info.living_fours == 0
    assert info.jump_blocked_fours == 0


def test_jump_blocked_four():
    """跳冲四：OXX.XX. 一端被堵"""
    board = Board(size=9, win_length=5)
    _place(board, [(4, 0, -1), (4, 1, 1), (4, 2, 1), (4, 4, 1)])
    move = Move(4, 5)
    board.grid[4, 5] = 1
    info = get_threat_info(board, move, 1)
    assert info.jump_blocked_fours >= 1, "跳冲四应被识别"
    assert info.jump_living_fours == 0


def test_no_false_living_three_when_blocked():
    """确保眠三不会被误判为活三"""
    board = Board(size=9, win_length=5)
    _place(board, [(4, 4, 1), (4, 5, 1), (4, 6, -1)])
    move = Move(4, 3)
    board.grid[4, 3] = 1
    info = get_threat_info(board, move, 1)
    assert info.living_threes == 0, "一端被堵应为眠三，不是活三"
    assert info.blocked_threes >= 1


def test_analyze_line_open_ends_at_board_edge():
    """跑出棋盘时正确计为开放端"""
    board = Board(size=5, win_length=5)
    _place(board, [(2, 0, 1), (2, 1, 1)])
    move = Move(2, 2)
    board.grid[2, 2] = 1
    line = analyze_line(board, move, 1, 0, 1)
    assert line.count == 3
    assert line.open_ends == 2, "两端都抵边应视为两端开放（或至少一端）"
    # 注：抵边时一端跑出棋盘会+1，另一端也跑出会+1，共2


def test_analyze_line_blocked_by_opponent_no_double_count():
    """遇空位时 open_ends 不重复计数"""
    board = Board(size=9, win_length=5)
    _place(board, [(4, 4, 1), (4, 5, 1), (4, 6, -1)])
    move = Move(4, 3)
    board.grid[4, 3] = 1
    line = analyze_line(board, move, 1, 0, 1)
    assert line.count == 3
    assert line.open_ends == 1, "一端空一端堵，open_ends 应为 1"
    assert line.open_ends != 2, "不应重复计为 2"


def test_living_three_diagonal():
    """活三：斜向 .XXX."""
    board = Board(size=9, win_length=5)
    _place(board, [(4, 4, 1), (5, 5, 1)])
    move = Move(3, 3)
    board.grid[3, 3] = 1
    info = get_threat_info(board, move, 1)
    assert info.living_threes >= 1, "斜向活三应被识别"


def test_blocked_four_vertical():
    """冲四：竖向"""
    board = Board(size=9, win_length=5)
    _place(board, [(2, 4, -1), (3, 4, 1), (4, 4, 1), (5, 4, 1)])
    move = Move(6, 4)
    board.grid[6, 4] = 1
    info = get_threat_info(board, move, 1)
    assert info.blocked_fours >= 1, "竖向冲四应被识别"


def test_jump_living_three_anti_diagonal():
    """跳活三：反斜向 .X.XX."""
    board = Board(size=9, win_length=5)
    _place(board, [(2, 6, 1), (3, 5, 1)])
    move = Move(5, 3)
    board.grid[5, 3] = 1
    info = get_threat_info(board, move, 1)
    assert info.jump_living_threes >= 1, "反斜向跳活三应被识别"
