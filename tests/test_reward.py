import random

import pytest
import torch

from five.common.config import RewardConfig
from five.core.board import Board
from five.core.move import Move
from five.core.state import GameState
from five.train.dataset import EpisodeBatch, Transition
from five.train.reward import (
    _scan_existing_threat_inventory,
    compute_process_reward_with_details,
    find_winning_moves,
    scan_threat_instances,
)
from five.train.self_play import _apply_hybrid_rewards


def _place(board: Board, stones: list[tuple[int, int, int]]) -> Board:
    for row, col, player in stones:
        board.grid[row, col] = player
    return board


def test_attack_reward_for_open_three_is_positive():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 4, 1),
            (4, 5, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(4, 3), 1)

    assert result.total_reward > 0
    assert any("活三" in detail.reason for detail in result.details)


def test_blocking_opponent_open_four_is_better_than_ignoring_it():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 1, -1),
            (4, 2, -1),
            (4, 3, -1),
            (4, 4, -1),
        ],
    )

    config = RewardConfig(max_process_reward=5.0, opening_position_horizon=0)
    block_result = compute_process_reward_with_details(board, Move(4, 0), 1, config)
    ignore_result = compute_process_reward_with_details(board, Move(0, 0), 1, config)

    assert block_result.total_reward > ignore_result.total_reward
    assert any("封堵对方活四" in detail.reason for detail in block_result.details)
    assert any("未阻止对方制胜手" in detail.reason for detail in block_result.details)


def test_missing_opponent_immediate_win_is_penalized():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 0, 1),
            (4, 1, -1),
            (4, 2, -1),
            (4, 3, -1),
            (4, 4, -1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(0, 0), 1)

    assert result.total_reward < 0
    assert any("未阻止对方制胜手" in detail.reason for detail in result.details)


def test_missing_opponent_open_four_is_penalized():
    """对方有活四（双赢点）且未堵时，应扣「未阻止对方制胜手」。"""
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 1, -1),
            (4, 2, -1),
            (4, 3, -1),
            (4, 4, -1),
            (3, 2, 1),
            (3, 3, 1),
            (3, 4, 1),
            (3, 5, 1),
        ],
    )
    # 白(1)有活四可赢，黑(-1)有活四可赢。白下(5,2)既错失己方获胜又漏防对方活四
    result = compute_process_reward_with_details(board, Move(5, 2), 1)
    assert result.total_reward < 0
    assert any("未阻止对方制胜手" in detail.reason for detail in result.details)
    assert any("错失直接获胜落点" in detail.reason for detail in result.details)


def test_no_miss_penalty_for_only_potential_future_threes():
    board = _place(
        Board(size=9, win_length=5),
        [
            (3, 3, 1),
            (3, 4, 1),
            (4, 3, -1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(4, 4), -1)

    assert not any("未压制对方活三" in detail.reason for detail in result.details)
    assert not any("未压制对方跳活三" in detail.reason for detail in result.details)


def test_blocking_existing_open_three_is_still_rewarded():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 3, 1),
            (4, 4, 1),
            (4, 5, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(4, 2), -1)

    assert result.total_reward > 0
    assert any("封堵对方活三" in detail.reason for detail in result.details)


def test_missing_own_immediate_win_is_penalized_more_than_open_three_reward():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 1, 1),
            (4, 2, 1),
            (4, 3, 1),
            (4, 4, 1),
            (3, 6, 1),
            (3, 7, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(3, 5), 1)

    assert result.total_reward < 0
    assert any("错失直接获胜落点" in detail.reason for detail in result.details)


def test_jump_five_shape_is_not_treated_as_immediate_winning_move():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 1, 1),
            (4, 2, 1),
            (4, 4, 1),
            (4, 5, 1),
        ],
    )

    winning_moves = find_winning_moves(board, 1)

    assert Move(4, 3) in winning_moves
    assert Move(4, 0) not in winning_moves
    assert Move(4, 6) not in winning_moves


def test_false_jump_five_endpoint_does_not_bypass_missed_own_win_penalty():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 1, 1),
            (4, 2, 1),
            (4, 4, 1),
            (4, 5, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(4, 0), 1)

    assert any("错失直接获胜落点" in detail.reason for detail in result.details)


def test_winning_move_is_not_penalized_for_unresolved_opponent_open_three():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 0, 1),
            (4, 1, 1),
            (4, 2, 1),
            (4, 3, 1),
            (2, 2, -1),
            (2, 3, -1),
            (2, 4, -1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(4, 4), 1)

    assert not any("未压制对方活三" in detail.reason for detail in result.details)
    assert not any("未压制对方跳活三" in detail.reason for detail in result.details)


def test_blocking_opponent_winning_point_does_not_double_count_as_rush_four():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 4, 1),
            (3, 5, -1),
            (3, 4, 1),
            (2, 4, -1),
            (3, 3, 1),
            (4, 6, -1),
            (5, 4, 1),
            (5, 7, -1),
            (6, 4, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(7, 4), -1)

    assert any("封堵对方冲四/跳四" in detail.reason for detail in result.details)
    assert not any("封堵对方直接成五点" in detail.reason for detail in result.details)
    assert result.total_reward < 0


def test_partially_blocking_open_four_keeps_credit_and_miss_penalty():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 1, -1),
            (4, 2, -1),
            (4, 3, -1),
            (4, 4, -1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(4, 0), 1)

    assert any("封堵对方活四" in detail.reason for detail in result.details)
    assert any("未阻止对方制胜手" in detail.reason for detail in result.details)
    assert result.total_reward < 0


def test_move_20_partial_block_still_penalizes_remaining_winning_moves():
    """截图局面：白 20 封住活四一端，但黑仍有两个直接制胜点。"""
    board = _place(
        Board(size=9, win_length=5),
        [
            # 黑棋（奇数手）
            (2, 2, 1),
            (4, 4, 1),
            (3, 5, 1),
            (6, 4, 1),
            (3, 4, 1),
            (4, 5, 1),
            (4, 6, 1),
            (2, 6, 1),
            (5, 4, 1),
            (1, 7, 1),
            # 白棋（偶数手，第 20 手之前）
            (4, 3, -1),
            (1, 4, -1),
            (5, 6, -1),
            (6, 3, -1),
            (4, 2, -1),
            (6, 6, -1),
            (2, 7, -1),
            (5, 5, -1),
            (5, 3, -1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(2, 4), -1)

    assert any("封堵对方活四" in detail.reason for detail in result.details)
    assert any("未阻止对方制胜手" in detail.reason for detail in result.details)
    assert result.total_reward < 0


def test_missing_opponent_rush_four_penalised_alongside_open_four_gain():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 4, 1),
            (4, 3, -1),
            (3, 4, 1),
            (5, 4, -1),
            (2, 4, 1),
            (6, 5, -1),
            (3, 2, 1),
            (1, 4, -1),
            (3, 3, 1),
            (7, 6, -1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(3, 5), 1)

    assert any("形成活四" in detail.reason for detail in result.details)
    assert any("未阻止对方制胜手" in detail.reason for detail in result.details)
    attack = sum(d.amount for d in result.details if d.amount > 0)
    penalty = sum(d.amount for d in result.details if d.amount < 0)
    assert attack > 0
    assert penalty < 0


def test_double_three_uses_primary_shape_reward_without_open_three_stacking():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 4, 1),
            (4, 5, 1),
            (3, 3, 1),
            (5, 3, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(4, 3), 1)

    assert any("形成双活三" in detail.reason for detail in result.details)
    assert not any("形成活三" in detail.reason for detail in result.details)


# 白棋第 14 手的实战局面：黑棋 (5,6) 可一手成双活三，白棋 (4,3) 只成跳活三未解危。
_COUNTER_THREAT_STONES = [
    (4, 4, 1), (3, 5, -1), (3, 4, 1), (2, 4, -1), (5, 4, 1), (6, 4, -1),
    (4, 5, 1), (4, 6, -1), (5, 7, 1), (1, 3, -1), (0, 2, 1), (3, 3, -1),
    (4, 2, 1),
]


def test_counter_attacking_open_three_halves_double_three_miss_penalty():
    board = _place(Board(size=9, win_length=5), _COUNTER_THREAT_STONES)
    config = RewardConfig(counter_threat_waiver_scale=0.5)

    counter = compute_process_reward_with_details(board, Move(4, 3), -1, config)
    idle = compute_process_reward_with_details(board, Move(6, 6), -1, config)

    miss = next(d for d in counter.details if "未阻止对方一手成双活三" in d.reason)
    assert miss.amount == -config.miss_one_move_double_three_penalty * 0.5
    assert "本手反击活三" in miss.reason
    # 反击虽仍是次优（不解危），但必须和毫无意义的一手明显拉开差距。
    assert counter.total_reward < 0
    assert counter.total_reward - idle.total_reward > 1.0


def test_counter_threat_waiver_does_not_apply_without_own_open_three():
    board = _place(Board(size=9, win_length=5), _COUNTER_THREAT_STONES)
    config = RewardConfig(counter_threat_waiver_scale=0.5)

    result = compute_process_reward_with_details(board, Move(6, 6), -1, config)

    miss = next(d for d in result.details if "未阻止对方一手成双活三" in d.reason)
    assert miss.amount == -config.miss_one_move_double_three_penalty
    assert "本手反击活三" not in miss.reason


def test_counter_threat_waiver_scale_of_one_keeps_full_penalty():
    board = _place(Board(size=9, win_length=5), _COUNTER_THREAT_STONES)
    config = RewardConfig(counter_threat_waiver_scale=1.0)

    result = compute_process_reward_with_details(board, Move(4, 3), -1, config)

    miss = next(d for d in result.details if "未阻止对方一手成双活三" in d.reason)
    assert miss.amount == -config.miss_one_move_double_three_penalty


def test_move_that_defuses_double_three_still_beats_pure_counter_attack():
    board = _place(Board(size=9, win_length=5), _COUNTER_THREAT_STONES)
    config = RewardConfig(counter_threat_waiver_scale=0.5)

    # (5,5) 既成活三又消掉黑棋的双活三点，必须仍优于只反击的 (4,3)。
    defuse = compute_process_reward_with_details(board, Move(5, 5), -1, config)
    counter = compute_process_reward_with_details(board, Move(4, 3), -1, config)

    assert defuse.total_reward > counter.total_reward
    assert not any("未阻止" in d.reason for d in defuse.details)


def test_rush_four_still_gets_full_waiver_not_partial():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 4, 1), (4, 5, 1), (3, 3, 1), (5, 3, 1),
            (1, 1, -1), (1, 2, -1), (1, 3, -1),
        ],
    )
    config = RewardConfig(counter_threat_waiver_scale=0.5)

    # 白棋 (1,4) 成冲四/活四：绝对先手，漏防惩罚应完全消失而非折半。
    result = compute_process_reward_with_details(board, Move(1, 4), -1, config)

    assert not any("未阻止对方一手成" in d.reason for d in result.details)


def test_opening_center_move_is_rewarded():
    config = RewardConfig(
        opening_center_bonus=0.05,
        opening_edge_penalty=0.04,
        opening_corner_penalty=0.1,
    )
    board = Board(size=9, win_length=5)

    result = compute_process_reward_with_details(board, Move(4, 4), 1, config)

    assert result.total_reward > 0
    assert any("开局中心落子奖励" == detail.reason for detail in result.details)


def test_opening_edge_move_is_penalized():
    config = RewardConfig(
        opening_center_bonus=0.05,
        opening_edge_penalty=0.04,
        opening_corner_penalty=0.1,
    )
    board = Board(size=9, win_length=5)

    result = compute_process_reward_with_details(board, Move(0, 4), 1, config)

    assert result.total_reward < 0
    assert any("开局边线落子惩罚" == detail.reason for detail in result.details)


def test_opening_corner_penalty_is_stronger_than_edge():
    config = RewardConfig(
        opening_center_bonus=0.05,
        opening_edge_penalty=0.04,
        opening_corner_penalty=0.1,
    )
    board = Board(size=9, win_length=5)

    edge_result = compute_process_reward_with_details(board, Move(0, 4), 1, config)
    corner_result = compute_process_reward_with_details(board, Move(0, 0), 1, config)

    assert corner_result.total_reward < edge_result.total_reward
    assert any("开局角落落子惩罚" == detail.reason for detail in corner_result.details)


def test_opening_position_reward_only_applies_in_first_eight_plies():
    config = RewardConfig(opening_position_horizon=8)
    board = _place(
        Board(size=9, win_length=5),
        [
            (0, 0, 1),
            (1, 2, -1),
            (2, 4, 1),
            (3, 6, -1),
            (4, 8, 1),
            (5, 1, -1),
            (6, 3, 1),
            (7, 5, -1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(4, 4), 1, config)

    assert not any("开局" in detail.reason for detail in result.details)


def test_opening_position_reward_can_stack_with_shape_reward():
    config = RewardConfig(
        opening_center_bonus=0.05,
        opening_edge_penalty=0.04,
        opening_corner_penalty=0.1,
    )
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 4, 1),
            (4, 5, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(4, 3), 1, config)

    assert any("形成活三" in detail.reason for detail in result.details)
    assert any("开局中心落子奖励" == detail.reason for detail in result.details)
    assert result.total_reward > config.opening_center_bonus


def test_opening_position_reward_is_suppressed_when_ignoring_opponent_open_three():
    """漏防对方活三时必有漏防惩罚且总分为负；开局位置奖惩仍可叠加（漏防+走边会同时扣分）。"""
    board = _place(
        Board(size=9, win_length=5),
        [
            (3, 3, 1),
            (4, 4, -1),
            (3, 4, 1),
            (4, 5, -1),
            (3, 5, 1),
            (5, 3, -1),
            (5, 5, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(4, 3), -1)

    assert any("形成活三" in detail.reason for detail in result.details)
    assert any("未阻止对方一手成活四" in detail.reason for detail in result.details)
    assert result.total_reward < 0


def test_miss_open_three_plus_edge_gets_both_penalties():
    """漏防对方活三且走边线时，同时扣漏防与边线惩罚。"""
    board = _place(
        Board(size=9, win_length=5),
        [
            (3, 3, 1),
            (4, 4, -1),
            (3, 4, 1),
            (4, 5, -1),
            (3, 5, 1),
            (5, 3, -1),
            (5, 5, 1),
        ],
    )
    result = compute_process_reward_with_details(board, Move(0, 4), -1)
    assert any("未阻止对方一手成活四" in detail.reason for detail in result.details)
    assert any("开局边线落子惩罚" in detail.reason for detail in result.details)
    assert result.total_reward < -0.4


def test_opening_second_ring_move_is_penalized_less_than_outer_edge():
    """次外圈（第1行/列）受半额边线惩罚，且弱于最外圈。"""
    config = RewardConfig(
        opening_center_bonus=0.05,
        opening_edge_penalty=0.40,
        opening_corner_penalty=0.50,
    )
    board = Board(size=9, win_length=5)

    outer_result = compute_process_reward_with_details(board, Move(0, 4), 1, config)
    second_result = compute_process_reward_with_details(board, Move(1, 4), 1, config)

    assert any("开局边线落子惩罚" == d.reason for d in outer_result.details)
    assert any("开局次边线落子惩罚" == d.reason for d in second_result.details)
    assert second_result.total_reward > outer_result.total_reward
    assert second_result.total_reward < 0


def test_second_ring_shape_decay_is_milder_than_outer_edge():
    """次外圈棋型折减弱于最外圈。"""
    config = RewardConfig(
        attack_scale=0.1,
        opening_position_horizon=0,
        edge_shape_decay=0.65,
    )
    outer_board = _place(
        Board(size=9, win_length=5),
        [(0, 4, 1), (0, 5, 1)],
    )
    second_board = _place(
        Board(size=9, win_length=5),
        [(1, 4, 1), (1, 5, 1)],
    )

    outer_result = compute_process_reward_with_details(outer_board, Move(0, 3), 1, config)
    second_result = compute_process_reward_with_details(second_board, Move(1, 3), 1, config)

    assert any("边线棋型价值折减" in d.reason for d in outer_result.details)
    assert any("次边线棋型价值折减" in d.reason for d in second_result.details)
    assert second_result.total_reward > outer_result.total_reward


def test_opening_position_reward_is_strongly_reduced_in_tactical_blocking_positions():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 4, 1),
            (3, 4, -1),
            (5, 4, 1),
            (3, 5, -1),
            (6, 4, 1),
            (3, 6, -1),
            (7, 4, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(8, 4), -1)

    assert any("封堵对方冲四/跳四" in detail.reason for detail in result.details)
    assert any("开局位置权重降低（对手强威胁" in detail.reason for detail in result.details)
    assert any("开局" in detail.reason for detail in result.details)


def test_opening_position_reward_is_softened_when_blocking_open_three():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 3, 1),
            (4, 4, 1),
            (4, 5, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(4, 2), -1)

    assert any("封堵对方活三" in detail.reason for detail in result.details)
    assert any("开局位置权重降低（对手牵制威胁" in detail.reason for detail in result.details)
    assert any("开局" in detail.reason for detail in result.details)


def test_edge_open_three_reward_is_discounted_relative_to_center():
    config = RewardConfig(
        attack_scale=0.1,
        opening_position_horizon=0,
        edge_shape_decay=0.9,
        corner_shape_decay=0.75,
    )
    center_board = _place(
        Board(size=9, win_length=5),
        [
            (4, 4, 1),
            (4, 5, 1),
        ],
    )
    edge_board = _place(
        Board(size=9, win_length=5),
        [
            (0, 4, 1),
            (0, 5, 1),
        ],
    )

    center_result = compute_process_reward_with_details(center_board, Move(4, 3), 1, config)
    edge_result = compute_process_reward_with_details(edge_board, Move(0, 3), 1, config)

    assert center_result.total_reward > edge_result.total_reward
    assert any("边线棋型价值折减" in detail.reason for detail in edge_result.details)


def test_open_four_reward_is_not_discounted_on_edge():
    config = RewardConfig(
        attack_scale=0.1,
        opening_position_horizon=0,
        edge_shape_decay=0.5,
        corner_shape_decay=0.5,
    )
    board = _place(
        Board(size=9, win_length=5),
        [
            (0, 1, 1),
            (0, 2, 1),
            (0, 3, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(0, 4), 1, config)

    assert any("形成活四" in detail.reason for detail in result.details)
    assert not any("棋型价值折减" in detail.reason for detail in result.details)


def test_miss_own_open_four_penalty_triggers_when_no_opponent_threat():
    board = _place(
        Board(size=9, win_length=5),
        [
            (3, 3, 1),
            (4, 4, -1),
            (3, 4, 1),
            (4, 5, -1),
            (3, 5, 1),
            (4, 2, -1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(4, 6), 1)

    assert any("错失形成活四/必胜棋型" in detail.reason for detail in result.details)
    assert result.total_reward < 0


def test_miss_own_open_four_penalty_suppressed_when_opponent_has_winning_move():
    board = _place(
        Board(size=9, win_length=5),
        [
            (3, 3, 1),
            (3, 4, 1),
            (3, 5, 1),
            (0, 0, -1),
            (0, 1, -1),
            (0, 2, -1),
            (0, 3, -1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(0, 4), 1)

    assert any("封堵对方冲四/跳四" in detail.reason for detail in result.details)
    assert not any("错失形成活四" in detail.reason for detail in result.details)
    assert result.total_reward > 0


def test_missed_own_win_suppresses_attack_reward():
    """错失直接获胜时不再叠加进攻奖励，总分必为负。"""
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 0, 1),
            (4, 1, 1),
            (4, 2, 1),
            (4, 3, 1),
            (3, 5, 1),
            (3, 6, 1),
            (3, 7, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(3, 4), 1)

    assert any("错失直接获胜落点" in d.reason for d in result.details)
    assert not any(d.amount > 0 for d in result.details)
    assert result.total_reward < 0
    assert result.missed_own_win is True


def test_forming_own_open_four_when_opponent_has_winning_move_is_negative():
    board = _place(
        Board(size=9, win_length=5),
        [
            (3, 3, 1),
            (3, 4, 1),
            (3, 5, 1),
            (0, 0, -1),
            (0, 1, -1),
            (0, 2, -1),
            (0, 3, -1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(3, 6), 1)

    assert any("形成活四" in detail.reason for detail in result.details)
    assert any("未阻止对方制胜手" in detail.reason for detail in result.details)
    assert result.total_reward < 0


def test_self_play_rewards_use_terminal_tail_instead_of_uniform_winner_bonus():
    config = RewardConfig()
    state = GameState.new(board_size=5, win_length=3)
    episode = EpisodeBatch()
    moves = [Move(0, 0), Move(1, 0), Move(0, 1), Move(1, 1), Move(0, 2)]

    for move in moves:
        board_before = state.board.copy()
        episode.add(
            Transition(
                state=torch.zeros((4, 5, 5), dtype=torch.float32),
                action=move.to_index(state.board.size),
                old_log_prob=0.0,
                reward=0.0,
                done=False,
                value=0.0,
                player=state.current_player,
                legal_mask=torch.from_numpy(state.legal_mask()),
                board_before=board_before,
                move=move,
            )
        )
        state.apply_move(move)

    rewards = _apply_hybrid_rewards(episode, winner=1, config=config)

    assert rewards[-1][0] > 0
    assert rewards[-1][0] > rewards[0][0]
    assert rewards[1][0] < 0
    assert rewards[3][0] < 0


def test_details_sum_to_total_when_missing_own_win_while_blocking():
    """错失获胜的一手若同时封堵了对方，封堵奖励要留在明细里（曾被按正负筛掉）。"""
    board = _place(
        Board(size=9, win_length=5),
        [
            # 黑棋 (2,0)-(2,3) 成四，(2,4) 是直接获胜点
            (2, 0, 1), (2, 1, 1), (2, 2, 1), (2, 3, 1),
            # 白棋活三，(6,2) 可封堵
            (6, 3, -1), (6, 4, -1), (6, 5, -1),
            (5, 2, 1), (4, 2, 1),
            (0, 8, -1), (1, 8, -1), (8, 8, -1), (8, 7, -1), (0, 6, -1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(6, 2), 1)

    assert any("错失" in detail.reason for detail in result.details)
    assert any("封堵" in detail.reason for detail in result.details)
    assert not any("形成" in detail.reason for detail in result.details)
    assert sum(d.amount for d in result.details) == pytest.approx(result.total_reward, abs=1e-6)


def test_opening_detail_amounts_match_scaled_total():
    """对手有威胁时开局分被缩放，明细金额必须同步缩放。"""
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 3, -1), (4, 4, -1), (4, 5, -1),
            (2, 2, 1), (6, 6, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(4, 2), 1)

    assert any("开局" in detail.reason for detail in result.details)
    assert sum(d.amount for d in result.details) == pytest.approx(result.total_reward, abs=1e-6)


def test_reward_details_always_sum_to_total_reward():
    """模糊测试：任意局面下明细合计都必须等于总分。"""
    rng = random.Random(7)
    checked = 0
    for _ in range(120):
        board = Board(size=9, win_length=5)
        player = 1
        for _ in range(rng.randrange(2, 30)):
            empties = [(r, c) for r in range(9) for c in range(9) if board.grid[r, c] == 0]
            if not empties:
                break
            row, col = rng.choice(empties)
            board.grid[row, col] = player
            if board.check_winner(Move(row, col)) != 0:
                board.grid[row, col] = 0  # 保持非终局局面
                continue
            player = -player

        empties = [(r, c) for r in range(9) for c in range(9) if board.grid[r, c] == 0]
        for row, col in rng.sample(empties, min(4, len(empties))):
            for side in (1, -1):
                result = compute_process_reward_with_details(board, Move(row, col), side)
                total = sum(d.amount for d in result.details)
                checked += 1
                assert total == pytest.approx(result.total_reward, abs=1e-6), (
                    f"move=({row},{col}) player={side} "
                    f"details={[(d.amount, d.reason) for d in result.details]}"
                )
    assert checked > 500


def _rush_four_spam_board() -> Board:
    """黑棋可在 (0,4) 走一个远离战场的冲四；白棋在 row 4 有现成活三。"""
    return _place(
        Board(size=9, win_length=5),
        [
            (0, 0, -1), (0, 1, 1), (0, 2, 1), (0, 3, 1),
            (4, 3, -1), (4, 4, -1), (4, 5, -1),
            (8, 8, 1), (7, 7, -1), (8, 0, 1), (7, 0, -1), (6, 8, 1),
        ],
    )


def test_rush_four_attack_reward_is_discounted_when_it_waives_a_miss_penalty():
    board = _rush_four_spam_board()

    result = compute_process_reward_with_details(board, Move(0, 4), 1)

    attack = [d for d in result.details if "形成冲四" in d.reason]
    assert len(attack) == 1
    assert "强攻豁免漏防" in attack[0].reason
    expected = (
        RewardConfig().rush_four_score
        * RewardConfig().attack_scale
        * RewardConfig().rush_four_waiver_attack_scale
    )
    assert attack[0].amount == pytest.approx(expected, abs=1e-6)


def test_rush_four_keeps_full_attack_reward_when_there_is_nothing_to_waive():
    board = _place(
        Board(size=9, win_length=5),
        [
            (0, 0, -1), (0, 1, 1), (0, 2, 1), (0, 3, 1),
            (7, 7, -1), (8, 8, 1), (6, 0, -1), (8, 1, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(0, 4), 1)

    attack = [d for d in result.details if "形成冲四" in d.reason]
    assert len(attack) == 1
    assert "豁免" not in attack[0].reason
    expected = RewardConfig().rush_four_score * RewardConfig().attack_scale
    assert attack[0].amount == pytest.approx(expected, abs=1e-6)


def test_four_three_attack_reward_is_not_discounted_by_the_waiver():
    """四三近乎制胜，不该因为豁免漏防而被折减——折减只针对光杆冲四。"""
    board = _place(
        Board(size=9, win_length=5),
        [
            (2, 0, -1), (2, 1, 1), (2, 2, 1), (2, 3, 1),
            (3, 5, 1), (4, 6, 1),
            (7, 2, -1), (7, 3, -1), (7, 4, -1),
            (0, 8, -1), (8, 8, -1), (6, 0, 1), (0, 6, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(2, 4), 1)

    assert not any("豁免漏防" in d.reason for d in result.details)
    expected = RewardConfig().four_three_score * RewardConfig().attack_scale
    assert any(d.amount == pytest.approx(expected, abs=1e-6) for d in result.details)


def test_blocking_beats_rush_four_spam_which_still_beats_doing_nothing():
    board = _rush_four_spam_board()

    block = compute_process_reward_with_details(board, Move(4, 2), 1).total_reward
    spam = compute_process_reward_with_details(board, Move(0, 4), 1).total_reward
    idle = compute_process_reward_with_details(board, Move(5, 5), 1).total_reward

    assert block > spam > idle


def _two_disjoint_open_threes() -> Board:
    """白棋在 row 2 和 row 6 各有一个互不相交的活三。"""
    return _place(
        Board(size=9, win_length=5),
        [
            (2, 2, -1), (2, 3, -1), (2, 4, -1),
            (6, 2, -1), (6, 3, -1), (6, 4, -1),
            (0, 8, 1), (1, 8, 1), (8, 0, 1), (8, 1, 1), (4, 4, 1), (0, 0, 1),
        ],
    )


def test_threat_instances_are_deduplicated_per_line_not_per_stone():
    """一个活三由 3 颗子组成，会被每颗子各扫到一次，必须归并为 1 个实例。"""
    board = _place(Board(size=9, win_length=5), [(4, 3, -1), (4, 4, -1), (4, 5, -1)])

    instances = scan_threat_instances(board, -1)

    assert len(instances) == 1
    assert instances[0].kind == "living_three"
    assert instances[0].stones == frozenset({(4, 3), (4, 4), (4, 5)})
    assert instances[0].category == "open_three"


def test_threat_inventory_counts_instances_instead_of_binary_flags():
    inventory = _scan_existing_threat_inventory(_two_disjoint_open_threes(), -1)

    assert inventory.get("open_three") == 2


def test_two_plain_open_threes_are_recognised_as_a_double_three():
    """二值化时两个直活三同属 open_three 被压成 1，判不出双活三。"""
    inventory = _scan_existing_threat_inventory(_two_disjoint_open_threes(), -1)

    assert inventory.get("double_three") == 1


def test_blocking_one_of_two_open_threes_beats_ignoring_both():
    board = _two_disjoint_open_threes()

    blocked = compute_process_reward_with_details(board, Move(2, 1), 1)
    ignored = compute_process_reward_with_details(board, Move(4, 0), 1)

    assert blocked.total_reward > ignored.total_reward
    assert any("封堵对方活三" in d.reason for d in blocked.details)


def test_miss_penalty_scales_with_the_number_of_unblocked_threats():
    board = _two_disjoint_open_threes()

    ignored = compute_process_reward_with_details(board, Move(4, 0), 1)

    assert any("未阻止对方一手成活四 x2" in d.reason for d in ignored.details)


def test_move_defusing_two_threats_at_once_is_credited_for_both():
    board = _place(
        Board(size=9, win_length=5),
        [
            (4, 5, -1), (4, 6, -1), (4, 7, -1),
            (5, 4, -1), (6, 4, -1), (7, 4, -1),
            (0, 0, 1), (0, 1, 1), (8, 8, 1), (1, 0, 1),
        ],
    )

    result = compute_process_reward_with_details(board, Move(4, 4), 1)

    assert any("封堵对方活三 x2" in d.reason for d in result.details)


def _loss_penalty_task(winner: int, players: list[int]):
    """构造一个只关心终局项的最小 RewardTask（网格为 None -> 过程奖励为 0）。"""
    from five.train.self_play import RewardTask

    config = RewardConfig()
    config.final_loss_penalty = 3.0  # 默认已关闭，这里显式打开以测试机制本身
    return RewardTask(
        winner=winner,
        board_size=9,
        win_length=5,
        config=config,
        steps=[(None, 0, 0, p) for p in players],
    )


def test_loser_last_move_gets_the_terminal_loss_penalty():
    """启用时必须落在输家的最后一手上（默认关闭，见 test_loss_penalty_is_off_by_default）。"""
    from five.train.self_play import compute_episode_rewards

    # 黑(1)获胜；白(-1)的最后一手是 index 3
    results = compute_episode_rewards(_loss_penalty_task(1, [1, -1, 1, -1, 1]))

    reasons = [[d.reason for d in details] for _, details in results]
    assert "终局失败惩罚" in reasons[3]
    assert sum("终局失败惩罚" in r for r in reasons) == 1, "只应扣在输家最后一手上"
    penalty = [d.amount for d in results[3][1] if d.reason == "终局失败惩罚"][0]
    assert penalty == pytest.approx(-3.0)


def test_draw_has_no_loss_penalty():
    from five.train.self_play import compute_episode_rewards

    results = compute_episode_rewards(_loss_penalty_task(0, [1, -1, 1, -1]))

    assert not any("终局失败惩罚" in d.reason for _, details in results for d in details)


def test_loss_penalty_is_off_by_default_to_avoid_draw_seeking():
    """默认关闭：开到 3.0 时 draw_reward=0 使「求和」远优于「求胜」，48 局 42 和。

    「赢不了就尽快输」的解药是 TrainingConfig.kl_coef 的锚定，不是这一项。
    """
    assert RewardConfig().final_loss_penalty == 0.0
