import torch

from five.common.config import RewardConfig
from five.core.board import Board
from five.core.move import Move
from five.core.state import GameState
from five.train.dataset import EpisodeBatch, Transition
from five.train.reward import compute_process_reward_with_details, find_winning_moves
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

    block_result = compute_process_reward_with_details(board, Move(4, 0), 1)
    ignore_result = compute_process_reward_with_details(board, Move(0, 0), 1)

    assert block_result.total_reward < 0
    assert block_result.total_reward > ignore_result.total_reward
    assert any("延缓对方活四一手" in detail.reason for detail in block_result.details)
    assert any("堵截活四不彻底仍留冲四" in detail.reason for detail in block_result.details)
    assert not any("封堵对方活四" in detail.reason for detail in block_result.details)


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


def test_delaying_open_four_gives_small_positive_credit_without_counting_as_full_block():
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

    assert any("延缓对方活四一手" in detail.reason for detail in result.details)
    assert not any("封堵对方活四" in detail.reason for detail in result.details)
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
    assert any("未压制对方活三" in detail.reason for detail in result.details)
    assert not any("开局" in detail.reason for detail in result.details)
    assert result.total_reward < 0


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
