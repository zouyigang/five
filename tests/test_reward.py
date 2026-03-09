import torch

from five.common.config import RewardConfig
from five.core.board import Board
from five.core.move import Move
from five.core.state import GameState
from five.train.dataset import EpisodeBatch, Transition
from five.train.reward import compute_process_reward_with_details
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


def test_blocking_opponent_immediate_win_is_rewarded():
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

    result = compute_process_reward_with_details(board, Move(4, 5), 1)

    assert result.total_reward > 0
    assert any("成五点" in detail.reason or "活四" in detail.reason for detail in result.details)


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
    assert any("未化解" in detail.reason for detail in result.details)


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
