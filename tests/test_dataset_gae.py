import numpy as np
import pytest
import torch

from five.train.dataset import EpisodeBatch, Transition

GAMMA = 0.97
LAMBDA = 0.95


def _transition(player: int, reward: float, value: float, done: bool = False) -> Transition:
    return Transition(
        state=torch.zeros(4, 9, 9),
        action=0,
        old_log_prob=0.0,
        reward=reward,
        done=done,
        value=value,
        player=player,
        legal_mask=torch.ones(81),
    )


def _episode(transitions: list[Transition]) -> EpisodeBatch:
    batch = EpisodeBatch()
    for transition in transitions:
        batch.add(transition)
    batch.transitions[-1].done = True
    return batch


def _legacy_gae(rewards, values, dones, gamma=GAMMA, gae_lambda=LAMBDA):
    """修复前的单序列公式，用来验证「对手局结果不变」。"""
    values = list(values) + [0.0]
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last = 0.0
    for step in reversed(range(len(rewards))):
        non_terminal = 1.0 - float(dones[step])
        delta = rewards[step] + gamma * values[step + 1] * non_terminal - values[step]
        last = delta + gamma * gae_lambda * non_terminal * last
        advantages[step] = last
    return advantages


def test_single_player_episode_matches_the_previous_formula():
    """对手局只记录模型一方，整条序列同框，结果必须与修复前逐位相同。"""
    rewards = [0.1, -0.2, 0.35, 1.5]
    values = [0.2, 0.05, -0.1, 0.6]
    episode = _episode([_transition(1, r, v) for r, v in zip(rewards, values)])

    _, advantages = episode.compute_returns_and_advantages(GAMMA, LAMBDA)

    expected = _legacy_gae(rewards, values, [False, False, False, True])
    assert advantages.numpy() == pytest.approx(expected, abs=1e-6)


def test_self_play_episode_is_split_per_player():
    """自博弈局相邻 transition 换边，两方各自成链，互不污染。"""
    # 黑白交替：player 序列 [1, -1, 1, -1]
    episode = _episode([
        _transition(1, 0.1, 0.20),
        _transition(-1, -0.3, -0.50),
        _transition(1, 0.4, 0.30),
        _transition(-1, -1.0, -0.70),
    ])

    _, advantages = episode.compute_returns_and_advantages(GAMMA, LAMBDA)
    got = advantages.numpy()

    black = _legacy_gae([0.1, 0.4], [0.20, 0.30], [False, True])
    white = _legacy_gae([-0.3, -1.0], [-0.50, -0.70], [False, True])
    assert got[0] == pytest.approx(black[0], abs=1e-6)
    assert got[2] == pytest.approx(black[1], abs=1e-6)
    assert got[1] == pytest.approx(white[0], abs=1e-6)
    assert got[3] == pytest.approx(white[1], abs=1e-6)


def test_winning_move_gets_positive_advantage_in_self_play():
    """回归：跨边相减会让**获胜的那一手**拿到负优势，等于在惩罚赢棋。

    黑方最后一手成五拿到 +3.0 终局奖励，白方对应 -3.0。修复前该手的优势被白方的
    负价值污染而变号，梯度方向与「赢棋是好事」相反。
    """
    rewards = [0.0, 0.0, 3.0, -3.0]
    values = [0.10, -0.60, 0.80, -0.90]
    episode = _episode([
        _transition(1, rewards[0], values[0]),
        _transition(-1, rewards[1], values[1]),
        _transition(1, rewards[2], values[2]),   # 黑：制胜一手
        _transition(-1, rewards[3], values[3]),  # 白：输棋
    ])

    _, advantages = episode.compute_returns_and_advantages(GAMMA, LAMBDA)
    got = advantages.numpy()

    assert got[2] > 0, "制胜一手的优势必须为正"
    assert got[0] > 0, "赢棋一方的前一手也应为正"
    assert got[3] < 0, "输棋一方的最后一手应为负"

    legacy = _legacy_gae(rewards, values, [False, False, False, True])
    assert legacy[2] < 0, "修复前制胜一手的优势是负的，正是这个 bug"


def test_returns_stay_consistent_with_values_plus_advantages():
    episode = _episode([
        _transition(1, 0.2, 0.3),
        _transition(-1, -0.1, -0.2),
        _transition(1, 0.5, 0.4),
    ])

    returns, advantages = episode.compute_returns_and_advantages(GAMMA, LAMBDA)

    values = np.array([0.3, -0.2, 0.4], dtype=np.float32)
    assert returns.numpy() == pytest.approx(advantages.numpy() + values, abs=1e-6)
