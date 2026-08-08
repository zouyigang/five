import numpy as np
import pytest
import torch

from five.ai.mcts import MCTSConfig, MCTSEngine
from five.ai.model import PolicyValueNet
from five.core.game import GomokuGame
from five.train.alphazero import (
    AlphaZeroConfig,
    TrainingExample,
    _GameBuffer,
    _sample_move,
    play_self_play_games,
    train_on_examples,
)


def _config(**kw) -> AlphaZeroConfig:
    base = dict(
        board_size=9, games_per_iteration=4, simulations=8,
        temperature_moves=2, batch_size=16, updates_per_iteration=1, device="cpu",
    )
    base.update(kw)
    return AlphaZeroConfig(**base)


def _model() -> PolicyValueNet:
    torch.manual_seed(0)
    return PolicyValueNet(board_size=9, channels=8, blocks=1)


# --------------------------------------------------------------- 价值标签视角
def test_value_target_is_from_the_side_to_move_perspective():
    """z 必须按「该局面走子方是否获胜」，而不是某个固定颜色——与 encode_state 一致。"""
    buffer = _GameBuffer()
    for player in (1, -1, 1, -1):
        buffer.add(torch.zeros(4, 9, 9), np.zeros(81, dtype=np.float32), player)

    examples = buffer.finish(winner=1)

    assert [e.value for e in examples] == [1.0, -1.0, 1.0, -1.0]


def test_draw_gives_zero_value_to_both_sides():
    buffer = _GameBuffer()
    for player in (1, -1):
        buffer.add(torch.zeros(4, 9, 9), np.zeros(81, dtype=np.float32), player)

    assert [e.value for e in buffer.finish(winner=0)] == [0.0, 0.0]


# --------------------------------------------------------------- 落子采样
def test_zero_temperature_takes_the_most_visited_move():
    policy = np.zeros(81, dtype=np.float32)
    policy[40] = 0.6
    policy[41] = 0.4

    import random

    assert _sample_move(policy, 0.0, random.Random(0)) == 40


def test_sampling_never_picks_an_unvisited_move():
    import random

    policy = np.zeros(81, dtype=np.float32)
    policy[10] = 0.5
    policy[20] = 0.5
    rng = random.Random(1)

    picked = {_sample_move(policy, 1.0, rng) for _ in range(50)}

    assert picked <= {10, 20}
    assert len(picked) == 2, "温度 1.0 应在两个点之间产生多样性"


# --------------------------------------------------------------- 自博弈
def test_self_play_produces_one_example_per_move():
    game = GomokuGame(board_size=9, win_length=5)
    engine = MCTSEngine(_model(), device="cpu", config=MCTSConfig(simulations=8, seed=3))
    config = _config()
    import random

    examples, stats = play_self_play_games(game, engine, 4, config, random.Random(0))

    assert stats.games == 4
    assert len(examples) == stats.moves
    assert stats.black_wins + stats.white_wins + stats.draws == 4


def test_self_play_policies_are_valid_distributions():
    game = GomokuGame(board_size=9, win_length=5)
    engine = MCTSEngine(_model(), device="cpu", config=MCTSConfig(simulations=8, seed=3))
    import random

    examples, _ = play_self_play_games(game, engine, 2, _config(), random.Random(0))

    for example in examples:
        assert example.policy.shape == (81,)
        assert example.policy.min() >= 0.0
        assert example.policy.sum() == pytest.approx(1.0, abs=1e-5)
        assert example.value in (-1.0, 0.0, 1.0)


def test_self_play_games_all_reach_a_terminal_state():
    game = GomokuGame(board_size=9, win_length=5)
    engine = MCTSEngine(_model(), device="cpu", config=MCTSConfig(simulations=8, seed=3))
    import random

    _, stats = play_self_play_games(game, engine, 4, _config(), random.Random(0))

    # 每局至少 9 手（成五的理论最短），至多填满棋盘
    assert 9 <= stats.average_length <= 81


# --------------------------------------------------------------- 训练
def test_training_reduces_loss_on_a_repeated_example():
    """监督目标应当能被拟合：反复喂同一批样本，损失必须下降。"""
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    policy = np.zeros(81, dtype=np.float32)
    policy[40] = 1.0
    examples = [
        TrainingExample(state=torch.zeros(4, 9, 9), policy=policy, value=1.0)
        for _ in range(32)
    ]
    config = _config(updates_per_iteration=1)

    first = train_on_examples(model, optimizer, examples, config, torch.device("cpu"))
    for _ in range(15):
        last = train_on_examples(model, optimizer, examples, config, torch.device("cpu"))

    assert last.policy_loss < first.policy_loss
    assert last.value_loss < first.value_loss


def test_policy_head_learns_to_match_the_visit_distribution():
    """把全部访问压在一个点上，训练后网络的最高概率点应当就是它。"""
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    policy = np.zeros(81, dtype=np.float32)
    policy[17] = 1.0
    examples = [
        TrainingExample(state=torch.zeros(4, 9, 9), policy=policy, value=0.0)
        for _ in range(32)
    ]

    for _ in range(30):
        train_on_examples(model, optimizer, examples, _config(), torch.device("cpu"))

    with torch.no_grad():
        logits, _ = model(torch.zeros(1, 4, 9, 9))
    assert int(torch.argmax(logits)) == 17


def test_training_on_empty_examples_is_a_noop():
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    stats = train_on_examples(model, optimizer, [], _config(), torch.device("cpu"))

    assert stats.policy_loss == 0.0 and stats.value_loss == 0.0


def test_self_play_from_a_fresh_network_is_not_first_player_degenerate():
    """自博弈退化检测：若对局塌到理论最短且一方全胜，价值标签就没有信息了。

    实测对比（24 局）：随机初始 @32 模拟为 11:13、均长 36.3；而用 PPO 的 best.pt
    起步是 24:0、均长 10.8——监督起点的攻守偏置会被 z 标签自我强化，故默认从随机
    权重开始。这里只对随机起点做断言，因为那才是被支持的用法。
    """
    import random

    game = GomokuGame(board_size=9, win_length=5)
    torch.manual_seed(0)
    model = PolicyValueNet(board_size=9, channels=16, blocks=2)
    engine = MCTSEngine(model, device="cpu", config=MCTSConfig(simulations=16, seed=5))

    _, stats = play_self_play_games(game, engine, 12, _config(games_per_iteration=12), random.Random(0))

    assert stats.average_length > 12, f"对局塌到 {stats.average_length:.1f} 手，接近理论最短"
    assert min(stats.black_wins, stats.white_wins) > 0 or stats.draws > 0, (
        f"一方全胜（黑 {stats.black_wins} 白 {stats.white_wins}），价值标签退化为「轮到谁走」"
    )


def test_value_targets_are_not_perfectly_predictable_from_turn_parity():
    """若 z 完全由走子方奇偶决定，价值头会以 0 损失拟合而学不到局面。"""
    import random

    game = GomokuGame(board_size=9, win_length=5)
    torch.manual_seed(0)
    model = PolicyValueNet(board_size=9, channels=16, blocks=2)
    engine = MCTSEngine(model, device="cpu", config=MCTSConfig(simulations=16, seed=5))

    examples, _ = play_self_play_games(game, engine, 12, _config(games_per_iteration=12), random.Random(0))

    values = {e.value for e in examples}
    assert len(values) > 1, "所有样本的 z 相同，没有可学的差异"


def test_augmentation_produces_eight_consistent_symmetries():
    """增广必须同时变换局面与标签，否则等于喂噪声。"""
    from five.train.alphazero import augment

    state = torch.zeros(4, 9, 9)
    state[0, 0, 1] = 1.0  # 己方一子在 (0,1)
    policy = np.zeros(81, dtype=np.float32)
    policy[0 * 9 + 1] = 1.0  # 标签也指向 (0,1)
    example = TrainingExample(state=state, policy=policy, value=0.5)

    variants = augment(example)

    assert len(variants) == 8
    for variant in variants:
        assert variant.value == 0.5
        assert variant.policy.sum() == pytest.approx(1.0)
        # 标签指向的格子，在变换后的局面里必须仍是那颗己方子
        index = int(np.argmax(variant.policy))
        assert variant.state[0].flatten()[index] == 1.0


def test_augmentation_covers_eight_distinct_orientations():
    from five.train.alphazero import augment

    state = torch.zeros(4, 9, 9)
    state[0, 0, 1] = 1.0
    state[1, 2, 0] = 1.0  # 加一个不对称的点，避免自身对称导致重复
    policy = np.zeros(81, dtype=np.float32)
    policy[1] = 1.0
    variants = augment(TrainingExample(state=state, policy=policy, value=0.0))

    signatures = {tuple(v.state[0].flatten().tolist()) for v in variants}
    assert len(signatures) == 8, f"只得到 {len(signatures)} 种朝向，增广有重复"


def test_augmentation_can_be_switched_off():
    config = _config(augment_symmetries=False)
    assert config.augment_symmetries is False
