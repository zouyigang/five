"""AlphaZero 式训练：用 MCTS 的搜索结果当监督标签，不用塑形奖励。

与 PPO 路径的根本差别：

| | PPO（trainer.py） | 本模块 |
|---|---|---|
| 落子 | 网络直出采样 | MCTS 搜索 |
| 策略目标 | 优势加权 + 裁剪比值 | 交叉熵拟合访问分布 π |
| 价值目标 | GAE 回报 | 终局胜负 z（+1/-1/0） |
| 奖励 | 20+ 项手工塑形 | 无，只用胜负 |

不复用 `self_play.py`：那里的 `tracked_players`、对手混合、塑形奖励回填在这里都不需要，
硬凑成一个函数只会让「改 A 弄坏 B」变成常态。两条路径本就该走不同逻辑。

**价值标签的视角**必须与 `encode_state` 一致：平面 0/1 是相对走子方的，所以 z 取
「该局面走子方最终是否获胜」，而不是某个固定颜色。
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from five.ai.encoder import encode_state
from five.ai.mcts import MCTSConfig, MCTSEngine
from five.ai.model import PolicyValueNet
from five.common.logging import get_logger
from five.core.game import GomokuGame
from five.core.move import Move
from five.core.state import GameState

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class AlphaZeroConfig:
    board_size: int = 9
    win_length: int = 5
    # 自博弈
    games_per_iteration: int = 128
    simulations: int = 64
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25
    # 前 N 手按访问分布采样，之后贪心。探索集中在开局，中后盘保持满强度。
    temperature_moves: int = 8
    # 模拟数不足会让自博弈退化：挡棋所需的搜索量远大于攻棋（实测随机网络下相差 32 倍），
    # 模拟太少时防守方根本挡不住，对局塌到理论最短的 9~11 手，价值头只要看「轮到谁走」
    # 就能满分预测，等于没有信号。实测 24 局的黑/白胜负与均长：
    #   随机初始 @32 -> 11:13, 36.3 手   @400 -> 12:12, 33.8 手（健康）
    #   best.pt  @32 -> 24: 0, 10.8 手   @400 -> 19: 0, 28.7 手（退化）
    # 训练
    iterations: int = 200
    batch_size: int = 512
    updates_per_iteration: int = 8
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    value_coef: float = 1.0
    grad_clip_norm: float = 1.5
    # 经验回放：保留最近多少局的样本。只用当代数据会严重欠采样，
    # 且策略每代跳变会让价值头拟合不稳。
    replay_games: int = 2048
    device: str = "cuda"
    seed: int = 7


@dataclass(slots=True)
class TrainingExample:
    """一个训练样本：局面、MCTS 访问分布、该局面走子方的终局结果。"""

    state: torch.Tensor
    policy: np.ndarray
    value: float


@dataclass(slots=True)
class SelfPlayStats:
    games: int = 0
    moves: int = 0
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0

    @property
    def average_length(self) -> float:
        return self.moves / max(self.games, 1)

    @property
    def draw_rate(self) -> float:
        return self.draws / max(self.games, 1)


class _GameBuffer:
    """一局进行中的记录：局面 + 访问分布 + 走子方，终局后再回填 z。"""

    __slots__ = ("states", "policies", "players")

    def __init__(self) -> None:
        self.states: list[torch.Tensor] = []
        self.policies: list[np.ndarray] = []
        self.players: list[int] = []

    def add(self, state: torch.Tensor, policy: np.ndarray, player: int) -> None:
        self.states.append(state)
        self.policies.append(policy)
        self.players.append(player)

    def finish(self, winner: int) -> list[TrainingExample]:
        # z 取该局面**走子方**视角：与 encode_state 的己方/对方平面保持一致。
        return [
            TrainingExample(
                state=state,
                policy=policy,
                value=0.0 if winner == 0 else (1.0 if player == winner else -1.0),
            )
            for state, policy, player in zip(self.states, self.policies, self.players)
        ]


def _sample_move(policy: np.ndarray, temperature: float, rng: random.Random) -> int:
    """按访问分布选点；温度 <= 0 时取访问最多的那个。"""
    if temperature <= 1e-6:
        return int(np.argmax(policy))
    weights = policy ** (1.0 / temperature)
    total = float(weights.sum())
    if total <= 0:
        legal = np.flatnonzero(policy > 0)
        return int(rng.choice(legal.tolist())) if legal.size else int(np.argmax(policy))
    weights = weights / total
    return int(rng.choices(range(len(weights)), weights=weights.tolist(), k=1)[0])


def play_self_play_games(
    game: GomokuGame,
    engine: MCTSEngine,
    games: int,
    config: AlphaZeroConfig,
    rng: random.Random,
) -> tuple[list[TrainingExample], SelfPlayStats]:
    """并行推进一批自博弈，返回训练样本与统计。

    所有对局同步走子，每一拍把全部未终局的局面凑成一次批量搜索——单局串行搜索会让
    每次模拟退化成 batch=1 前向，慢一到两个数量级。
    """
    states = [game.new_game() for _ in range(games)]
    buffers = [_GameBuffer() for _ in range(games)]
    move_index = [0] * games
    stats = SelfPlayStats(games=games)

    while True:
        active = [i for i, state in enumerate(states) if not state.is_terminal]
        if not active:
            break
        results = engine.search([states[i] for i in active])
        for index, result in zip(active, results):
            state = states[index]
            buffers[index].add(encode_state(state), result.policy.copy(), state.current_player)
            temperature = 1.0 if move_index[index] < config.temperature_moves else 0.0
            action = _sample_move(result.policy, temperature, rng)
            state.apply_move(Move.from_index(action, state.board.size))
            move_index[index] += 1
            stats.moves += 1

    examples: list[TrainingExample] = []
    for state, buffer in zip(states, buffers):
        examples.extend(buffer.finish(state.winner))
        if state.winner == 1:
            stats.black_wins += 1
        elif state.winner == -1:
            stats.white_wins += 1
        else:
            stats.draws += 1
    return examples, stats


@dataclass(slots=True)
class LossStats:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total: float = 0.0


def train_on_examples(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    examples: list[TrainingExample],
    config: AlphaZeroConfig,
    device: torch.device,
) -> LossStats:
    """监督拟合：策略走交叉熵（软标签），价值走 MSE。

    没有优势、没有裁剪比值、没有 KL 锚定——目标是固定的监督信号，不会像 PPO 那样
    因为行为策略与目标策略不一致而失真。
    """
    if not examples:
        return LossStats()

    states = torch.stack([e.state for e in examples]).to(device)
    policies = torch.from_numpy(np.stack([e.policy for e in examples])).to(device)
    values = torch.tensor([e.value for e in examples], dtype=torch.float32, device=device)

    model.train()
    stats = LossStats()
    batches = 0
    count = states.size(0)
    for _ in range(config.updates_per_iteration):
        permutation = torch.randperm(count, device=device)
        for start in range(0, count, config.batch_size):
            index = permutation[start : start + config.batch_size]
            logits, predicted = model(states[index])
            target = policies[index]
            # 软标签交叉熵：-sum(π * log p)。π 已按访问数归一化，非法点为 0。
            policy_loss = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            value_loss = nn.functional.mse_loss(predicted.flatten(), values[index])
            loss = policy_loss + config.value_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            stats.policy_loss += float(policy_loss.item())
            stats.value_loss += float(value_loss.item())
            stats.total += float(loss.item())
            batches += 1

    if batches:
        stats.policy_loss /= batches
        stats.value_loss /= batches
        stats.total /= batches
    model.eval()
    return stats


class AlphaZeroTrainer:
    """AlphaZero 训练循环。

    **默认从随机权重开始，不要用监督/PPO 检查点预热。** 那些模型带着自己的偏置——
    实测用 best.pt 起步时自博弈黑胜 24:0、对局塌到 10.8 手（成五理论最短是 9 手），
    而 z 标签只记「谁赢了」，会把这个偏置自我强化；随机初始起步则是 11:13、36.3 手。
    checkpoint_path 保留下来只为续训自己的 AlphaZero run。
    """

    def __init__(self, config: AlphaZeroConfig, checkpoint_path: str | None = None) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.game = GomokuGame(board_size=config.board_size, win_length=config.win_length)
        self.model = PolicyValueNet(board_size=config.board_size).to(self.device)
        if checkpoint_path:
            self._load(checkpoint_path)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.replay: deque[TrainingExample] = deque(maxlen=self._replay_capacity())
        self.rng = random.Random(config.seed)

    def _replay_capacity(self) -> int:
        # 按「局数 x 每局平均手数」估容量；对局长度会随训练变化，取一个宽裕的估计。
        return self.config.replay_games * self.config.board_size

    def _load(self, checkpoint_path: str) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        saved = (payload.get("config") or {}).get("model") or {}
        channels = int(saved.get("channels", 64))
        blocks = int(saved.get("blocks", 6))
        self.model = PolicyValueNet(
            board_size=self.config.board_size, channels=channels, blocks=blocks
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        LOGGER.info("Loaded %s (%dx%d)", checkpoint_path, channels, blocks)

    def _engine(self) -> MCTSEngine:
        return MCTSEngine(
            self.model,
            device=self.config.device,
            config=MCTSConfig(
                simulations=self.config.simulations,
                c_puct=self.config.c_puct,
                dirichlet_alpha=self.config.dirichlet_alpha,
                dirichlet_weight=self.config.dirichlet_weight,
            ),
        )

    def run_iteration(self) -> tuple[SelfPlayStats, LossStats]:
        self.model.eval()
        examples, stats = play_self_play_games(
            self.game, self._engine(), self.config.games_per_iteration, self.config, self.rng
        )
        self.replay.extend(examples)
        losses = train_on_examples(
            self.model, self.optimizer, list(self.replay), self.config, self.device
        )
        return stats, losses
