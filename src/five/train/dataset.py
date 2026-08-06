from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from five.core.board import Board
from five.core.move import Move


@dataclass(slots=True)
class Transition:
    state: torch.Tensor
    action: int
    old_log_prob: float
    reward: float
    done: bool
    value: float
    player: int
    legal_mask: torch.Tensor
    board_before: Board | None = None
    move: Move | None = None
    move_record_index: int | None = None


@dataclass(slots=True)
class EpisodeBatch:
    transitions: list[Transition] = field(default_factory=list)

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def compute_returns_and_advantages(
        self,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按**走子方**分别计算 GAE，再散回原位置。

        价值头输出的是「当前走子方」的期望胜负（`encode_state` 的己方/对方平面相对
        走子方），所以只有同一方的两个价值才在同一坐标系里。

        对手局只记录模型一方，相邻 transition 天然是同一方隔两手的连续决策，整条
        序列同框；自博弈局双方都记录，相邻 transition 每步换边，若直接用
        `V(s_{t+1}) - V(s_t)`，减掉的是对手视角的价值，优势函数整个失真。

        拆分后每条子序列都退化成与对手局相同的结构，因此对手局的结果与拆分前逐位
        一致，只有自博弈局被纠正。子序列末尾即该方在本局的最后一次决策，按终止处理。
        """
        count = len(self.transitions)
        advantages = np.zeros(count, dtype=np.float32)
        values = np.array([transition.value for transition in self.transitions], dtype=np.float32)

        indices_by_player: dict[int, list[int]] = {}
        for index, transition in enumerate(self.transitions):
            indices_by_player.setdefault(transition.player, []).append(index)

        for indices in indices_by_player.values():
            last_advantage = 0.0
            for position in reversed(range(len(indices))):
                index = indices[position]
                is_final = position == len(indices) - 1
                next_non_terminal = 0.0 if is_final else 1.0
                next_value = 0.0 if is_final else values[indices[position + 1]]
                delta = (
                    self.transitions[index].reward
                    + gamma * next_value * next_non_terminal
                    - values[index]
                )
                last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
                advantages[index] = last_advantage

        returns = advantages + values
        return torch.from_numpy(returns), torch.from_numpy(advantages)
