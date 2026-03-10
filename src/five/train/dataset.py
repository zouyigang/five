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
        rewards = [transition.reward for transition in self.transitions]
        values = [transition.value for transition in self.transitions] + [0.0]
        dones = [transition.done for transition in self.transitions]
        advantages = np.zeros(len(self.transitions), dtype=np.float32)
        last_advantage = 0.0
        for step in reversed(range(len(self.transitions))):
            next_non_terminal = 1.0 - float(dones[step])
            delta = rewards[step] + gamma * values[step + 1] * next_non_terminal - values[step]
            last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            advantages[step] = last_advantage
        returns = advantages + np.asarray(values[:-1], dtype=np.float32)
        return torch.from_numpy(returns), torch.from_numpy(advantages)
