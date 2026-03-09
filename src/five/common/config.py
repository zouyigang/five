from __future__ import annotations

from dataclasses import asdict, dataclass, field

from pathlib import Path

from five.train.reward import RewardConfig


@dataclass(slots=True)
class ModelConfig:
    channels: int = 128
    blocks: int = 10


@dataclass(slots=True)
class TrainingConfig:
    board_size: int = 9
    win_length: int = 5
    run_name: str = "ppo_gomoku"
    seed: int = 7
    self_play_games_per_epoch: int = 128
    epochs: int = 1000
    batch_size: int = 256
    updates_per_epoch: int = 10
    learning_rate: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    temperature: float = 1.0
    eval_games: int = 8
    checkpoint_every: int = 1
    device: str = "cuda"
    runs_dir: str = "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def runs_path(self) -> Path:
        return Path(self.runs_dir)


@dataclass(slots=True)
class GUIConfig:
    window_width: int = 1300
    window_height: int = 800
    poll_interval_ms: int = 500