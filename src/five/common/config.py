from __future__ import annotations

from dataclasses import asdict, dataclass, field

from pathlib import Path


@dataclass(slots=True)
class RewardConfig:
    attack_scale: float = 0.02
    block_scale: float = 0.018
    max_process_reward: float = 1.5
    max_total_reward: float = 2.5
    opening_position_horizon: int = 8
    opening_center_bonus: float = 0.05
    opening_edge_penalty: float = 0.04
    opening_corner_penalty: float = 0.1
    opening_center_radius_ratio: float = 0.18

    final_win_reward: float = 1.0
    draw_reward: float = 0.0
    outcome_tail_bonus: float = 0.35
    outcome_decay: float = 0.9
    outcome_horizon: int = 8

    immediate_win_score: float = 100.0
    open_four_score: float = 45.0
    double_four_score: float = 55.0
    four_three_score: float = 40.0
    double_three_score: float = 35.0
    rush_four_score: float = 20.0
    open_three_score: float = 10.0
    jump_open_three_score: float = 7.0
    sleep_three_score: float = 3.0

    miss_immediate_win_penalty: float = 1.5
    miss_own_immediate_win_penalty: float = 2.0
    delay_open_four_reward: float = 0.08
    miss_open_four_penalty: float = 1.7
    miss_four_three_penalty: float = 0.9
    miss_double_three_penalty: float = 0.9
    miss_rush_four_penalty: float = 1.1
    miss_open_three_penalty: float = 0.25
    miss_jump_open_three_penalty: float = 0.15


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