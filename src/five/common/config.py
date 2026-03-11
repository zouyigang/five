from __future__ import annotations

from dataclasses import asdict, dataclass, field

from pathlib import Path


@dataclass(slots=True)
class RewardConfig:
    attack_scale: float = 0.03
    block_scale: float = 0.025
    max_process_reward: float = 1.5
    max_total_reward: float = 4.0
    opening_position_horizon: int = 12
    opening_center_bonus: float = 0.08
    opening_edge_penalty: float = 0.12
    opening_corner_penalty: float = 0.25
    opening_center_radius_ratio: float = 0.22

    final_win_reward: float = 3.0
    draw_reward: float = 0.0
    outcome_tail_bonus: float = 0.5
    outcome_decay: float = 0.92
    outcome_horizon: int = 12

    immediate_win_score: float = 100.0
    open_four_score: float = 45.0
    double_four_score: float = 55.0
    four_three_score: float = 40.0
    double_three_score: float = 35.0
    rush_four_score: float = 20.0
    open_three_score: float = 10.0
    jump_open_three_score: float = 7.0
    sleep_three_score: float = 3.0

    miss_immediate_win_penalty: float = 0.6
    miss_own_immediate_win_penalty: float = 0.8
    delay_open_four_reward: float = 0.1
    miss_open_four_penalty: float = 0.6
    miss_four_three_penalty: float = 0.3
    miss_double_three_penalty: float = 0.3
    miss_rush_four_penalty: float = 0.4
    miss_open_three_penalty: float = 0.1
    miss_jump_open_three_penalty: float = 0.05


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
    updates_per_epoch: int = 4
    learning_rate: float = 2e-4
    lr_min: float = 1e-5
    grad_clip_norm: float = 1.0
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.05
    temperature_init: float = 1.0
    temperature_min: float = 0.5
    temperature_anneal_fraction: float = 0.8
    historical_opponent_prob: float = 0.75
    opponent_pool_size: int = 50
    opponent_snapshot_interval: int = 5
    heuristic_opponent_max_prob: float = 0.25
    heuristic_start_fraction: float = 0.15
    heuristic_ramp_fraction: float = 0.5
    eval_games: int = 32
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