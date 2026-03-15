from __future__ import annotations

from dataclasses import asdict, dataclass, field

from pathlib import Path


@dataclass(slots=True)
class RewardConfig:
    # 进攻/防守奖励的全局缩放
    attack_scale: float = 0.03
    block_scale: float = 0.035
    # 单步过程奖励与总奖励的裁剪上下界（绝对值）
    max_process_reward: float = 2.5
    max_total_reward: float = 5.0
    # ---------- 开局 ---------- 前 N 步内生效的位置奖惩
    opening_position_horizon: int = 36
    opening_center_bonus: float = 0.12
    opening_edge_penalty: float = 0.8
    opening_corner_penalty: float = 1.2
    opening_center_radius_ratio: float = 0.28
    opening_minor_threat_scale: float = 0.3
    opening_major_threat_scale: float = 0.05
    edge_shape_decay: float = 0.4
    corner_shape_decay: float = 0.25

    # 终局：获胜/平局时的额外奖励
    final_win_reward: float = 3.0
    draw_reward: float = 0.0
    # 终局结果回传：距终局 N 步内的衰减 bonus；只对最后 outcome_horizon 步生效，避免前期好棋被输棋回传压成负分
    outcome_tail_bonus: float = 0.3
    outcome_decay: float = 0.85
    outcome_horizon: int = 6

    # 各棋型的基础分数（再乘 attack_scale/block_scale 得到实际奖励）
    immediate_win_score: float = 100.0
    open_four_score: float = 45.0
    double_four_score: float = 55.0
    four_three_score: float = 40.0
    double_three_score: float = 35.0
    rush_four_score: float = 20.0
    open_three_score: float = 10.0
    jump_open_three_score: float = 7.0
    # 仅能成冲四的连子活三（两端再延一步即边线或敌子）；跳活三中间填跳必成活四，无此类
    restricted_open_three_score: float = 5.0
    sleep_three_score: float = 3.0

    # ---------- 错失 ---------- 己方有机会未把握的惩罚
    miss_own_immediate_win_penalty: float = 1.2
    miss_own_open_four_penalty: float = 1.0
    # ---------- 未阻止 ---------- 对方制胜手或可执行着法未拦住（冲四/跳四由 miss_immediate_win 覆盖；活四双赢点、一手成活四、一手成四三或双活三）
    miss_immediate_win_penalty: float = 2.8
    # 对方下一手成活四/四三/双活三后均近制胜，扣分与制胜手同档
    miss_open_three_penalty: float = 2.2
    miss_jump_open_three_penalty: float = 2.0
    miss_one_move_four_three_penalty: float = 2.2
    miss_one_move_double_three_penalty: float = 2.0


@dataclass(slots=True)
class ModelConfig:
    # 策略/价值网络：卷积通道数；残差块数量
    channels: int = 256
    blocks: int = 16


@dataclass(slots=True)
class TrainingConfig:
    # 棋盘与规则
    board_size: int = 9
    win_length: int = 5
    run_name: str = "ppo_gomoku"
    seed: int = 7
    # 自对弈：每轮对局数；总轮数
    self_play_games_per_epoch: int = 384
    epochs: int = 600
    # PPO 更新：批大小；每轮更新次数；学习率及下限；梯度裁剪
    batch_size: int = 768
    updates_per_epoch: int = 6
    learning_rate: float = 3.5e-4
    lr_min: float = 1.5e-5
    grad_clip_norm: float = 1.5
    # GAE 与折扣
    gamma: float = 0.97
    gae_lambda: float = 0.95
    # PPO clip、value clip、value 损失系数、熵系数
    clip_epsilon: float = 0.25
    value_clip_epsilon: float = 0.25
    value_coef: float = 0.6
    entropy_coef: float = 0.03
    # 采样温度：初始值、下限、在前多少比例轮数内线性衰减到下限
    temperature_init: float = 1.3
    temperature_min: float = 0.35
    temperature_anneal_fraction: float = 0.75
    # 对手采样：历史模型概率、池大小、每隔多少轮取一次快照；启发式对手最大概率及从第几轮开始、多少轮内 ramp 到最大
    historical_opponent_prob: float = 0.4
    opponent_pool_size: int = 80
    opponent_snapshot_interval: int = 3
    heuristic_opponent_max_prob: float = 0.55
    heuristic_start_fraction: float = 0.02
    heuristic_ramp_fraction: float = 0.18
    # 评估与 checkpoint：每轮评估局数；每隔多少轮保存一次模型
    eval_games: int = 48
    checkpoint_every: int = 2
    # 训练设备（cuda/cpu）；运行结果根目录
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
    # 主窗口宽高（像素）；训练监控等轮询间隔（毫秒）
    window_width: int = 1300
    window_height: int = 800
    poll_interval_ms: int = 500


@dataclass(slots=True)
class GenerateConfig:
    """five-generate 默认参数：启发式自博弈数据生成。"""
    games: int = 50000
    board_size: int = 9
    win_length: int = 5
    output: str = "data/heuristic_50k.pt"
    seed: int = 42


@dataclass(slots=True)
class PretrainConfig:
    """five-pretrain 默认参数：行为克隆预训练。"""
    dataset: str = "data/heuristic_50k.pt"
    board_size: int = 9
    channels: int = 256
    blocks: int = 16
    epochs: int = 30
    batch_size: int = 1024
    lr: float = 1e-3
    value_coef: float = 0.5
    device: str = "cuda"
    output_dir: str = "pretrain_output"
    seed: int = 42