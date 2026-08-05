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
    # 活四/双四/四三/双活三：下一手大概率能赢，奖励放大
    immediate_win_score: float = 100.0
    open_four_score: float = 65.0
    double_four_score: float = 75.0
    four_three_score: float = 60.0
    double_three_score: float = 55.0
    rush_four_score: float = 20.0
    open_three_score: float = 10.0
    jump_open_three_score: float = 7.0
    # 仅能成冲四的连子活三（两端再延一步即边线或敌子）；跳活三中间填跳必成活四，无此类
    restricted_open_three_score: float = 5.0
    sleep_three_score: float = 3.0

    # 封堵专用分数：直接消除制胜手（冲四/活三等）高于堵了对方还有（活四/双四等）
    # 活四/双四/四三/双活三：堵一端对方仍有制胜手，分数减半
    block_open_four_score: float = 10.0
    block_double_four_score: float = 10.0
    block_four_three_score: float = 10.0
    block_double_three_score: float = 10.0
    block_rush_four_score: float = 55.0
    block_open_three_score: float = 50.0
    block_jump_open_three_score: float = 45.0
    block_restricted_open_three_score: float = 40.0

    # ---------- 错失 ---------- 己方有机会未把握的惩罚
    # miss_own_immediate_win_penalty=2.5 确保错失直接获胜时总为负（含封堵活四）
    miss_own_immediate_win_penalty: float = 2.5
    # 需 > 封堵活三奖励(50*0.035=1.75)，确保错失形成活四时总为负
    miss_own_open_four_penalty: float = 2.0
    # ---------- 未阻止 ---------- 对方制胜手或可执行着法未拦住（冲四/跳四由 miss_immediate_win 覆盖；活四双赢点、一手成活四、一手成四三或双活三）
    miss_immediate_win_penalty: float = 2.8
    # 对方下一手成活四/四三/双活三后均近制胜，扣分与制胜手同档
    miss_open_three_penalty: float = 2.2
    miss_jump_open_three_penalty: float = 2.0
    miss_one_move_four_three_penalty: float = 2.2
    miss_one_move_double_three_penalty: float = 2.0
    # ---------- 反击先手折减 ----------
    # 本手形成活三/跳活三时，对「未阻止对方一手成双活三/四三」的惩罚折减系数。
    # 冲四是绝对先手（唯一挡点，对方必应且无法反击），故 my_strong_attack 全额豁免；
    # 活三只是相对先手（对方可用一手兼具封堵与反击来化解），故只部分豁免。
    # 取 1.0 等于关闭折减，取 0.0 等于与冲四同级全额豁免。
    counter_threat_waiver_scale: float = 0.5
    # ---------- 强攻豁免时的进攻分折减 ----------
    # 冲四豁免漏防在棋理上成立（绝对先手），但若同时全额保留冲四的进攻分，
    # 「远处随手冲四 + 全额豁免」净收益为正，会变成刷分燃料：模型可以攒一堆无用冲四
    # 轮流点，每手净赚，把真正的防守决策无限延后。故豁免生效时折减冲四的进攻分。
    # 只作用于冲四：活四/双四/四三/双活三本身近乎制胜，不折减。
    # 取 1.0 等于关闭折减。
    rush_four_waiver_attack_scale: float = 0.3


@dataclass(slots=True)
class ModelConfig:
    # 策略/价值网络：卷积通道数；残差块数量。
    # 9x9 只有 81 个格点，256x16（约 1900 万参数，每格 23 万）远超需要，且自博弈是
    # 逐步前向，网络越大每步越慢、迭代越慢。64x6 约 100 万参数，容量对该棋盘充裕。
    # 改动这两个值会使既有 checkpoint 因结构不符而无法加载，需要重跑预训练。
    channels: int = 64
    blocks: int = 6


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
    # 同时并行推进的对局数。各局在同一时刻待决策的局面会凑成一次网络前向，
    # 取代 batch=1 的逐局串行（实测每局面成本相差一到两个数量级）。
    # 调大更快但显存占用更高；设为 1 即退回串行。
    self_play_batch_games: int = 64
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
    heuristic_opponent_max_prob: float = 0.70
    heuristic_start_fraction: float = 0.0
    heuristic_ramp_fraction: float = 0.08
    # 评估与 checkpoint：每轮评估局数；启发式对手评估时的采样温度（>0 时增加对局多样性，避免胜率只有 0/0.5/1）
    eval_games: int = 48
    eval_heuristic_temperature: float = 0.15
    checkpoint_every: int = 20
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
    poll_interval_ms: int = 2000


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
    # 必须与 ModelConfig 保持一致：产出的 checkpoint 要能直接被 five-train 加载续训。
    channels: int = 64
    blocks: int = 6
    # 行为克隆收敛极快：实测第 1 轮 41.2%、第 2 轮 72.4%，之后 28 轮只涨到 76.1%
    # （+3.7 个点），而余弦调度把大半预算花在了这段几乎无收益的尾巴上。模仿的上限
    # 本就是启发式老师本身，多训无益，省下的时间应该给 PPO。
    epochs: int = 6
    batch_size: int = 1024
    lr: float = 1e-3
    value_coef: float = 0.5
    device: str = "cuda"
    output_dir: str = "pretrain_output"
    seed: int = 42