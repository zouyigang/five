from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class MoveSummary:
    row: int
    col: int
    score: float
    visits: int | None = None
    value: float | None = None


@dataclass(slots=True)
class RewardDetail:
    amount: float
    reason: str


@dataclass(slots=True)
class MoveRecord:
    move_index: int
    player: int
    row: int
    col: int
    action_probability: float
    value_before: float
    legal_count: int
    total_reward: float = 0.0
    reward_details: list[RewardDetail] = field(default_factory=list)
    policy_topk: list[MoveSummary] = field(default_factory=list)
    # 人类评分：None=未评，0.0=标记为坏棋，后续可扩展 0~1
    human_rating: float | None = None
    # 标记为坏棋时选择的原因（来自奖励扣分项），可多选
    human_bad_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GameRecord:
    game_id: str
    run_id: str
    board_size: int
    win_length: int
    winner: int
    total_moves: int
    black_player: str
    white_player: str
    result: str
    model_checkpoint: str | None
    moves: list[MoveRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class MetricRecord:
    epoch: int
    games: int
    policy_loss: float
    value_loss: float
    entropy: float
    grad_norm: float
    return_mean: float
    return_std: float
    return_abs_max: float
    avg_game_length: float
    eval_win_rate_random: float
    eval_win_rate_heuristic: float
    opening_edge_rate: float = 0.0
    opening_corner_rate: float = 0.0
    opening_center_rate: float = 0.0
    policy_topk_edge_rate: float = 0.0


@dataclass(slots=True)
class ModelRecord:
    checkpoint_name: str
    checkpoint_path: str
    epoch: int
    eval_win_rate_random: float
    eval_win_rate_heuristic: float
