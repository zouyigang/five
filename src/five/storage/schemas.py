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
    avg_game_length: float
    eval_win_rate_random: float
    eval_win_rate_heuristic: float


@dataclass(slots=True)
class ModelRecord:
    checkpoint_name: str
    checkpoint_path: str
    epoch: int
    eval_win_rate_random: float
    eval_win_rate_heuristic: float
