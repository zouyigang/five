from __future__ import annotations

from pathlib import Path

from five.common.utils import ensure_dir, read_json, write_json
from five.storage.schemas import GameRecord, MoveRecord, MoveSummary, RewardDetail


class GameStore:
    def __init__(self, games_dir: Path) -> None:
        self.games_dir = ensure_dir(games_dir)

    def save(self, record: GameRecord) -> Path:
        path = self.games_dir / f"{record.game_id}.json"
        write_json(path, record.to_dict())
        return path

    def list_game_paths(self) -> list[Path]:
        return sorted(self.games_dir.glob("*.json"))

    def load(self, path: Path) -> GameRecord:
        data = read_json(path)
        moves = [
            MoveRecord(
                move_index=item["move_index"],
                player=item["player"],
                row=item["row"],
                col=item["col"],
                action_probability=item["action_probability"],
                value_before=item["value_before"],
                legal_count=item["legal_count"],
                total_reward=item.get("total_reward", 0.0),
                reward_details=[
                    RewardDetail(
                        amount=detail["amount"],
                        reason=detail["reason"],
                    )
                    for detail in item.get("reward_details", [])
                ],
                policy_topk=[
                    MoveSummary(
                        row=summary["row"],
                        col=summary["col"],
                        score=summary["score"],
                        visits=summary.get("visits"),
                        value=summary.get("value"),
                    )
                    for summary in item.get("policy_topk", [])
                ],
            )
            for item in data["moves"]
        ]
        return GameRecord(
            game_id=data["game_id"],
            run_id=data["run_id"],
            board_size=data["board_size"],
            win_length=data["win_length"],
            winner=data["winner"],
            total_moves=data["total_moves"],
            black_player=data["black_player"],
            white_player=data["white_player"],
            result=data["result"],
            model_checkpoint=data.get("model_checkpoint"),
            moves=moves,
        )
