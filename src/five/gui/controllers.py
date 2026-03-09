from __future__ import annotations

from pathlib import Path

from five.storage.game_store import GameStore
from five.storage.metric_store import MetricStore
from five.storage.model_registry import ModelRegistry


class RunController:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir

    def list_runs(self) -> list[Path]:
        if not self.runs_dir.exists():
            return []
        return sorted([path for path in self.runs_dir.iterdir() if path.is_dir()])

    def game_store(self, run_path: Path) -> GameStore:
        return GameStore(run_path / "games")

    def metric_store(self, run_path: Path) -> MetricStore:
        return MetricStore(run_path / "metrics.csv")

    def model_registry(self, run_path: Path) -> ModelRegistry:
        return ModelRegistry(run_path / "models.json")
