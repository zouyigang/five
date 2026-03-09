from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from five.common.config import TrainingConfig
from five.common.utils import ensure_dir, timestamp, write_json
from five.storage.checkpoint_store import CheckpointStore
from five.storage.game_store import GameStore
from five.storage.metric_store import MetricStore
from five.storage.model_registry import ModelRegistry


@dataclass(slots=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    game_store: GameStore
    metric_store: MetricStore
    checkpoint_store: CheckpointStore
    model_registry: ModelRegistry


def create_run(config: TrainingConfig) -> RunArtifacts:
    run_id = f"{config.run_name}_{timestamp()}"
    run_dir = ensure_dir(config.runs_path / run_id)
    games_dir = ensure_dir(run_dir / "games")
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    write_json(run_dir / "config.json", config.to_dict())
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        game_store=GameStore(games_dir),
        metric_store=MetricStore(run_dir / "metrics.csv"),
        checkpoint_store=CheckpointStore(checkpoints_dir),
        model_registry=ModelRegistry(run_dir / "models.json"),
    )
