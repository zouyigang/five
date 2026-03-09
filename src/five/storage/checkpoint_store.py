from __future__ import annotations

from pathlib import Path

import torch

from five.common.utils import ensure_dir


class CheckpointStore:
    def __init__(self, directory: Path) -> None:
        self.directory = ensure_dir(directory)

    def save(self, name: str, payload: dict) -> Path:
        path = self.directory / name
        torch.save(payload, path)
        return path

    def latest(self) -> Path | None:
        candidates = sorted(self.directory.glob("*.pt"))
        if not candidates:
            return None
        return candidates[-1]
