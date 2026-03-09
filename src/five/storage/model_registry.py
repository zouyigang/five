from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from five.common.utils import ensure_dir, read_json, write_json
from five.storage.schemas import ModelRecord


class ModelRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_dir(path.parent)
        if not self.path.exists():
            write_json(self.path, {"models": []})

    def add(self, record: ModelRecord) -> None:
        payload = read_json(self.path)
        payload["models"].append(asdict(record))
        write_json(self.path, payload)

    def list_models(self) -> list[ModelRecord]:
        payload = read_json(self.path)
        return [ModelRecord(**item) for item in payload.get("models", [])]
