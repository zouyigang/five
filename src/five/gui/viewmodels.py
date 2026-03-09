from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunOption:
    name: str
    path: Path


@dataclass(slots=True)
class GameOption:
    name: str
    path: Path
