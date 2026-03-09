from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TrainEvent:
    event_type: str
    payload: dict[str, Any]
