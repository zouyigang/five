from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from five.common.utils import ensure_dir
from five.storage.schemas import MetricRecord


class MetricStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_dir(path.parent)
        self.fieldnames = list(asdict(MetricRecord(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)).keys())
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()

    def append(self, record: MetricRecord) -> None:
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(asdict(record))

    def read_frame(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=self.fieldnames)
        return pd.read_csv(self.path)
