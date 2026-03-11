from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd


class MetricsPanel(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.figure = Figure(figsize=(9, 10), dpi=100)
        self.axes = self.figure.subplots(4, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _rolling_window(self, frame: pd.DataFrame) -> int:
        return max(5, min(25, len(frame) // 20 if len(frame) >= 20 else len(frame)))

    def _series(self, frame: pd.DataFrame, column: str) -> pd.Series | None:
        if column not in frame.columns:
            return None
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().sum() == 0:
            return None
        return series

    def _best_epoch(self, frame: pd.DataFrame) -> int | None:
        epoch = self._series(frame, "epoch")
        heuristic = self._series(frame, "eval_win_rate_heuristic")
        random = self._series(frame, "eval_win_rate_random")
        value_loss = self._series(frame, "value_loss")
        if epoch is None or heuristic is None or random is None or value_loss is None:
            return None

        ranked = pd.DataFrame(
            {
                "epoch": epoch,
                "eval_win_rate_heuristic": heuristic.fillna(-1.0),
                "eval_win_rate_random": random.fillna(-1.0),
                "value_loss": value_loss.fillna(float("inf")),
            }
        )
        ranked = ranked.sort_values(
            by=["eval_win_rate_heuristic", "eval_win_rate_random", "value_loss", "epoch"],
            ascending=[False, False, True, False],
        )
        if ranked.empty:
            return None
        return int(ranked.iloc[0]["epoch"])

    def _anomaly_epochs(self, frame: pd.DataFrame) -> list[int]:
        epoch = self._series(frame, "epoch")
        if epoch is None:
            return []

        scored: dict[int, int] = {}
        window = self._rolling_window(frame)
        specs = [
            ("value_loss", 2.5),
            ("grad_norm", 2.5),
            ("return_abs_max", 2.0),
            ("avg_game_length", 1.8),
        ]
        for column, multiplier in specs:
            series = self._series(frame, column)
            if series is None:
                continue
            baseline = series.rolling(window=window, min_periods=max(3, window // 2)).median()
            for idx, value in enumerate(series):
                base = baseline.iloc[idx]
                if pd.isna(value) or pd.isna(base) or base <= 0:
                    continue
                if value > base * multiplier:
                    epoch_value = int(epoch.iloc[idx])
                    scored[epoch_value] = scored.get(epoch_value, 0) + 1

        heuristic = self._series(frame, "eval_win_rate_heuristic")
        if heuristic is not None:
            baseline = heuristic.rolling(window=window, min_periods=max(3, window // 2)).median()
            for idx, value in enumerate(heuristic):
                base = baseline.iloc[idx]
                if pd.isna(value) or pd.isna(base):
                    continue
                if base >= 0.5 and value <= max(0.0, base - 0.4):
                    epoch_value = int(epoch.iloc[idx])
                    scored[epoch_value] = scored.get(epoch_value, 0) + 1

        ranked_epochs = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
        return [epoch_value for epoch_value, _ in ranked_epochs[:8]]

    def _add_epoch_markers(self, axis, best_epoch: int | None, anomaly_epochs: list[int]) -> None:
        if best_epoch is not None:
            axis.axvline(best_epoch, color="tab:green", linestyle="--", linewidth=1.4, alpha=0.9)
        for epoch in anomaly_epochs:
            axis.axvline(epoch, color="tab:red", linestyle=":", linewidth=1.0, alpha=0.45)

    def _plot_metric(
        self,
        axis,
        x: pd.Series,
        series: pd.Series | None,
        title: str,
        label: str | None = None,
        color: str | None = None,
        *,
        ylim: tuple[float, float] | None = None,
        log_scale: bool = False,
        best_epoch: int | None = None,
        anomaly_epochs: list[int] | None = None,
    ) -> None:
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        if series is None:
            axis.text(0.5, 0.5, "N/A", ha="center", va="center", transform=axis.transAxes)
            return

        window = self._rolling_window(series.to_frame())
        smooth = series.rolling(window=window, min_periods=1).mean()
        axis.plot(x, series, color=color, alpha=0.22, linewidth=1.0)
        axis.plot(x, smooth, color=color, linewidth=2.0, label=label)
        if ylim is not None:
            axis.set_ylim(*ylim)
        if log_scale and (series.dropna() > 0).all():
            axis.set_yscale("log")
        self._add_epoch_markers(axis, best_epoch, anomaly_epochs or [])
        if label is not None:
            axis.legend()

    def _plot_multi_metric(
        self,
        axis,
        x: pd.Series,
        frame: pd.DataFrame,
        specs: list[tuple[str, str, str]],
        title: str,
        *,
        ylim: tuple[float, float] | None = None,
        best_epoch: int | None = None,
        anomaly_epochs: list[int] | None = None,
    ) -> None:
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        has_data = False
        window = self._rolling_window(frame)
        for column, label, color in specs:
            series = self._series(frame, column)
            if series is None:
                continue
            smooth = series.rolling(window=window, min_periods=1).mean()
            axis.plot(x, series, color=color, alpha=0.2, linewidth=1.0)
            axis.plot(x, smooth, color=color, linewidth=2.0, label=label)
            has_data = True
        if not has_data:
            axis.text(0.5, 0.5, "N/A", ha="center", va="center", transform=axis.transAxes)
            return
        if ylim is not None:
            axis.set_ylim(*ylim)
        self._add_epoch_markers(axis, best_epoch, anomaly_epochs or [])
        axis.legend()

    def update_metrics(self, frame: pd.DataFrame) -> None:
        for axis in self.axes.flat:
            axis.clear()
        if frame.empty:
            self.canvas.draw_idle()
            return

        x = frame["epoch"]
        best_epoch = self._best_epoch(frame)
        anomaly_epochs = self._anomaly_epochs(frame)
        self._plot_metric(
            self.axes[0, 0],
            x,
            self._series(frame, "policy_loss"),
            "Policy Loss",
            label="policy",
            color="tab:blue",
            best_epoch=best_epoch,
            anomaly_epochs=anomaly_epochs,
        )
        self._plot_metric(
            self.axes[0, 1],
            x,
            self._series(frame, "value_loss"),
            "Value Loss",
            label="value",
            color="tab:orange",
            log_scale=True,
            best_epoch=best_epoch,
            anomaly_epochs=anomaly_epochs,
        )
        self._plot_metric(
            self.axes[1, 0],
            x,
            self._series(frame, "entropy"),
            "Entropy",
            label="entropy",
            color="tab:green",
            best_epoch=best_epoch,
            anomaly_epochs=anomaly_epochs,
        )
        self._plot_metric(
            self.axes[1, 1],
            x,
            self._series(frame, "avg_game_length"),
            "Avg Game Length",
            label="avg length",
            color="tab:red",
            best_epoch=best_epoch,
            anomaly_epochs=anomaly_epochs,
        )
        self._plot_multi_metric(
            self.axes[2, 0],
            x,
            frame,
            [
                ("eval_win_rate_random", "random", "tab:blue"),
                ("eval_win_rate_heuristic", "heuristic", "tab:orange"),
            ],
            "Eval Win Rate",
            ylim=(0.0, 1.0),
            best_epoch=best_epoch,
            anomaly_epochs=anomaly_epochs,
        )
        self._plot_metric(
            self.axes[2, 1],
            x,
            self._series(frame, "grad_norm"),
            "Grad Norm",
            label="grad norm",
            color="tab:purple",
            log_scale=True,
            best_epoch=best_epoch,
            anomaly_epochs=anomaly_epochs,
        )
        self._plot_multi_metric(
            self.axes[3, 0],
            x,
            frame,
            [
                ("return_mean", "return mean", "tab:blue"),
                ("return_std", "return std", "tab:orange"),
            ],
            "Return Mean / Std",
            best_epoch=best_epoch,
            anomaly_epochs=anomaly_epochs,
        )
        self._plot_metric(
            self.axes[3, 1],
            x,
            self._series(frame, "return_abs_max"),
            "Return Abs Max",
            label="return abs max",
            color="tab:brown",
            log_scale=True,
            best_epoch=best_epoch,
            anomaly_epochs=anomaly_epochs,
        )
        if best_epoch is not None or anomaly_epochs:
            status_parts = []
            if best_epoch is not None:
                status_parts.append(f"best epoch={best_epoch}")
            if anomaly_epochs:
                status_parts.append("anomaly epochs=" + ", ".join(str(epoch) for epoch in anomaly_epochs[:5]))
            self.figure.suptitle(" | ".join(status_parts), fontsize=10)
            self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
        else:
            self.figure.suptitle("")
            self.figure.tight_layout()
        self.canvas.draw_idle()
