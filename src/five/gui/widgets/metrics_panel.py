from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
import pandas as pd

from five.train.best_epoch import compute_best_epoch


class MetricsPanel(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.figure = Figure(figsize=(10, 12), dpi=100, constrained_layout=True)
        self.axes = self.figure.subplots(5, 2)
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
        return compute_best_epoch(frame)

    def _anomaly_epochs(self, frame: pd.DataFrame) -> list[int]:
        epoch = self._series(frame, "epoch")
        if epoch is None:
            return []

        scored: dict[int, int] = {}
        window = self._rolling_window(frame)
        min_periods = max(3, window // 2)

        def _mark(idx: int, weight: int = 1) -> None:
            epoch_value = int(epoch.iloc[idx])
            scored[epoch_value] = scored.get(epoch_value, 0) + weight

        spike_specs = [
            ("value_loss", 2.5),
            ("grad_norm", 2.5),
            ("policy_loss", 2.5),
        ]
        for column, multiplier in spike_specs:
            series = self._series(frame, column)
            if series is None:
                continue
            abs_series = series.abs() if column == "policy_loss" else series
            baseline = abs_series.rolling(window=window, min_periods=min_periods).median()
            for idx in range(len(abs_series)):
                value = abs_series.iloc[idx]
                base = baseline.iloc[idx]
                if pd.isna(value) or pd.isna(base) or base <= 1e-9:
                    continue
                if value > base * multiplier:
                    _mark(idx)

        avg_length = self._series(frame, "avg_game_length")
        if avg_length is not None:
            baseline = avg_length.rolling(window=window, min_periods=min_periods).median()
            for idx in range(len(avg_length)):
                value = avg_length.iloc[idx]
                base = baseline.iloc[idx]
                if pd.isna(value) or pd.isna(base) or base <= 0:
                    continue
                if value > base * 1.8:
                    _mark(idx)
                if value < base * 0.4:
                    _mark(idx)

        entropy = self._series(frame, "entropy")
        if entropy is not None:
            baseline = entropy.rolling(window=window, min_periods=min_periods).median()
            for idx in range(len(entropy)):
                value = entropy.iloc[idx]
                base = baseline.iloc[idx]
                if pd.isna(value) or pd.isna(base) or base <= 1e-9:
                    continue
                if value < base * 0.4:
                    _mark(idx, weight=2)

        for wr_col, drop_thresh in [("eval_win_rate_heuristic", 0.35), ("eval_win_rate_random", 0.25)]:
            wr = self._series(frame, wr_col)
            if wr is None:
                continue
            baseline = wr.rolling(window=window, min_periods=min_periods).median()
            for idx in range(len(wr)):
                value = wr.iloc[idx]
                base = baseline.iloc[idx]
                if pd.isna(value) or pd.isna(base):
                    continue
                if base >= 0.3 and value <= max(0.0, base - drop_thresh):
                    _mark(idx, weight=2)

        trend_window = max(window * 2, 20)
        for column, direction in [("entropy", "down"), ("eval_win_rate_random", "down"), ("value_loss", "up")]:
            series = self._series(frame, column)
            if series is None or len(series) < trend_window:
                continue
            for idx in range(trend_window, len(series)):
                segment = series.iloc[idx - trend_window : idx]
                if segment.isna().sum() > trend_window // 3:
                    continue
                start_val = segment.iloc[: trend_window // 4].mean()
                end_val = segment.iloc[-(trend_window // 4) :].mean()
                if pd.isna(start_val) or pd.isna(end_val) or abs(start_val) < 1e-9:
                    continue
                change_ratio = (end_val - start_val) / abs(start_val)
                if direction == "down" and change_ratio < -0.5:
                    _mark(idx)
                elif direction == "up" and change_ratio > 1.0:
                    _mark(idx)

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
        axis.plot(x, series, color=color, alpha=0.22, linewidth=1.0, marker="o", markersize=3)
        axis.plot(x, smooth, color=color, linewidth=2.0, label=label, marker="o", markersize=3)
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
            axis.plot(x, series, color=color, alpha=0.2, linewidth=1.0, marker="o", markersize=3)
            axis.plot(x, smooth, color=color, linewidth=2.0, label=label, marker="o", markersize=3)
            has_data = True
        if not has_data:
            axis.text(0.5, 0.5, "N/A", ha="center", va="center", transform=axis.transAxes)
            return
        if ylim is not None:
            axis.set_ylim(*ylim)
        self._add_epoch_markers(axis, best_epoch, anomaly_epochs or [])
        axis.legend()

    def _set_epoch_axis(self, x_min: float, x_max: float) -> None:
        """横轴为整数轮次，刻度数量随范围动态调整，避免轮次多时过密卡顿。"""
        span = max(1.0, x_max - x_min + 1)
        # 目标约 8–12 个刻度，轮次少时步长 1
        step = max(1, int(span / 10))
        for ax in self.axes.flat:
            ax.set_xlim(x_min - 0.5, x_max + 0.5)
            ax.xaxis.set_major_locator(MultipleLocator(step))

    def update_metrics(self, frame: pd.DataFrame) -> None:
        for axis in self.axes.flat:
            axis.clear()
        if frame.empty:
            self._set_epoch_axis(0.0, 10.0)
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
        self._plot_multi_metric(
            self.axes[4, 0],
            x,
            frame,
            [
                ("opening_edge_rate", "opening edge", "tab:red"),
                ("opening_corner_rate", "opening corner", "tab:purple"),
                ("opening_center_rate", "opening center", "tab:green"),
            ],
            "Opening Position Rates",
            ylim=(0.0, 1.0),
            best_epoch=best_epoch,
            anomaly_epochs=anomaly_epochs,
        )
        self._plot_metric(
            self.axes[4, 1],
            x,
            self._series(frame, "policy_topk_edge_rate"),
            "Opening Top-K Edge Rate",
            label="top-k edge",
            color="tab:cyan",
            ylim=(0.0, 1.0),
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
        else:
            self.figure.suptitle("")
        x_min, x_max = float(x.min()), float(x.max())
        self._set_epoch_axis(x_min, x_max)
        self.canvas.draw()
