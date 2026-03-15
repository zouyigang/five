"""GUI 页：从 five-pretrain 进度文件实时刷新预训练指标与曲线。"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

# 使用支持中文的字体，避免 "Glyph missing from font(s) DejaVu Sans" 警告
matplotlib.rcParams.setdefault("font.sans-serif", ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"])
matplotlib.rcParams["axes.unicode_minus"] = False

from five.common.config import PretrainConfig


def _default_progress_file() -> str:
    return str(Path(PretrainConfig().output_dir) / "pretrain.progress.json")


def _scan_pretrain_progress_files() -> list[str]:
    """扫描当前目录及常见输出目录下的 pretrain.progress.json，统一用相对路径并去重。"""
    default = _default_progress_file()
    cwd = Path.cwd()
    seen: set[Path] = set()
    candidates: list[str] = []

    def add(path: Path) -> None:
        r = path.resolve()
        if r in seen:
            return
        seen.add(r)
        try:
            candidates.append(str(r.relative_to(cwd)))
        except ValueError:
            candidates.append(str(r))

    for d in (".", "pretrain_output", Path(default).parent):
        p = Path(d).resolve()
        if not p.exists():
            continue
        f = p / "pretrain.progress.json"
        if f.exists():
            add(f)
    for p in Path(".").resolve().rglob("pretrain.progress.json"):
        if p.is_file():
            add(p)
    candidates.sort()
    if Path(default).resolve() not in seen and default not in candidates:
        candidates.insert(0, default)
    return candidates


class PretrainPage(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master)
        self._progress_path = tk.StringVar(value=_default_progress_file())
        self._history: dict[str, list[float]] = {
            "epoch": [],
            "policy_loss": [],
            "value_loss": [],
            "accuracy": [],
            "lr": [],
        }
        self._last_mtime: float = 0
        self._status_var = tk.StringVar(value="未检测到预训练任务（请用 five-pretrain 后台运行并指定输出目录）")

        # 进度文件（与训练监控一致：下拉框 + 刷新）
        path_frame = ttk.Frame(self)
        path_frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(path_frame, text="进度文件").pack(side=tk.LEFT, padx=4)
        self._progress_combo = ttk.Combobox(
            path_frame, textvariable=self._progress_path, state="readonly", width=50
        )
        self._progress_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(path_frame, text="刷新", command=self._refresh_progress_files).pack(side=tk.LEFT)
        self._refresh_progress_files()

        # 当前指标
        metrics_frame = ttk.LabelFrame(self, text="当前指标")
        metrics_frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(metrics_frame, textvariable=self._status_var).pack(fill=tk.X, padx=4, pady=2)
        row = ttk.Frame(metrics_frame)
        row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(row, text="轮次:").pack(side=tk.LEFT, padx=4)
        self.epoch_var = tk.StringVar(value="-")
        ttk.Label(row, textvariable=self.epoch_var).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="策略损失:").pack(side=tk.LEFT, padx=4)
        self.policy_loss_var = tk.StringVar(value="-")
        ttk.Label(row, textvariable=self.policy_loss_var).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="价值损失:").pack(side=tk.LEFT, padx=4)
        self.value_loss_var = tk.StringVar(value="-")
        ttk.Label(row, textvariable=self.value_loss_var).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="准确率(%):").pack(side=tk.LEFT, padx=4)
        self.accuracy_var = tk.StringVar(value="-")
        ttk.Label(row, textvariable=self.accuracy_var).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="学习率:").pack(side=tk.LEFT, padx=4)
        self.lr_display_var = tk.StringVar(value="-")
        ttk.Label(row, textvariable=self.lr_display_var).pack(side=tk.LEFT, padx=4)

        # 曲线
        fig_frame = ttk.LabelFrame(self, text="训练曲线")
        fig_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.figure = Figure(figsize=(8, 4), dpi=90, constrained_layout=True)
        self.ax_loss = self.figure.add_subplot(1, 2, 1)
        self.ax_acc = self.figure.add_subplot(1, 2, 2)
        self.ax_loss.set_title("Policy / Value Loss")
        self.ax_acc.set_title("Accuracy %")
        self.canvas = FigureCanvasTkAgg(self.figure, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.after(800, self._poll_progress_file)

    def _refresh_progress_files(self) -> None:
        values = _scan_pretrain_progress_files()
        self._progress_combo["values"] = values
        current = self._progress_path.get().strip()
        if values and (not current or current not in values):
            self._progress_path.set(values[0])

    def _poll_progress_file(self) -> None:
        path = self._progress_path.get().strip()
        if path:
            try:
                p = Path(path)
                if p.exists():
                    mtime = p.stat().st_mtime
                    if mtime != self._last_mtime:
                        self._last_mtime = mtime
                        raw = p.read_text(encoding="utf-8")
                        data = json.loads(raw)
                        self._apply_progress(data)
            except (OSError, json.JSONDecodeError):
                pass
        self.after(800, self._poll_progress_file)

    def _apply_progress(self, data: dict) -> None:
        running = data.get("running", False)
        total = int(data.get("total_epochs", 0))
        history = data.get("history", [])

        out = data.get("output_dir", "")
        status = "预训练中…" if running else "已结束"
        self._status_var.set(f"{status}  →  {out}" if out else status)

        if history:
            self._history["epoch"] = [float(h["epoch"]) for h in history]
            self._history["policy_loss"] = [h["policy_loss"] for h in history]
            self._history["value_loss"] = [h["value_loss"] for h in history]
            self._history["accuracy"] = [h["accuracy"] for h in history]
            self._history["lr"] = [h["lr"] for h in history]
            last = history[-1]
            self.epoch_var.set(f"{int(last['epoch'])} / {total}")
            self.policy_loss_var.set(f"{last['policy_loss']:.4f}")
            self.value_loss_var.set(f"{last['value_loss']:.4f}")
            self.accuracy_var.set(f"{last['accuracy']:.2f}")
            self.lr_display_var.set(f"{last['lr']:.2e}")
        else:
            self.epoch_var.set("-")
            self.policy_loss_var.set("-")
            self.value_loss_var.set("-")
            self.accuracy_var.set("-")
            self.lr_display_var.set("-")

        self._redraw_plots()

    def _redraw_plots(self) -> None:
        self.ax_loss.clear()
        self.ax_acc.clear()
        self.ax_loss.set_title("Policy / Value Loss")
        self.ax_acc.set_title("Accuracy %")
        ep = self._history["epoch"]
        if ep:
            # 单点时 plot 无线段，加 marker 才能看到；多点时也保留 marker 便于读每个 epoch
            self.ax_loss.plot(
                ep, self._history["policy_loss"], label="policy_loss", marker="o", markersize=4
            )
            self.ax_loss.plot(
                ep, self._history["value_loss"], label="value_loss", marker="o", markersize=4
            )
            self.ax_loss.legend()
            self.ax_loss.set_xlabel("epoch")
            x_min, x_max = min(ep), max(ep)
            self.ax_loss.set_xlim(x_min - 0.5, x_max + 0.5)
            self.ax_loss.xaxis.set_major_locator(MultipleLocator(1))
            self.ax_acc.plot(ep, self._history["accuracy"], color="green", marker="o", markersize=4)
            self.ax_acc.set_xlabel("epoch")
            self.ax_acc.set_xlim(x_min - 0.5, x_max + 0.5)
            self.ax_acc.xaxis.set_major_locator(MultipleLocator(1))
        self.canvas.draw_idle()
