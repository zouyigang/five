from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd


class MetricsPanel(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.axes = self.figure.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_metrics(self, frame: pd.DataFrame) -> None:
        for axis in self.axes.flat:
            axis.clear()
        if frame.empty:
            self.canvas.draw_idle()
            return
        x = frame["epoch"]
        self.axes[0, 0].plot(x, frame["policy_loss"], label="policy")
        self.axes[0, 0].plot(x, frame["value_loss"], label="value")
        self.axes[0, 0].set_title("Loss")
        self.axes[0, 0].legend()

        self.axes[0, 1].plot(x, frame["entropy"])
        self.axes[0, 1].set_title("Entropy")

        self.axes[1, 0].plot(x, frame["avg_game_length"])
        self.axes[1, 0].set_title("Avg Game Length")

        self.axes[1, 1].plot(x, frame["eval_win_rate_random"], label="random")
        self.axes[1, 1].plot(x, frame["eval_win_rate_heuristic"], label="heuristic")
        self.axes[1, 1].set_title("Eval Win Rate")
        self.axes[1, 1].legend()
        self.figure.tight_layout()
        self.canvas.draw_idle()
