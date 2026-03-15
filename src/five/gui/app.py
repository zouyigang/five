from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from five.common.config import GUIConfig
from five.gui.controllers import RunController
from five.gui.pages.generate_page import GeneratePage
from five.gui.pages.pretrain_page import PretrainPage
from five.gui.pages.replay_page import ReplayPage
from five.gui.pages.train_page import TrainMonitorPage
from five.gui.pages.versus_ai_page import VersusAIPage
from five.gui.pages.reward_test_page import RewardTestPage


class FiveApp(tk.Tk):
    def __init__(self, runs_dir: Path = Path("runs"), config: GUIConfig | None = None) -> None:
        super().__init__()
        self.config_obj = config or GUIConfig()
        self.title("五子棋强化学习系统")
        self.geometry(f"{self.config_obj.window_width}x{self.config_obj.window_height}")
        self._style_notebook_tabs()
        controller = RunController(runs_dir)

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)
        notebook.add(GeneratePage(notebook), text="数据生成")
        notebook.add(PretrainPage(notebook), text="行为克隆预训练")
        notebook.add(
            TrainMonitorPage(notebook, controller, poll_interval_ms=self.config_obj.poll_interval_ms),
            text="PPO 微调",
        )
        notebook.add(ReplayPage(notebook, controller), text="对局回放")
        notebook.add(VersusAIPage(notebook, controller), text="人机对弈")
        notebook.add(RewardTestPage(notebook), text="奖励检验")

    def _style_notebook_tabs(self) -> None:
        """让 6 个标签页更易区分：选中态更醒目，未选中略弱化。"""
        style = ttk.Style(self)
        style.configure(
            "TNotebook.Tab",
            padding=(16, 10),
            font=("", 10),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#ffffff"), ("!selected", "#c0c0c0")],
            expand=[("selected", [1, 1, 1, 0])],
        )
        try:
            style.configure("TNotebook", tabmargins=[2, 4, 2, 0])
        except tk.TclError:
            pass


def main() -> None:
    app = FiveApp()
    app.mainloop()


if __name__ == "__main__":
    main()
