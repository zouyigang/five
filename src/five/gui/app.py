from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from five.common.config import GUIConfig
from five.gui.controllers import RunController
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
        controller = RunController(runs_dir)

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)
        notebook.add(
            TrainMonitorPage(notebook, controller, poll_interval_ms=self.config_obj.poll_interval_ms),
            text="训练监控",
        )
        notebook.add(ReplayPage(notebook, controller), text="对局回放")
        notebook.add(VersusAIPage(notebook, controller), text="人机对弈")
        notebook.add(RewardTestPage(notebook), text="奖励检验")


def main() -> None:
    app = FiveApp()
    app.mainloop()


if __name__ == "__main__":
    main()
