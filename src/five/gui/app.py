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
        self._pages: list[ttk.Frame] = []
        self.title("五子棋强化学习系统")
        self.geometry(f"{self.config_obj.window_width}x{self.config_obj.window_height}")
        self._style_notebook_tabs()
        controller = RunController(runs_dir)

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook = notebook

        # 先创建各个页面实例，方便根据 Tab 激活状态控制轮询
        self.generate_page = GeneratePage(notebook)
        self.pretrain_page = PretrainPage(notebook)
        self.train_page = TrainMonitorPage(
            notebook, controller, poll_interval_ms=self.config_obj.poll_interval_ms
        )
        self.replay_page = ReplayPage(notebook, controller)
        self.versus_page = VersusAIPage(notebook, controller)
        self.reward_test_page = RewardTestPage(notebook)

        notebook.add(self.generate_page, text="数据生成")
        notebook.add(self.pretrain_page, text="行为克隆预训练")
        notebook.add(self.train_page, text="PPO 微调")
        notebook.add(self.replay_page, text="对局回放")
        notebook.add(self.versus_page, text="人机对弈")
        notebook.add(self.reward_test_page, text="奖励检验")

        self._pages = [
            self.generate_page,
            self.pretrain_page,
            self.train_page,
            self.replay_page,
            self.versus_page,
            self.reward_test_page,
        ]

        # 绑定 Tab 切换事件：只有当前可见页面才执行自身的轮询逻辑
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        # 初始化一次激活状态（Tkinter 默认选中第一个 Tab）
        self.after(0, self._update_page_active_states)

    def _on_tab_changed(self, _event) -> None:
        self._update_page_active_states()

    def _update_page_active_states(self) -> None:
        """根据当前选中的 Tab，通知各页是否处于激活状态。"""
        current = self.notebook.select()
        for page in self._pages:
            active = str(page) == str(current)
            # 仅当页面实现 set_active 时才调用
            setter = getattr(page, "set_active", None)
            if callable(setter):
                setter(active)

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
