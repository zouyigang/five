from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from five.gui.controllers import RunController
from five.gui.widgets.metrics_panel import MetricsPanel


class TrainMonitorPage(ttk.Frame):
    def __init__(self, master, controller: RunController, poll_interval_ms: int = 500) -> None:
        super().__init__(master)
        self.controller = controller
        self.poll_interval_ms = poll_interval_ms
        self.selected_run = tk.StringVar()
        self._run_lookup: dict[str, Path] = {}

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(controls, text="训练运行").pack(side=tk.LEFT)
        self.run_box = ttk.Combobox(controls, textvariable=self.selected_run, state="readonly", width=40)
        self.run_box.pack(side=tk.LEFT, padx=8)
        ttk.Button(controls, text="刷新", command=self.refresh_runs).pack(side=tk.LEFT)

        self.metrics_panel = MetricsPanel(self)
        self.metrics_panel.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.run_box.bind("<<ComboboxSelected>>", lambda _: self.refresh_metrics())
        self.refresh_runs()
        self.after(self.poll_interval_ms, self._poll)

    def refresh_runs(self) -> None:
        runs = self.controller.list_runs()
        self._run_lookup = {run.name: run for run in runs}
        self.run_box["values"] = list(self._run_lookup.keys())
        if runs and not self.selected_run.get():
            self.selected_run.set(runs[-1].name)
        self.refresh_metrics()

    def refresh_metrics(self) -> None:
        run_name = self.selected_run.get()
        if not run_name:
            return
        run_path = self._run_lookup.get(run_name)
        if run_path is None:
            return
        frame = self.controller.metric_store(run_path).read_frame()
        self.metrics_panel.update_metrics(frame)

    def _poll(self) -> None:
        self.refresh_metrics()
        self.after(self.poll_interval_ms, self._poll)
