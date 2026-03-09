from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from five.core.move import Move
from five.core.replay import reconstruct_states
from five.gui.controllers import RunController
from five.gui.widgets.board_canvas import BoardCanvas
from five.gui.widgets.move_detail_panel import MoveDetailPanel


class ReplayPage(ttk.Frame):
    def __init__(self, master, controller: RunController) -> None:
        super().__init__(master)
        self.controller = controller
        self.selected_run = tk.StringVar()
        self.selected_game = tk.StringVar()
        self._run_lookup: dict[str, Path] = {}
        self._game_lookup: dict[str, Path] = {}
        self.frames = []
        self.current_index = 0
        self.current_record = None

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=8, pady=8)
        self.run_box = ttk.Combobox(controls, textvariable=self.selected_run, state="readonly", width=35)
        self.game_box = ttk.Combobox(controls, textvariable=self.selected_game, state="readonly", width=35)
        self.run_box.pack(side=tk.LEFT, padx=4)
        self.game_box.pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="刷新", command=self.refresh_runs).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="上一手", command=self.prev_move).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="下一手", command=self.next_move).pack(side=tk.LEFT, padx=4)

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)
        self.board = BoardCanvas(body)
        self.board.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.detail = MoveDetailPanel(body)
        self.detail.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=8, pady=8)

        self.run_box.bind("<<ComboboxSelected>>", lambda _: self.refresh_games())
        self.game_box.bind("<<ComboboxSelected>>", lambda _: self.load_game())
        self.refresh_runs()

    def refresh_runs(self) -> None:
        runs = self.controller.list_runs()
        self._run_lookup = {run.name: run for run in runs}
        self.run_box["values"] = list(self._run_lookup.keys())
        if runs and not self.selected_run.get():
            self.selected_run.set(runs[-1].name)
        self.refresh_games()

    def refresh_games(self) -> None:
        run_path = self._run_lookup.get(self.selected_run.get())
        if run_path is None:
            return
        games = self.controller.game_store(run_path).list_game_paths()
        self._game_lookup = {path.stem: path for path in games}
        self.game_box["values"] = list(self._game_lookup.keys())
        if games and not self.selected_game.get():
            self.selected_game.set(games[-1].stem)
        self.load_game()

    def load_game(self) -> None:
        run_path = self._run_lookup.get(self.selected_run.get())
        game_path = self._game_lookup.get(self.selected_game.get())
        if run_path is None or game_path is None:
            return
        record = self.controller.game_store(run_path).load(game_path)
        self.current_record = record
        moves = [Move(row=item.row, col=item.col) for item in record.moves]
        self.frames = reconstruct_states(moves, board_size=record.board_size, win_length=record.win_length)
        self.current_index = 0
        self.render_current()

    def render_current(self) -> None:
        if not self.frames:
            return
        frame = self.frames[self.current_index]
        highlights = None
        move_record = None
        if self.current_record and frame.ply > 0:
            move_record = self.current_record.moves[frame.ply - 1]
            highlights = [(item.row, item.col, item.score) for item in move_record.policy_topk]
        self.board.render(frame.state, highlights=highlights)
        self.detail.show_move(move_record)

    def prev_move(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self.render_current()

    def next_move(self) -> None:
        if self.current_index + 1 < len(self.frames):
            self.current_index += 1
            self.render_current()
