from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from five.core.move import Move
from five.core.replay import reconstruct_states
from five.gui.bad_move_reasons import BAD_MOVE_REASONS
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
        self.current_run_path: Path | None = None
        self.mark_bad_mode = tk.BooleanVar(value=False)

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=8, pady=8)
        self.run_box = ttk.Combobox(controls, textvariable=self.selected_run, state="readonly", width=35)
        self.game_box = ttk.Combobox(controls, textvariable=self.selected_game, state="readonly", width=35)
        self.run_box.pack(side=tk.LEFT, padx=4)
        self.game_box.pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="刷新", command=self.refresh_runs).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="上一手", command=self.prev_move).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="下一手", command=self.next_move).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(controls, text="标记坏棋", variable=self.mark_bad_mode).pack(side=tk.LEFT, padx=4)

        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, padx=8, pady=2)
        self.bad_moves_var = tk.StringVar(value="坏棋：无")
        ttk.Label(status_frame, textvariable=self.bad_moves_var, foreground="#c0392b").pack(side=tk.LEFT)

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)
        self.board = BoardCanvas(body)
        self.board.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.board.set_click_handler(self._on_board_click)
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
        self.current_run_path = run_path
        record = self.controller.game_store(run_path).load(game_path)
        self.current_record = record
        moves = [Move(row=item.row, col=item.col) for item in record.moves]
        self.frames = reconstruct_states(moves, board_size=record.board_size, win_length=record.win_length)
        self.current_index = len(self.frames) - 1 if self.frames else 0
        self._update_bad_moves_label()
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
        bad_moves = None
        if self.current_record and frame.ply > 0:
            visible = [m for m in self.current_record.moves if m.move_index <= frame.ply and m.human_rating is not None]
            if visible:
                bad_moves = [(m.row, m.col, m.move_index) for m in visible]
        self.board.render(frame.state, highlights=highlights, bad_moves=bad_moves)
        self.detail.show_move(move_record, record=self.current_record)

    def _on_board_click(self, move: Move) -> None:
        if not self.mark_bad_mode.get() or not self.current_record or not self.frames:
            return
        frame = self.frames[self.current_index]
        if frame.ply == 0:
            return
        if self.current_record.board_size <= move.row or move.row < 0 or self.current_record.board_size <= move.col or move.col < 0:
            return
        if frame.state.board.grid[move.row, move.col] == 0:
            return
        # 人机对局：只允许标对方（AI）的棋；自对弈或未知：双方都可标
        if self.current_record.black_player == "human" and self.current_record.white_player != "human":
            ai_player: int | None = -1
        elif self.current_record.white_player == "human" and self.current_record.black_player != "human":
            ai_player = 1
        else:
            ai_player = None
        for rec in self.current_record.moves:
            if rec.row == move.row and rec.col == move.col and rec.move_index <= frame.ply:
                if ai_player is not None and rec.player != ai_player:
                    return
                if rec.human_rating is not None:
                    rec.human_rating = None
                    rec.human_bad_reasons = []
                    self._update_bad_moves_label()
                    self._save_record()
                    self.render_current()
                    return
                reasons = self._ask_bad_move_reasons(rec.move_index)
                if reasons is not None:
                    rec.human_rating = 0.0
                    rec.human_bad_reasons = reasons
                    self._update_bad_moves_label()
                    self._save_record()
                    self.render_current()
                return
        return

    def _ask_bad_move_reasons(self, move_index: int) -> list[str] | None:
        result_holder: list[list[str] | None] = [None]
        parent = self.winfo_toplevel()
        dialog = tk.Toplevel(parent)
        dialog.title(f"第 {move_index} 手标记为坏棋 — 选择原因")
        dialog.transient(parent)
        dialog.grab_set()
        vars_map = {r: tk.BooleanVar(value=False) for r in BAD_MOVE_REASONS}
        inner = ttk.Frame(dialog, padding=12)
        inner.pack(fill=tk.BOTH, expand=True)
        ttk.Label(inner, text="请勾选该手为坏棋的原因（可多选）：").pack(anchor=tk.W)
        for reason in BAD_MOVE_REASONS:
            ttk.Checkbutton(inner, text=reason, variable=vars_map[reason]).pack(anchor=tk.W, padx=8)
        btn_frame = ttk.Frame(inner)
        btn_frame.pack(fill=tk.X, pady=(12, 0))

        def on_ok() -> None:
            result_holder[0] = [r for r, v in vars_map.items() if v.get()]
            dialog.destroy()

        def on_cancel() -> None:
            result_holder[0] = None
            dialog.destroy()

        ttk.Button(btn_frame, text="确定", command=on_ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="取消", command=on_cancel).pack(side=tk.LEFT)
        dialog.protocol("WM_DELETE_WINDOW", on_cancel)
        dialog.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_x()
        py = parent.winfo_y()
        dw = dialog.winfo_reqwidth()
        dh = dialog.winfo_reqheight()
        x = px + max(0, (pw - dw) // 2)
        y = py + max(0, (ph - dh) // 2)
        dialog.geometry(f"+{x}+{y}")
        dialog.wait_window()
        return result_holder[0]

    def _update_bad_moves_label(self) -> None:
        if not self.current_record:
            self.bad_moves_var.set("坏棋：无")
            return
        marked = [m.move_index for m in self.current_record.moves if m.human_rating is not None]
        if marked:
            self.bad_moves_var.set("坏棋：第 " + "、".join(str(i) for i in sorted(marked)) + " 手")
        else:
            self.bad_moves_var.set("坏棋：无")

    def _save_record(self) -> None:
        """将当前对局（含坏棋标记）写入 run 的 game_store，标记/撤销时自动调用。"""
        if self.current_run_path is None or self.current_record is None:
            return
        self.controller.game_store(self.current_run_path).save(self.current_record)

    def prev_move(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self.render_current()

    def next_move(self) -> None:
        if self.current_index + 1 < len(self.frames):
            self.current_index += 1
            self.render_current()
