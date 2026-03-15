"""GUI 页：从 five-generate 进度文件实时刷新生成进度，分页查看对局详情。"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from five.common.config import GenerateConfig
from five.core.move import Move
from five.core.replay import reconstruct_states
from five.gui.widgets.board_canvas import BoardCanvas


def _default_progress_file() -> str:
    return str(Path(GenerateConfig().output).with_suffix(".progress.json"))


def _games_detail_file_from_progress(progress_path: str) -> str:
    p = Path(progress_path)
    return str(p.parent / (p.stem.replace(".progress", "") + ".games.jsonl"))


def _scan_generate_progress_files() -> list[str]:
    """扫描 data/ 下的 *.progress.json，统一用相对路径并去重。"""
    default = _default_progress_file()
    cwd = Path.cwd()
    seen: set[Path] = set()
    candidates: list[str] = []
    for d in ("data", "."):
        p = Path(d)
        if not p.exists():
            continue
        for f in sorted(p.glob("*.progress.json")):
            r = f.resolve()
            if r in seen:
                continue
            seen.add(r)
            try:
                candidates.append(str(r.relative_to(cwd)))
            except ValueError:
                candidates.append(str(r))
    default_resolved = Path(default).resolve()
    if default_resolved not in seen and default not in candidates:
        candidates.insert(0, default)
    return candidates


class GeneratePage(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master)
        self._progress_path = tk.StringVar(value=_default_progress_file())
        self._total_games = 0
        self._current_game = 1
        self._games_cache: list[dict] = []
        self._games_by_number: dict[int, dict] = {}
        self._games_detail_path = ""
        self._board_size = 9
        self._last_mtime: float = 0
        self._replay_frames: list = []
        self._replay_index = 0
        self._win_length = 5

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

        # 进度区
        prog_frame = ttk.LabelFrame(self, text="进度")
        prog_frame.pack(fill=tk.X, padx=8, pady=8)
        self.progress_label = ttk.Label(prog_frame, text="未检测到生成任务（请用 five-generate 后台运行并指定进度文件）")
        self.progress_label.pack(fill=tk.X, padx=4, pady=2)
        self.progress_bar = ttk.Progressbar(prog_frame, mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=4, pady=4)

        # 分页与列表
        nav_frame = ttk.LabelFrame(self, text="对局列表与分页")
        nav_frame.pack(fill=tk.X, padx=8, pady=8)
        row0 = ttk.Frame(nav_frame)
        row0.pack(fill=tk.X, pady=4)
        ttk.Label(row0, text="总盘数:").pack(side=tk.LEFT, padx=4)
        self.total_var = tk.StringVar(value="0")
        ttk.Label(row0, textvariable=self.total_var).pack(side=tk.LEFT, padx=2)
        ttk.Button(row0, text="首盘", command=self._go_first).pack(side=tk.LEFT, padx=4)
        ttk.Button(row0, text="上一盘", command=self._go_prev).pack(side=tk.LEFT, padx=2)
        ttk.Label(row0, text="当前第").pack(side=tk.LEFT, padx=4)
        self.current_var = tk.StringVar(value="1")
        self.current_entry = ttk.Entry(row0, textvariable=self.current_var, width=8)
        self.current_entry.pack(side=tk.LEFT, padx=2)
        self.current_entry.bind("<Return>", lambda e: self._go_to_entry())
        ttk.Label(row0, text="盘").pack(side=tk.LEFT, padx=2)
        ttk.Button(row0, text="下一盘", command=self._go_next).pack(side=tk.LEFT, padx=4)
        ttk.Button(row0, text="尾盘", command=self._go_last).pack(side=tk.LEFT, padx=2)
        ttk.Button(row0, text="跳转", command=self._go_to_entry).pack(side=tk.LEFT, padx=4)

        # 每局对弈情况（最近，来自 progress 的 recent_games）
        list_frame = ttk.Frame(nav_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        scroll = ttk.Scrollbar(list_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.game_list = tk.Listbox(list_frame, height=8, yscrollcommand=scroll.set)
        self.game_list.pack(fill=tk.BOTH, expand=True)
        scroll.config(command=self.game_list.yview)
        self.game_list.bind("<ButtonRelease-1>", self._on_list_click)

        # 对局详情（回放样式：棋盘 + 步进 + 简要文字）
        detail_frame = ttk.LabelFrame(self, text="当前对局详情（回放）")
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        detail_body = ttk.Frame(detail_frame)
        detail_body.pack(fill=tk.BOTH, expand=True)
        self.replay_board = BoardCanvas(detail_body, board_size=self._board_size, pixel_size=360, show_coordinates=True)
        self.replay_board.pack(side=tk.LEFT, padx=8, pady=8)
        right_panel = ttk.Frame(detail_body)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        replay_ctl = ttk.Frame(right_panel)
        replay_ctl.pack(fill=tk.X, pady=4)
        ttk.Button(replay_ctl, text="上一手", command=self._replay_prev).pack(side=tk.LEFT, padx=4)
        ttk.Button(replay_ctl, text="下一手", command=self._replay_next).pack(side=tk.LEFT, padx=4)
        self.replay_label = ttk.Label(replay_ctl, text="第 0 / 0 手")
        self.replay_label.pack(side=tk.LEFT, padx=8)
        self.detail_text = tk.Text(right_panel, height=14, state="disabled", wrap=tk.WORD)
        detail_scroll = ttk.Scrollbar(right_panel, command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_scroll.set)
        self.detail_text.pack(fill=tk.BOTH, expand=True)
        detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.after(800, self._poll_progress_file)

    def _refresh_progress_files(self) -> None:
        values = _scan_generate_progress_files()
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
            except (OSError, json.JSONDecodeError) as _:
                pass
        self.after(800, self._poll_progress_file)

    def _apply_progress(self, data: dict) -> None:
        self._total_games = int(data.get("total_games", 0))
        games_done = int(data.get("games_done", 0))
        running = data.get("running", False)
        self._board_size = int(data.get("board_size", 9))
        b = data.get("black_wins", 0)
        w = data.get("white_wins", 0)
        d = data.get("draws", 0)
        samples = data.get("samples", 0)
        out = data.get("output_path", "")

        self.total_var.set(str(self._total_games))
        self.progress_bar["maximum"] = max(self._total_games, 1)
        self.progress_bar["value"] = games_done
        if running:
            self.progress_label.config(
                text=f"生成中… 局数 {games_done}/{self._total_games}  黑胜 {b}  白胜 {w}  和 {d}  样本 {samples}  → {out}"
            )
        else:
            self.progress_label.config(
                text=f"已结束  局数 {games_done}/{self._total_games}  黑胜 {b}  白胜 {w}  和 {d}  样本 {samples}  → {out}"
            )

        recent = data.get("recent_games", [])
        self.game_list.delete(0, tk.END)
        for g in recent:
            winner = g.get("winner", 0)
            moves = g.get("moves", 0)
            num = g.get("game", 0)
            winner_str = "黑胜" if winner == 1 else "白胜" if winner == -1 else "和"
            self.game_list.insert(tk.END, f"局 {num}  {winner_str}  {moves} 手")

        self._games_detail_path = _games_detail_file_from_progress(self._progress_path.get().strip())
        self._ensure_cache_up_to(games_done)
        self._clamp_current()
        # 不刷新盘数输入框和详情区，避免列表重填时把用户当前查看的局冲掉
        self._restore_list_selection(recent)

    def _ensure_cache_up_to(self, games_done: int) -> None:
        path = self._games_detail_path
        if not path or not Path(path).exists():
            self._games_cache = []
            self._games_by_number = {}
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self._games_cache = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                self._games_cache.append(json.loads(line))
            self._games_by_number = {int(r["game"]): r for r in self._games_cache}
        except (OSError, json.JSONDecodeError, KeyError):
            self._games_cache = []
            self._games_by_number = {}

    def _clamp_current(self) -> None:
        total = self._total_games or 1
        self._current_game = max(1, min(self._current_game, total))

    def _go_first(self) -> None:
        self._current_game = 1
        self.current_var.set("1")
        self._show_detail()

    def _go_last(self) -> None:
        self._current_game = self._total_games or 1
        self.current_var.set(str(self._current_game))
        self._show_detail()

    def _go_prev(self) -> None:
        self._current_game = max(1, self._current_game - 1)
        self.current_var.set(str(self._current_game))
        self._show_detail()

    def _go_next(self) -> None:
        total = self._total_games or 1
        self._current_game = min(total, self._current_game + 1)
        self.current_var.set(str(self._current_game))
        self._show_detail()

    def _go_to_entry(self) -> None:
        try:
            n = int(self.current_var.get())
            self._current_game = max(1, min(self._total_games or 1, n))
            self.current_var.set(str(self._current_game))
            self._show_detail()
        except ValueError:
            self.current_var.set(str(self._current_game))

    def _restore_list_selection(self, recent: list) -> None:
        """进度刷新后重填列表，恢复选中当前局（仅同步选中项，不刷新详情）。"""
        for i, g in enumerate(recent):
            if g.get("game") == self._current_game:
                self.game_list.selection_clear(0, tk.END)
                self.game_list.selection_set(i)
                self.game_list.see(i)
                break

    def _on_list_click(self, event: tk.Event) -> None:
        """仅响应用户点击列表项，滚动列表不会触发，避免误刷新盘数与详情。"""
        sel = self.game_list.curselection()
        if not sel:
            return
        idx = sel[0]
        try:
            line = self.game_list.get(idx)
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "局":
                game_num = int(parts[1])
                if game_num == self._current_game:
                    return
                self._current_game = game_num
                self.current_var.set(str(self._current_game))
                self._show_detail()
        except (ValueError, IndexError):
            pass

    def _show_detail(self) -> None:
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", tk.END)
        n = self._current_game
        total = self._total_games or 0
        self._replay_frames = []
        self._replay_index = 0
        bs = self._board_size
        if total == 0:
            self.detail_text.insert(tk.END, "暂无对局数据；请先运行 five-generate 并指定进度文件。\n")
            self.detail_text.configure(state="disabled")
            self._render_replay()
            return
        rec = self._games_by_number.get(n)
        if rec is None:
            self.detail_text.insert(
                tk.END,
                f"第 {n} 盘：详情文件尚未包含本局，或尚未生成。\n（当前缓存 {len(self._games_cache)} 局）\n",
            )
            self.detail_text.configure(state="disabled")
            self._render_replay()
            return
        winner = rec.get("winner", 0)
        moves_count = rec.get("moves", 0)
        actions = rec.get("actions", [])
        winner_str = "黑胜" if winner == 1 else "白胜" if winner == -1 else "和"
        self.detail_text.insert(tk.END, f"局号: {n}\n")
        self.detail_text.insert(tk.END, f"结果: {winner_str}\n")
        self.detail_text.insert(tk.END, f"手数: {moves_count}\n")
        self.detail_text.insert(tk.END, "使用「上一手」「下一手」回放对局。\n")
        self.detail_text.configure(state="disabled")

        move_list = []
        for idx in actions:
            if isinstance(idx, (list, tuple)):
                move_list.append(Move(row=int(idx[0]), col=int(idx[1])))
            else:
                move_list.append(Move.from_index(int(idx), bs))
        self._replay_frames = reconstruct_states(move_list, board_size=bs, win_length=self._win_length)
        self._replay_index = 0
        self.replay_board.set_board_size(bs)
        self._render_replay()

    def _replay_prev(self) -> None:
        if self._replay_index > 0:
            self._replay_index -= 1
            self._render_replay()

    def _replay_next(self) -> None:
        if self._replay_frames and self._replay_index + 1 < len(self._replay_frames):
            self._replay_index += 1
            self._render_replay()

    def _render_replay(self) -> None:
        total_ply = len(self._replay_frames)
        if not self._replay_frames:
            self.replay_label.config(text="第 0 / 0 手")
            from five.core.state import GameState
            empty = GameState.new(board_size=self.replay_board.board_size, win_length=self._win_length)
            self.replay_board.render(empty)
            return
        frame = self._replay_frames[self._replay_index]
        num_played = len(frame.state.history)
        total_moves = total_ply - 1 if total_ply > 0 else 0
        self.replay_label.config(text=f"第 {num_played} / {total_moves} 手")
        move_history = [
            (m.row, m.col, 1 if i % 2 == 0 else -1)
            for i, m in enumerate(frame.state.history)
        ]
        self.replay_board.render(frame.state, move_history=move_history)
