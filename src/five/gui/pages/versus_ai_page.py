from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
import torch

from five.ai.inference import ModelAIEngine
from five.ai.model import PolicyValueNet
from five.common.utils import timestamp
from five.core.move import Move
from five.core.state import GameState
from five.gui.controllers import RunController
from five.gui.widgets.board_canvas import BoardCanvas
from five.storage.schemas import GameRecord, MoveRecord, MoveSummary


class VersusAIPage(ttk.Frame):
    def __init__(self, master, controller: RunController) -> None:
        super().__init__(master)
        self.controller = controller
        self.selected_run = tk.StringVar()
        self.selected_model = tk.StringVar()
        self.selected_difficulty = tk.StringVar(value="标准")
        self.human_first = tk.BooleanVar(value=True)
        self._run_lookup: dict[str, Path] = {}
        self._model_lookup: dict[str, str] = {}
        self.current_run_path: Path | None = None
        self.current_model_path: str | None = None
        self.board_size = 9
        self.win_length = 5
        self.current_game_moves: list[MoveRecord] = []
        self.saved_current_game = False
        self.model_loaded = False

        self.state = GameState.new(board_size=self.board_size, win_length=self.win_length)
        self.engine = ModelAIEngine(PolicyValueNet(board_size=self.board_size))
        self.ai_busy = False

        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=8)
        self.run_box = ttk.Combobox(top, textvariable=self.selected_run, state="readonly", width=35)
        self.model_box = ttk.Combobox(top, textvariable=self.selected_model, state="readonly", width=35)
        self.difficulty_box = ttk.Combobox(
            top,
            textvariable=self.selected_difficulty,
            state="readonly",
            width=10,
            values=["固定", "稳健", "标准", "探索"],
        )
        self.run_box.pack(side=tk.LEFT, padx=4)
        self.model_box.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="刷新", command=self.refresh_runs).pack(side=tk.LEFT, padx=4)
        self.difficulty_box.pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(top, text="人类先手", variable=self.human_first).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="新对局", command=self.new_game).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="加载模型", command=self.load_model).pack(side=tk.LEFT, padx=4)

        self.board = BoardCanvas(self)
        self.board.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.board.set_click_handler(self.on_human_move)

        self.status_var = tk.StringVar(value="请选择模型后开始对弈。")
        ttk.Label(self, textvariable=self.status_var).pack(fill=tk.X, padx=8, pady=8)

        self.run_box.bind("<<ComboboxSelected>>", lambda _: self.refresh_models())
        self.refresh_runs()
        self.render()

    def refresh_runs(self) -> None:
        runs = self.controller.list_runs()
        self._run_lookup = {run.name: run for run in runs}
        self.run_box["values"] = list(self._run_lookup.keys())
        if runs and not self.selected_run.get():
            self.selected_run.set(runs[-1].name)
        self.refresh_models()

    def refresh_models(self) -> None:
        run_path = self._run_lookup.get(self.selected_run.get())
        if run_path is None:
            return
        self.current_run_path = run_path
        models = self.controller.model_registry(run_path).list_models()
        self._model_lookup = {model.checkpoint_name: model.checkpoint_path for model in models}
        self.model_box["values"] = list(self._model_lookup.keys())
        if models and not self.selected_model.get():
            self.selected_model.set(models[-1].checkpoint_name)

    def load_model(self) -> None:
        checkpoint_path = self._model_lookup.get(self.selected_model.get())
        if not checkpoint_path:
            messagebox.showwarning("提示", "当前运行还没有可用模型。")
            return
        payload = torch.load(checkpoint_path, map_location="cpu")
        config = payload.get("config", {})
        board_size = int(config.get("board_size", 9))
        channels = int(config.get("model", {}).get("channels", 64))
        blocks = int(config.get("model", {}).get("blocks", 4))
        self.board_size = board_size
        self.win_length = int(config.get("win_length", 5))
        self.engine = ModelAIEngine(
            PolicyValueNet(board_size=board_size, channels=channels, blocks=blocks)
        )
        self.engine.load_checkpoint(checkpoint_path)
        self.current_model_path = checkpoint_path
        self.model_loaded = True
        self.status_var.set(f"已加载模型: {self.selected_model.get()}")
        self.new_game()

    def new_game(self) -> None:
        if not self.model_loaded:
            self.status_var.set("请先加载模型。")
            return
        self.state = GameState.new(board_size=self.board_size, win_length=self.win_length)
        self.current_game_moves = []
        self.saved_current_game = False
        self.render()
        if not self.human_first.get():
            self.request_ai_move()

    def on_human_move(self, move: Move) -> None:
        if self.ai_busy or self.state.is_terminal:
            return
        human_player = 1 if self.human_first.get() else -1
        if self.state.current_player != human_player or not self.state.board.is_legal(move):
            return
        self.current_game_moves.append(
            MoveRecord(
                move_index=len(self.current_game_moves) + 1,
                player=self.state.current_player,
                row=move.row,
                col=move.col,
                action_probability=1.0,
                value_before=0.0,
                legal_count=int(self.state.legal_mask().sum()),
                policy_topk=[],
            )
        )
        self.state.apply_move(move)
        self.render()
        self._show_terminal_if_needed()
        if not self.state.is_terminal:
            self.request_ai_move()

    def request_ai_move(self) -> None:
        if self.ai_busy or not self.model_loaded:
            return
        self.ai_busy = True
        self.status_var.set("AI 思考中...")
        threading.Thread(target=self._ai_worker, daemon=True).start()

    def _ai_worker(self) -> None:
        analysis = self.engine.select_move(self.state.copy(), temperature=self._difficulty_temperature())
        self.after(0, lambda: self._apply_ai_move(analysis))

    def _apply_ai_move(self, analysis) -> None:
        self.ai_busy = False
        move = analysis.action
        if not self.state.is_terminal and self.state.board.is_legal(move):
            self.current_game_moves.append(
                MoveRecord(
                    move_index=len(self.current_game_moves) + 1,
                    player=self.state.current_player,
                    row=move.row,
                    col=move.col,
                    action_probability=analysis.action_probability,
                    value_before=analysis.value_estimate,
                    legal_count=int(self.state.legal_mask().sum()),
                    policy_topk=[
                        MoveSummary(
                            row=item.move.row,
                            col=item.move.col,
                            score=item.score,
                            visits=item.visits,
                            value=item.value,
                        )
                        for item in analysis.candidates
                    ],
                )
            )
            self.state.apply_move(move)
            self.render()
            self._show_terminal_if_needed()
            if not self.state.is_terminal:
                self.status_var.set("轮到你。")
        else:
            self.status_var.set("AI 返回了非法着法。")

    def _show_terminal_if_needed(self) -> None:
        if not self.state.is_terminal:
            return
        self._save_finished_game()
        if self.state.winner == 0:
            self.status_var.set("本局平局。")
        else:
            winner = "黑棋" if self.state.winner == 1 else "白棋"
            self.status_var.set(f"对局结束，{winner} 获胜。")

    def render(self) -> None:
        self.board.render(self.state)

    def _difficulty_temperature(self) -> float:
        # 不同思考模式对应的温度：
        # - 固定：始终选择当前概率最大的动作（贪心，完全确定）
        # - 稳健：小幅探索
        # - 标准：中等探索
        # - 探索：更大随机性
        mapping = {
            "固定": 0.0,
            "稳健": 0.1,
            "标准": 0.2,
            "探索": 0.6,
        }
        return mapping.get(self.selected_difficulty.get(), 0.2)

    def _save_finished_game(self) -> None:
        if self.saved_current_game or self.current_run_path is None or not self.current_game_moves:
            return
        black_player = "human" if self.human_first.get() else self.selected_model.get()
        white_player = self.selected_model.get() if self.human_first.get() else "human"
        record = GameRecord(
            game_id=f"human_{timestamp()}",
            run_id=self.current_run_path.name,
            board_size=self.board_size,
            win_length=self.win_length,
            winner=self.state.winner,
            total_moves=len(self.current_game_moves),
            black_player=black_player,
            white_player=white_player,
            result="draw" if self.state.winner == 0 else "five_in_a_row",
            model_checkpoint=self.current_model_path,
            moves=list(self.current_game_moves),
        )
        self.controller.game_store(self.current_run_path).save(record)
        self.saved_current_game = True
