from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
import torch

from five.ai.inference import ModelAIEngine
from five.ai.mcts import MCTSConfig, MCTSEngine
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
        # 搜索强度就是难度：模拟数越少算得越浅，像一个水平较低但正常的对手。
        # 这比调采样温度好——温度只会让 AI 偶尔走一手蠢棋，行为不自然。
        # 单局对弈用不上跨对局批量，每次模拟即一次 batch=1 前向，
        # 实测本机 64/200/800 模拟约 113/390/1983 ms 一手。
        self.selected_search = tk.StringVar(value="标准(200)")
        # 开局随机手数。推理本该全程取最优手，但双方都确定性时同一开局必然走出同一盘棋
        # （实测 24 局只有 1 局不同），人只要摸到一条赢棋路线就能无限重放。
        # 把随机性集中在开局：前 N 手按分布采样，之后全程最优，中后盘棋力一点不损失。
        self.selected_opening = tk.StringVar(value="2 手")
        self._ai_move_count = 0
        self._search_engines: dict[int, MCTSEngine] = {}
        self._model: PolicyValueNet | None = None
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
        self.opening_box = ttk.Combobox(
            top,
            textvariable=self.selected_opening,
            state="readonly",
            width=8,
            values=["关闭", "1 手", "2 手", "4 手"],
        )
        self.search_box = ttk.Combobox(
            top,
            textvariable=self.selected_search,
            state="readonly",
            width=12,
            values=["关闭(直出)", "快(64)", "标准(200)", "强(800)"],
        )
        self.run_box.pack(side=tk.LEFT, padx=4)
        self.model_box.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="刷新", command=self.refresh_runs).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="搜索:").pack(side=tk.LEFT)
        self.search_box.pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="开局随机:").pack(side=tk.LEFT)
        self.opening_box.pack(side=tk.LEFT, padx=4)
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
        self._model = PolicyValueNet(board_size=board_size, channels=channels, blocks=blocks)
        self.engine = ModelAIEngine(self._model)
        self.engine.load_checkpoint(checkpoint_path)
        # 换模型后原有的搜索引擎持有旧权重，必须丢弃。
        self._search_engines.clear()
        self.current_model_path = checkpoint_path
        self.model_loaded = True
        self.status_var.set(f"已加载模型: {self.selected_model.get()}")
        self.new_game()

    def new_game(self) -> None:
        if not self.model_loaded:
            self.status_var.set("请先加载模型。")
            return
        self.state = GameState.new(board_size=self.board_size, win_length=self.win_length)
        self._ai_move_count = 0
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
        engine = self._active_engine()
        analysis = engine.select_move(self.state.copy(), temperature=self._move_temperature())
        self.after(0, lambda: self._apply_ai_move(analysis))

    def _describe_ai_move(self, analysis) -> str:
        """把这一手的依据摘要到状态栏：搜索强度、局面估值、首选手的访问占比。

        估值是**走子方视角**，正数表示 AI 认为自己占优。
        """
        simulations = self._search_simulations()
        source = "直出" if simulations <= 0 else f"搜索{simulations}"
        # 这一手是否还在开局随机阶段——落子后计数已 +1，故用 <= 判断。
        opening = self._ai_move_count <= self._opening_random_moves()
        parts = [f"AI({source}{'·开局随机' if opening else ''})", f"估值 {analysis.value_estimate:+.2f}"]
        if analysis.candidates:
            top = analysis.candidates[0]
            if top.visits is not None:
                parts.append(f"首选 ({top.move.row},{top.move.col}) 访问 {top.score:.0%}")
        return " | ".join(parts)

    def _search_simulations(self) -> int:
        """下拉框选项 -> 模拟次数；0 表示不搜索，直接用网络输出。"""
        mapping = {"关闭(直出)": 0, "快(64)": 64, "标准(200)": 200, "强(800)": 800}
        return mapping.get(self.selected_search.get(), 200)

    def _active_engine(self):
        """按当前搜索强度返回引擎；搜索引擎按模拟次数缓存，复用同一份权重。"""
        simulations = self._search_simulations()
        if simulations <= 0 or self._model is None:
            return self.engine
        if simulations not in self._search_engines:
            self._search_engines[simulations] = MCTSEngine(
                self._model,
                # 开局采样只在搜索认可的前 5 手里选，否则低模拟数下访问计数几乎是平的，
                # 温度采样会抽到角、边这类明显坏手。
                config=MCTSConfig(simulations=simulations, sample_top_k=5),
            )
        return self._search_engines[simulations]

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
            self._ai_move_count += 1
            self.render()
            self._show_terminal_if_needed()
            if not self.state.is_terminal:
                self.status_var.set(f"轮到你。 {self._describe_ai_move(analysis)}")
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

    def _opening_random_moves(self) -> int:
        mapping = {"关闭": 0, "1 手": 1, "2 手": 2, "4 手": 4}
        return mapping.get(self.selected_opening.get(), 2)

    def _move_temperature(self) -> float:
        """AI 前 N 手按分布采样，之后取最优手。

        推理本该全程取最优，但那样同一开局会走出完全相同的一盘棋。把随机性限制在开局，
        既有多样性，中后盘（决定胜负的地方）又保持满强度。开搜索时采样的是访问计数，
        本身已高度集中在好手上，温度 1.0 仍然安全。
        """
        return 1.0 if self._ai_move_count < self._opening_random_moves() else 0.0

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
