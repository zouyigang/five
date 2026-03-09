from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from five.core.move import Move
from five.core.state import GameState
from five.gui.widgets.board_canvas import BoardCanvas
from five.train.reward import (
    RewardResult,
    compute_hybrid_reward_with_details,
    compute_process_reward_with_details,
)


class RewardTestPage(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.board_size = 9
        self.win_length = 5
        self.state = GameState.new(board_size=self.board_size, win_length=self.win_length)
        self.last_reward_result: RewardResult | None = None
        self.move_history: list[tuple[int, int, int, RewardResult | None]] = []
        self.history_states: list[GameState] = []

        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(top, text="新对局", command=self.new_game).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="回退", command=self.undo).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="当前玩家:").pack(side=tk.LEFT, padx=4)
        self.current_player_var = tk.StringVar(value="黑棋")
        ttk.Label(top, textvariable=self.current_player_var, font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=4)

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.board = BoardCanvas(left_frame, show_coordinates=True)
        self.board.pack(fill=tk.BOTH, expand=True)
        self.board.set_click_handler(self.on_human_move)

        middle_frame = ttk.Frame(left_frame)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8)

        ttk.Label(middle_frame, text="落子顺序", font=("Arial", 12, "bold")).pack(fill=tk.X, pady=4)
        self.move_list_text = tk.Text(middle_frame, width=25, height=15, state="disabled")
        self.move_list_text.pack(fill=tk.BOTH, expand=True)

        ttk.Label(middle_frame, text="当前步坐标", font=("Arial", 12, "bold")).pack(fill=tk.X, pady=4)
        self.current_move_var = tk.StringVar(value="当前步: (,")
        ttk.Label(middle_frame, textvariable=self.current_move_var, font=("Arial", 14, "bold"), foreground="red").pack(fill=tk.X, pady=4)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=8)

        ttk.Label(right_frame, text="奖励分数与明细", font=("Arial", 12, "bold")).pack(fill=tk.X, pady=4)
        self.total_reward_var = tk.StringVar(value="总奖励: 0.00")
        ttk.Label(right_frame, textvariable=self.total_reward_var, font=("Arial", 14, "bold"), foreground="blue").pack(fill=tk.X, pady=4)

        self.reward_text = tk.Text(right_frame, width=40, height=25, state="disabled")
        self.reward_text.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="点击棋盘落子，查看奖励计算明细。")
        ttk.Label(self, textvariable=self.status_var).pack(fill=tk.X, padx=8, pady=8)

        self.render()

    def new_game(self) -> None:
        self.state = GameState.new(board_size=self.board_size, win_length=self.win_length)
        self.last_reward_result = None
        self.move_history = []
        self.history_states = []
        self.total_reward_var.set("总奖励: 0.00")
        self.current_move_var.set("当前步: (, )")
        self.status_var.set("点击棋盘落子，查看奖励计算明细。")
        self.render()
        self._clear_reward_details()
        self._clear_move_list()

    def undo(self) -> None:
        if not self.move_history or not self.history_states:
            return
        # 必须先恢复再 pop，否则会恢复到错误的历史（相当于回退两步）
        self.state = self.history_states[-1].copy()
        self.move_history.pop()
        self.history_states.pop()
        if self.move_history:
            last_row, last_col, last_player, last_reward = self.move_history[-1]
            self.last_reward_result = last_reward
            self.current_move_var.set(f"当前步: ({last_row}, {last_col})")
            if last_reward:
                self._show_reward_details(last_reward)
            else:
                self._clear_reward_details()
        else:
            self.last_reward_result = None
            self.current_move_var.set("当前步: (, )")
            self._clear_reward_details()
        self.render()
        self._update_move_list()
        self.status_var.set(f"已回退一步。")

    def on_human_move(self, move: Move) -> None:
        if self.state.is_terminal or not self.state.board.is_legal(move):
            return

        self.history_states.append(self.state.copy())

        player = self.state.current_player
        next_board = self.state.board.copy()
        next_board.apply_move(move, player)
        winner = next_board.check_winner(move)

        if winner != 0 or next_board.is_full():
            reward_result = compute_hybrid_reward_with_details(self.state.board, move, player, winner)
        else:
            reward_result = compute_process_reward_with_details(self.state.board, move, player)
        self.last_reward_result = reward_result

        self.move_history.append((move.row, move.col, player, reward_result))

        self.state.apply_move(move)
        self.render()
        self._show_reward_details(reward_result)
        self._update_move_list()
        self.current_move_var.set(f"当前步: ({move.row}, {move.col})")

        if self.state.is_terminal:
            if self.state.winner == 0:
                self.status_var.set("本局平局。")
            else:
                winner = "黑棋" if self.state.winner == 1 else "白棋"
                self.status_var.set(f"对局结束，{winner} 获胜。")
        else:
            next_player = "黑棋" if self.state.current_player == 1 else "白棋"
            self.status_var.set(f"等待 {next_player} 落子...")

    def _show_reward_details(self, reward_result: RewardResult) -> None:
        self.reward_text.configure(state="normal")
        self.reward_text.delete("1.0", tk.END)

        self.total_reward_var.set(f"总奖励: {reward_result.total_reward:+.2f}")

        self.reward_text.insert(tk.END, "=" * 35 + "\n")
        self.reward_text.insert(tk.END, "奖励明细:\n")
        self.reward_text.insert(tk.END, "=" * 35 + "\n\n")

        if reward_result.details:
            for detail in reward_result.details:
                sign = "+" if detail.amount >= 0 else ""
                self.reward_text.insert(tk.END, f"{sign}{detail.amount:+.4f} - {detail.reason}\n")
        else:
            self.reward_text.insert(tk.END, "（无奖励项）\n")

        self.reward_text.configure(state="disabled")

    def _clear_reward_details(self) -> None:
        self.reward_text.configure(state="normal")
        self.reward_text.delete("1.0", tk.END)
        self.reward_text.configure(state="disabled")

    def _update_move_list(self) -> None:
        self.move_list_text.configure(state="normal")
        self.move_list_text.delete("1.0", tk.END)

        for idx, (row, col, player, _) in enumerate(self.move_history, 1):
            color = "黑" if player == 1 else "白"
            self.move_list_text.insert(tk.END, f"{idx}. {color}: ({row}, {col})\n")

        self.move_list_text.configure(state="disabled")

    def _clear_move_list(self) -> None:
        self.move_list_text.configure(state="normal")
        self.move_list_text.delete("1.0", tk.END)
        self.move_list_text.configure(state="disabled")

    def render(self) -> None:
        move_history_for_board = [(row, col, player) for (row, col, player, _) in self.move_history]
        self.board.render(self.state, move_history=move_history_for_board)
        current_player = "黑棋" if self.state.current_player == 1 else "白棋"
        self.current_player_var.set(current_player)
