from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from five.storage.schemas import MoveRecord


class MoveDetailPanel(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.text = tk.Text(self, width=40, height=30, state="disabled")
        self.text.pack(fill=tk.BOTH, expand=True)

    def show_move(self, move: MoveRecord | None) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        if move is None:
            self.text.insert(tk.END, "选择一步后显示详细信息。\n")
        else:
            player = "黑棋" if move.player == 1 else "白棋"
            self.text.insert(tk.END, f"手数: {move.move_index}\n")
            self.text.insert(tk.END, f"执子: {player}\n")
            self.text.insert(tk.END, f"落点: ({move.row}, {move.col})\n")
            self.text.insert(tk.END, f"动作概率: {move.action_probability:.4f}\n")
            self.text.insert(tk.END, f"价值估计: {move.value_before:.4f}\n")
            self.text.insert(tk.END, f"合法着法数: {move.legal_count}\n")

            self.text.insert(tk.END, "\n" + "=" * 35 + "\n")
            self.text.insert(tk.END, "奖励明细:\n")
            self.text.insert(tk.END, "=" * 35 + "\n")
            self.text.insert(tk.END, f"总得（扣）分: {move.total_reward:+.4f}\n\n")

            if hasattr(move, 'reward_details') and move.reward_details:
                self.text.insert(tk.END, "明细:\n")
                for detail in move.reward_details:
                    sign = "+" if detail.amount >= 0 else ""
                    self.text.insert(tk.END, f"  {sign}{detail.amount:+.4f} - {detail.reason}\n")
            else:
                self.text.insert(tk.END, "  （无明细数据）\n")

            self.text.insert(tk.END, "\n" + "=" * 35 + "\n")
            self.text.insert(tk.END, "Top-K 候选:\n")
            for item in move.policy_topk:
                self.text.insert(
                    tk.END,
                    f"- ({item.row}, {item.col}) score={item.score:.4f}\n",
                )
        self.text.configure(state="disabled")
