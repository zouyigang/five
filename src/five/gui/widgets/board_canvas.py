from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from typing import Callable

from five.core.move import Move
from five.core.state import GameState


class BoardCanvas(tk.Canvas):
    def __init__(self, master, board_size: int = 9, pixel_size: int = 540, show_coordinates: bool = False, **kwargs) -> None:
        super().__init__(
            master,
            width=pixel_size,
            height=pixel_size,
            bg="#d9a45c",
            highlightthickness=0,
            **kwargs,
        )
        self.board_size = board_size
        self.pixel_size = pixel_size
        self.cell = pixel_size / max(board_size, 1)
        self.on_click_callback: Callable[[Move], None] | None = None
        self.on_right_click_callback: Callable[[], None] | None = None
        self.show_coordinates = show_coordinates
        self.bind("<Button-1>", self._handle_click)
        self.bind("<Button-3>", self._handle_right_click)
        self.bind("<ButtonRelease-3>", self._handle_right_click_release)

    def set_board_size(self, board_size: int) -> None:
        self.board_size = board_size
        self.cell = self.pixel_size / max(board_size, 1)

    def render(
        self,
        state: GameState,
        highlights: list[tuple[int, int, float]] | None = None,
        move_history: list[tuple[int, int, int]] | None = None,
        bad_moves: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """bad_moves: list of (row, col, move_index) for stones marked as bad (e.g. human_rating=0)."""
        self.set_board_size(state.board.size)
        self.delete("all")
        self._draw_grid()
        if highlights:
            self._draw_highlights(highlights)

        if move_history:
            for move_number, (row, col, player) in enumerate(move_history, 1):
                self._draw_stone(row, col, player, move_number)
        else:
            for row in range(state.board.size):
                for col in range(state.board.size):
                    cell = int(state.board.grid[row, col])
                    if cell != 0:
                        self._draw_stone(row, col, cell)

        if state.last_move is not None:
            self._draw_last_move_marker(state.last_move.row, state.last_move.col)
        if bad_moves:
            self._draw_bad_move_markers(bad_moves)

    def set_click_handler(self, callback: Callable[[Move], None]) -> None:
        self.on_click_callback = callback

    def set_right_click_handler(self, callback: Callable[[], None]) -> None:
        self.on_right_click_callback = callback

    def _draw_grid(self) -> None:
        pad = self.cell / 2
        for index in range(self.board_size):
            offset = pad + index * self.cell
            self.create_line(pad, offset, self.pixel_size - pad, offset, fill="black")
            self.create_line(offset, pad, offset, self.pixel_size - pad, fill="black")

    def _draw_stone(self, row: int, col: int, player: int, move_number: int | None = None) -> None:
        radius = self.cell * 0.4
        x = self.cell / 2 + col * self.cell
        y = self.cell / 2 + row * self.cell
        fill = "black" if player == 1 else "white"
        text_color = "white" if player == 1 else "black"
        self.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline="black")
        
        if self.show_coordinates:
            if move_number is not None:
                self.create_text(x, y, text=str(move_number), fill=text_color, font=("Arial", 10, "bold"))
            else:
                self.create_text(x, y, text=f"({row},{col})", fill=text_color, font=("Arial", 10, "bold"))

    def _draw_last_move_marker(self, row: int, col: int) -> None:
        x = self.cell / 2 + col * self.cell
        y = self.cell / 2 + row * self.cell
        size = self.cell * 0.1
        self.create_rectangle(x - size, y - size, x + size, y + size, outline="red", width=2)

    def _draw_bad_move_markers(self, bad_moves: list[tuple[int, int, int]]) -> None:
        """Draw red ring and move index for each (row, col, move_index) marked as bad."""
        for row, col, move_index in bad_moves:
            x = self.cell / 2 + col * self.cell
            y = self.cell / 2 + row * self.cell
            radius = self.cell * 0.45
            self.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                outline="#c0392b", width=3,
            )
            self.create_text(x, y, text=str(move_index), fill="#c0392b", font=("Arial", 12, "bold"))

    def _draw_highlights(self, highlights: list[tuple[int, int, float]]) -> None:
        for row, col, strength in highlights:
            x0 = col * self.cell
            y0 = row * self.cell
            x1 = x0 + self.cell
            y1 = y0 + self.cell
            alpha = max(0.1, min(strength, 1.0))
            color = f"#{int(255 * (1 - alpha)):02x}{int(80 * (1 - alpha)):02x}ff"
            self.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

    def _handle_click(self, event) -> None:
        if self.on_click_callback is None:
            return
        col = int(event.x // self.cell)
        row = int(event.y // self.cell)
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            self.on_click_callback(Move(row=row, col=col))

    def _handle_right_click(self, _event) -> str | None:
        if self.on_right_click_callback is None:
            return None
        # 按下时立即回退，消除等待用户松开鼠标造成的体感延时。
        self.on_right_click_callback()
        # 不等事件循环回到空闲，立刻把重绘任务刷到屏幕。
        self.winfo_toplevel().update_idletasks()
        return "break"

    def _handle_right_click_release(self, _event) -> str | None:
        if self.on_right_click_callback is None:
            return None
        # 释放时只补交显示任务，不再次执行回退。
        self._force_window_repaint()
        return "break"

    def _force_window_repaint(self) -> None:
        toplevel = self.winfo_toplevel()
        toplevel.update_idletasks()
        # Windows 可能在右键捕获期间丢弃已提交的绘制，而 Tk 认为画过了不会再补，
        # 表现为松开后棋盘/落子列表停留在旧画面，直到下一次事件才刷新。
        # 这里通过 Win32 强制整棵窗口树无条件重绘一遍，不依赖 Tk 的待绘状态。
        if sys.platform == "win32":
            hwnd = toplevel.winfo_id()
            if not hwnd:
                return
            RDW_INVALIDATE = 0x0001
            RDW_ALLCHILDREN = 0x0080
            RDW_UPDATENOW = 0x0100
            try:
                ctypes.windll.user32.RedrawWindow(
                    hwnd, None, None, RDW_INVALIDATE | RDW_ALLCHILDREN | RDW_UPDATENOW
                )
            except OSError:
                pass
