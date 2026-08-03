from five.gui.widgets.board_canvas import BoardCanvas
from five.gui.pages.pretrain_page import PretrainPage


def test_board_canvas_right_click_dispatches_callback_then_flushes() -> None:
    canvas = BoardCanvas.__new__(BoardCanvas)
    calls: list[str] = []
    canvas.on_right_click_callback = lambda: calls.append("undo")
    canvas.winfo_toplevel = lambda: canvas
    canvas.update_idletasks = lambda: calls.append("flush")

    result = canvas._handle_right_click(None)

    assert calls == ["undo", "flush"]
    assert result == "break"


def test_board_canvas_right_click_without_callback_is_ignored() -> None:
    canvas = BoardCanvas.__new__(BoardCanvas)
    canvas.on_right_click_callback = None

    assert canvas._handle_right_click(None) is None
    assert canvas._handle_right_click_release(None) is None


def test_board_canvas_right_click_release_only_repaints() -> None:
    canvas = BoardCanvas.__new__(BoardCanvas)
    calls: list[str] = []
    flushes: list[str] = []
    canvas.on_right_click_callback = lambda: calls.append("undo")
    canvas.winfo_toplevel = lambda: canvas
    canvas.update_idletasks = lambda: flushes.append("flush")
    # hwnd 为 0 时跳过 Win32 强制重绘，测试环境无真实窗口。
    canvas.winfo_id = lambda: 0

    result = canvas._handle_right_click_release(None)

    assert calls == []
    assert flushes == ["flush"]
    assert result == "break"


def test_pretrain_polling_only_runs_while_page_is_active() -> None:
    page = PretrainPage.__new__(PretrainPage)
    page._active = False
    page._poll_after_id = None
    polls: list[str] = []
    cancelled: list[str] = []
    page._poll_progress_file = lambda: polls.append("poll")
    page.after_cancel = lambda after_id: cancelled.append(after_id)

    page.set_active(True)
    page.set_active(True)
    assert polls == ["poll"]

    page._poll_after_id = "after#1"
    page.set_active(False)
    assert cancelled == ["after#1"]
    assert page._poll_after_id is None
