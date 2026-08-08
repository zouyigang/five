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


def _versus_page():
    """不启 Tk：只构造对象并塞入必要属性，测选择逻辑本身。"""
    from five.gui.pages.versus_ai_page import VersusAIPage

    page = VersusAIPage.__new__(VersusAIPage)
    page._search_engines = {}
    page._model = None
    page.engine = object()
    page._ai_move_count = 0
    page.selected_opening = _fixed("2 手")
    return page


def _fixed(value: str):
    """替代 tk.StringVar：只需要 .get()。"""
    return type("V", (), {"get": staticmethod(lambda v=value: v)})()


def test_search_off_uses_the_raw_network_engine():
    page = _versus_page()
    page.selected_search = _fixed("关闭(直出)")

    assert page._search_simulations() == 0
    assert page._active_engine() is page.engine


def test_search_levels_map_to_simulation_counts():
    page = _versus_page()
    for label, expected in [("快(64)", 64), ("标准(200)", 200), ("强(800)", 800)]:
        page.selected_search = _fixed(label)
        assert page._search_simulations() == expected


def test_search_engines_are_cached_per_simulation_count():
    from five.ai.model import PolicyValueNet

    page = _versus_page()
    page._model = PolicyValueNet(board_size=9, channels=8, blocks=1)
    page.selected_search = _fixed("快(64)")

    first = page._active_engine()
    second = page._active_engine()

    assert first is second, "同一强度应复用引擎，不该每手重建"
    assert first.config.simulations == 64


def test_no_model_falls_back_to_the_raw_engine():
    """模型未加载时不能去建搜索引擎（会拿到 None 权重）。"""
    page = _versus_page()
    page.selected_search = _fixed("强(800)")

    assert page._active_engine() is page.engine


def test_opening_moves_are_sampled_then_play_turns_greedy():
    """随机性只在开局：前 N 手采样，之后必须全程取最优手。"""
    page = _versus_page()
    page.selected_opening = _fixed("2 手")

    page._ai_move_count = 0
    assert page._move_temperature() == 1.0
    page._ai_move_count = 1
    assert page._move_temperature() == 1.0
    page._ai_move_count = 2
    assert page._move_temperature() == 0.0
    page._ai_move_count = 30
    assert page._move_temperature() == 0.0


def test_opening_randomisation_can_be_switched_off():
    page = _versus_page()
    page.selected_opening = _fixed("关闭")

    page._ai_move_count = 0
    assert page._opening_random_moves() == 0
    assert page._move_temperature() == 0.0, "关闭时第一手就该取最优"


def test_opening_levels_map_to_move_counts():
    page = _versus_page()
    for label, expected in [("关闭", 0), ("1 手", 1), ("2 手", 2), ("4 手", 4)]:
        page.selected_opening = _fixed(label)
        assert page._opening_random_moves() == expected
