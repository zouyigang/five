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


def _analysis(actions_scores):
    from five.ai.interfaces import AnalysisResult, CandidateMove
    from five.core.move import Move

    cands = [CandidateMove(move=Move(r, c), score=s) for (r, c), s in actions_scores]
    return AnalysisResult(
        action=cands[0].move,
        action_probability=cands[0].score,
        value_estimate=0.0,
        candidates=cands,
    )


def test_opening_randomness_only_picks_among_candidates():
    """回归：直出模式全分布采样能抽到概率 0% 的坏手；只在候选里采样可杜绝。"""
    page = _versus_page()
    page.selected_opening = _fixed("2 手")
    page._ai_move_count = 0
    analysis = _analysis([((4, 4), 0.5), ((4, 3), 0.3), ((3, 4), 0.2)])

    picked = set()
    for _ in range(40):
        action = page._apply_opening_choice(analysis).action
        picked.add((action.row, action.col))

    assert picked <= {(4, 4), (4, 3), (3, 4)}, f"越出候选范围: {picked}"
    assert len(picked) > 1, "开局阶段应有多样性"


def test_moves_past_the_opening_window_are_always_the_top_candidate():
    page = _versus_page()
    page.selected_opening = _fixed("2 手")
    page._ai_move_count = 2
    analysis = _analysis([((4, 4), 0.5), ((4, 3), 0.3)])

    for _ in range(10):
        result = page._apply_opening_choice(analysis)
        assert (result.action.row, result.action.col) == (4, 4)


def test_opening_randomisation_can_be_switched_off():
    page = _versus_page()
    page.selected_opening = _fixed("关闭")
    page._ai_move_count = 0
    analysis = _analysis([((4, 4), 0.5), ((4, 3), 0.3)])

    assert page._opening_random_moves() == 0
    action = page._apply_opening_choice(analysis).action
    assert (action.row, action.col) == (4, 4)


def test_zero_probability_candidates_are_never_played():
    page = _versus_page()
    page.selected_opening = _fixed("4 手")
    page._ai_move_count = 0
    analysis = _analysis([((4, 4), 0.9), ((0, 0), 0.0), ((8, 8), 0.0)])

    picked = set()
    for _ in range(40):
        action = page._apply_opening_choice(analysis).action
        picked.add((action.row, action.col))

    assert picked == {(4, 4)}, f"概率为 0 的手不该被选中: {picked}"


def test_opening_levels_map_to_move_counts():
    page = _versus_page()
    for label, expected in [("关闭", 0), ("1 手", 1), ("2 手", 2), ("4 手", 4)]:
        page.selected_opening = _fixed(label)
        assert page._opening_random_moves() == expected
