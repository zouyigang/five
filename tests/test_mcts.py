import numpy as np
import pytest
import torch

from five.ai.mcts import MCTSConfig, MCTSEngine, _Node, _terminal_value
from five.ai.model import PolicyValueNet
from five.core.board import Board
from five.core.move import Move
from five.core.state import GameState


def _engine(simulations: int = 64) -> MCTSEngine:
    """刻意用未训练的随机网络：找到的战术必须来自搜索本身，而非网络先验。"""
    torch.manual_seed(0)
    model = PolicyValueNet(board_size=9, channels=8, blocks=1)
    return MCTSEngine(model, device="cpu", config=MCTSConfig(simulations=simulations))


def _state(stones, player: int) -> GameState:
    board = Board(size=9, win_length=5)
    for row, col, side in stones:
        board.grid[row, col] = side
    return GameState(board=board, current_player=player)


# ---------------------------------------------------------------- 符号约定
def test_terminal_value_is_from_the_side_to_move_perspective():
    """apply_move 在终局不翻转 current_player，所以终局价值不能按它判断视角。

    约定：有人获胜时「本应轮到走子的一方」必然是输家，故为 -1，配合回传逐层取负。
    """
    won = _state([(4, c, 1) for c in range(5)], player=1)
    won.winner = 1
    assert _terminal_value(won) == -1.0

    drawn = _state([], player=1)
    drawn.winner = 0
    assert _terminal_value(drawn) == 0.0


def test_backup_alternates_sign_along_the_path():
    root, child, grandchild = _Node(1.0), _Node(0.5), _Node(0.5)

    MCTSEngine._backup([root, child, grandchild], value=1.0)

    # 叶子拿 +1，其父为对立方拿 -1，根再取负拿 +1
    assert grandchild.value_sum == pytest.approx(1.0)
    assert child.value_sum == pytest.approx(-1.0)
    assert root.value_sum == pytest.approx(1.0)
    assert [n.visits for n in (root, child, grandchild)] == [1, 1, 1]


# ---------------------------------------------------------------- 端到端
def test_search_finds_an_immediate_win_with_an_untrained_network():
    """赢只需 1 层：任一子节点即终局，少量模拟即可。"""
    state = _state([(4, c, 1) for c in range(4)], player=1)

    result = _engine(simulations=64).select_move(state, temperature=0.0)

    assert result.action == Move(4, 4)
    assert result.value_estimate > 0


def test_search_blocks_an_immediate_loss_but_needs_far_more_simulations():
    """挡需要 2 层，随机先验下代价高得多。

    实测同一局面：赢在一手 64 次模拟即可，挡在一手要 2048 次（32 倍）——先走非挡点后
    还得让对方在约 76 个点里恰好选中制胜点。这说明搜索放大先验、替代不了先验：
    真实使用时网络先验已把概率压在挡点上，所需模拟数远低于此。
    """
    state = _state([(4, c, -1) for c in range(4)], player=1)

    assert _engine(simulations=64).select_move(state.copy(), 0.0).action != Move(4, 4)
    assert _engine(simulations=2048).select_move(state.copy(), 0.0).action == Move(4, 4)


def test_batched_search_matches_single_state_search():
    """跨对局批量必须与逐个求解一致，否则批量化会改变行为。"""
    states = [
        _state([(4, c, 1) for c in range(4)], player=1),
        _state([(4, c, -1) for c in range(4)], player=1),
        _state([(2, 2, 1), (3, 3, 1), (6, 6, -1)], player=-1),
    ]

    batched = _engine().select_moves([s.copy() for s in states], temperature=0.0)
    single = [_engine().select_move(s.copy(), temperature=0.0) for s in states]

    assert [r.action for r in batched] == [r.action for r in single]
    for left, right in zip(batched, single):
        # 批量组成不同会改变浮点归约顺序，容差不能设到 1e-9
        assert left.action_probability == pytest.approx(right.action_probability, abs=1e-6)
        assert left.value_estimate == pytest.approx(right.value_estimate, abs=1e-6)


def test_search_does_not_mutate_the_input_state():
    state = _state([(4, c, 1) for c in range(4)], player=1)
    before = state.board.grid.copy()

    _engine(simulations=32).select_move(state, temperature=0.0)

    assert np.array_equal(state.board.grid, before)
    assert state.current_player == 1


def test_visit_counts_concentrate_on_the_winning_move():
    state = _state([(4, c, 1) for c in range(4)], player=1)

    top = _engine(simulations=64).select_move(state, temperature=0.0).candidates[0]

    assert top.move == Move(4, 4)
    assert top.visits is not None and top.visits > 1
    assert top.score > 0.2, "访问应明显偏向制胜手，而非均匀铺开"


def test_search_only_returns_legal_moves_on_a_nearly_full_board():
    """近乎填满的棋盘：大量分支立即终局，搜索不应崩溃或走非法手。"""
    stones = [
        (row, col, 1 if (row + col) % 2 == 0 else -1)
        for row in range(8)
        for col in range(9)
    ]
    state = _state(stones, player=1)

    result = _engine(simulations=64).select_move(state, temperature=0.0)

    assert state.board.grid[result.action.row, result.action.col] == 0


def test_temperature_sampling_stays_within_legal_moves():
    state = _state([(4, 4, 1), (3, 3, -1)], player=1)

    for _ in range(5):
        result = _engine(simulations=32).select_move(state.copy(), temperature=1.0)
        assert state.board.grid[result.action.row, result.action.col] == 0


def test_dirichlet_noise_perturbs_root_priors_without_breaking_search():
    state = _state([(4, c, 1) for c in range(4)], player=1)
    torch.manual_seed(0)
    model = PolicyValueNet(board_size=9, channels=8, blocks=1)
    engine = MCTSEngine(
        model, device="cpu",
        config=MCTSConfig(simulations=64, dirichlet_weight=0.25),
    )

    result = engine.select_move(state, temperature=0.0)

    assert result.action == Move(4, 4), "加噪不应让搜索错过唾手可得的制胜手"


def test_low_simulation_sampling_falls_back_to_the_prior_ordering():
    """开局分支约 80 个，低模拟数下访问计数是噪声，必须靠先验排序。

    实测 64 次模拟时前 5 名访问数是 [2,2,1,1,1]（仅占 11%），纯按访问排序会把角和边
    排进前列；加上先验做同分打破后，前 5 全部落在中心邻域。
    """
    from five.core.game import GomokuGame

    torch.manual_seed(0)
    model = PolicyValueNet(board_size=9, channels=8, blocks=1)
    engine = MCTSEngine(
        model, device="cpu", config=MCTSConfig(simulations=8, sample_top_k=3)
    )
    game = GomokuGame(board_size=9, win_length=5)
    state = game.new_game()
    state.apply_move(Move(4, 4))

    # 模拟次数远少于分支数：绝大多数子节点访问数为 0，排序完全由先验决定
    roots = engine._run_search([state.copy()], np.random.default_rng(0))
    root = roots[0]
    ranking = {a: root.children[a].visits + root.children[a].prior for a in root.children}
    top = max(ranking, key=ranking.get)
    priors_only = max(root.children, key=lambda a: root.children[a].prior)
    visited = [a for a in root.children if root.children[a].visits > 0]

    assert len(visited) < len(root.children) // 2, "本用例要求访问计数稀疏"
    assert ranking[top] > 0
    # 无访问信息时，排序必须退化为先验排序
    unvisited_ranking = {a: v for a, v in ranking.items() if root.children[a].visits == 0}
    assert max(unvisited_ranking, key=unvisited_ranking.get) == max(
        unvisited_ranking, key=lambda a: root.children[a].prior
    )
    assert priors_only in root.children


def test_sample_top_k_restricts_choices_to_the_best_ranked_moves():
    torch.manual_seed(0)
    model = PolicyValueNet(board_size=9, channels=8, blocks=1)
    engine = MCTSEngine(
        model, device="cpu", config=MCTSConfig(simulations=64, sample_top_k=3)
    )
    board = Board(size=9, win_length=5)
    board.grid[4, 4] = 1
    board.grid[3, 3] = -1
    state = GameState(board=board, current_player=1)

    chosen = {
        (r.action.row, r.action.col)
        for r in (engine.select_move(state.copy(), temperature=1.0) for _ in range(15))
    }

    assert 1 < len(chosen) <= 3, f"应在前 3 名内取多样性，实际 {chosen}"
