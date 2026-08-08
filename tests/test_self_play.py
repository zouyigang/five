import pytest
import torch

from five.ai.inference import ModelAIEngine
from five.ai.interfaces import AnalysisResult, CandidateMove
from five.ai.model import PolicyValueNet
from five.core.game import GomokuGame
from five.core.move import Move
from five.core.state import GameState
from five.train.self_play import SelfPlaySpec, play_self_play_games


class _DeterministicEngine:
    """纯由局面决定走子的引擎：批量与串行必须给出完全相同的对局。

    column_major 让两个实例产生不同的落子顺序，从而得到不同长度的对局，
    用来覆盖「各局先后结束」的调度分支。
    """

    def __init__(self, column_major: bool = False, start: tuple[int, int] = (0, 0)) -> None:
        self.column_major = column_major
        self.start = start
        self.calls = 0

    def load_checkpoint(self, path: str) -> None:  # pragma: no cover - 接口占位
        pass

    def _pick(self, state: GameState):
        moves = state.legal_moves()
        # 空盘时按各自的起手点分叉，保证不同实例产生结构不同、长度不同的对局；
        # 仍然只依赖局面，因此批量与串行必然一致。
        if len(moves) == state.board.size**2:
            opening = Move(row=self.start[0], col=self.start[1])
            if opening in moves:
                return opening
        key = (lambda m: (m.col, m.row)) if self.column_major else (lambda m: (m.row, m.col))
        return min(moves, key=key)

    def select_move(self, state: GameState, temperature: float = 0.0) -> AnalysisResult:
        self.calls += 1
        move = self._pick(state)
        return AnalysisResult(
            action=move,
            action_probability=0.5,
            value_estimate=0.25,
            candidates=[CandidateMove(move=move, score=0.5)],
        )

    def analyze(self, state: GameState, top_k: int = 5) -> list[CandidateMove]:  # pragma: no cover
        return [CandidateMove(move=self._pick(state), score=0.5)]


def _specs(engines) -> list[SelfPlaySpec]:
    return [
        SelfPlaySpec(game_index=index + 1, black_engine=engine, white_engine=engine)
        for index, engine in enumerate(engines)
    ]


def _summarise(results):
    return [
        (
            result.record.game_id,
            result.record.winner,
            result.record.total_moves,
            [(m.row, m.col, m.player) for m in result.record.moves],
            [round(m.total_reward, 6) for m in result.record.moves],
            len(result.episode.transitions),
        )
        for result in results
    ]


def test_batched_self_play_matches_sequential_play_exactly():
    game = GomokuGame(board_size=9, win_length=5)
    engines = [_DeterministicEngine(column_major=bool(i % 2), start=(i, i)) for i in range(6)]

    batched = play_self_play_games(game, _specs(engines), run_id="r")
    sequential = [
        play_self_play_games(game, [spec], run_id="r")[0]
        for spec in _specs([_DeterministicEngine(column_major=bool(i % 2), start=(i, i)) for i in range(6)])
    ]

    assert _summarise(batched) == _summarise(sequential)


def test_batched_self_play_handles_games_of_different_lengths():
    game = GomokuGame(board_size=9, win_length=5)
    engines = [_DeterministicEngine(column_major=bool(i % 2), start=(i, i)) for i in range(4)]

    results = play_self_play_games(game, _specs(engines), run_id="r")

    lengths = {result.record.total_moves for result in results}
    assert len(lengths) > 1, "两种落子顺序应产生不同长度的对局"
    assert all(result.record.total_moves > 0 for result in results)
    # 每局都必须真正走到终局，不能因为别的局先结束而被提前中断
    assert all(
        result.record.winner != 0 or result.record.total_moves == 81 for result in results
    )


def test_tracked_players_still_limits_transitions_to_one_side():
    game = GomokuGame(board_size=9, win_length=5)
    engine = _DeterministicEngine()
    specs = [
        SelfPlaySpec(
            game_index=1,
            black_engine=engine,
            white_engine=engine,
            tracked_players={1},
        )
    ]

    result = play_self_play_games(game, specs, run_id="r")[0]

    assert result.episode.transitions
    assert {t.player for t in result.episode.transitions} == {1}


def test_model_engine_batched_selection_matches_single_calls():
    """批量前向必须与逐个调用给出相同结果，否则批量化会悄悄改变策略行为。"""
    torch.manual_seed(0)
    model = PolicyValueNet(board_size=9, channels=8, blocks=1)
    engine = ModelAIEngine(model, device="cpu")
    game = GomokuGame(board_size=9, win_length=5)

    states = []
    state = game.new_game()
    for row, col in [(4, 4), (3, 3), (4, 5), (2, 2)]:
        states.append(state.copy())
        state.apply_move(Move(row=row, col=col))
    states.append(state.copy())

    batched = engine.select_moves(states, temperature=0.0)
    single = [engine.select_move(s, temperature=0.0) for s in states]

    assert [result.action for result in batched] == [result.action for result in single]
    for left, right in zip(batched, single):
        assert left.action_probability == pytest.approx(right.action_probability, abs=1e-5)
        assert left.value_estimate == pytest.approx(right.value_estimate, abs=1e-5)


def test_opponent_temperature_is_applied_only_to_the_untracked_side():
    """网络对手必须能用自己的温度：模型探索温度会把它明显削弱。"""
    game = GomokuGame(board_size=9, win_length=5)
    seen: list[tuple[str, float]] = []

    class _TemperatureProbe(_DeterministicEngine):
        def __init__(self, label: str) -> None:
            super().__init__()
            self.label = label

        def select_move(self, state, temperature=0.0):
            seen.append((self.label, temperature))
            return super().select_move(state, temperature=temperature)

    model = _TemperatureProbe("model")
    opponent = _TemperatureProbe("opponent")
    specs = [
        SelfPlaySpec(
            game_index=1,
            black_engine=model,
            white_engine=opponent,
            tracked_players={1},
            opponent_temperature=0.35,
        )
    ]

    play_self_play_games(game, specs, run_id="r", temperature=1.3)

    assert {t for label, t in seen if label == "model"} == {1.3}
    assert {t for label, t in seen if label == "opponent"} == {0.35}


def test_self_play_games_ignore_opponent_temperature():
    """自博弈局双方都被追踪，全程用模型温度。"""
    game = GomokuGame(board_size=9, win_length=5)
    seen: list[float] = []

    class _Probe(_DeterministicEngine):
        def select_move(self, state, temperature=0.0):
            seen.append(temperature)
            return super().select_move(state, temperature=temperature)

    engine = _Probe()
    specs = [SelfPlaySpec(game_index=1, black_engine=engine, opponent_temperature=0.35)]

    play_self_play_games(game, specs, run_id="r", temperature=1.3)

    assert set(seen) == {1.3}
