from __future__ import annotations

import math
from dataclasses import dataclass

from five.ai.inference import ModelAIEngine
from five.ai.interfaces import AIEngine, select_moves_batched
from five.ai.players import HeuristicPlayer, RandomPlayer
from five.core.game import GomokuGame
from five.core.state import GameState


@dataclass(slots=True)
class EvalResult:
    win_rate_random: float
    win_rate_heuristic: float
    win_rate_random_black: float
    win_rate_random_white: float
    win_rate_heuristic_black: float
    win_rate_heuristic_white: float
    # 与本次 run 起点策略的对局胜率；没有锚点对手时为 NaN。
    win_rate_anchor: float = math.nan
    win_rate_anchor_black: float = math.nan
    win_rate_anchor_white: float = math.nan


def play_match(
    game: GomokuGame,
    model_engine: ModelAIEngine,
    opponent: AIEngine,
    games: int,
    model_player: int,
    *,
    opponent_temperature: float = 0.0,
) -> float:
    """并行推进整场比赛：每一拍把同侧待决策的局面凑成一次前向。

    与自博弈同理，逐局串行会让网络一直以 batch=1 运行。这里按「该模型走」还是
    「该对手走」分成两组，各自批量求解；两组都可能非空，因为各局长度与手数奇偶不同。
    """
    if games <= 0:
        return 0.0

    states: list[GameState] = [game.new_game() for _ in range(games)]
    while True:
        active = [state for state in states if not state.is_terminal]
        if not active:
            break
        model_states = [state for state in active if state.current_player == model_player]
        opponent_states = [state for state in active if state.current_player != model_player]
        if model_states:
            analyses = select_moves_batched(model_engine, model_states, temperature=0.0)
            for state, analysis in zip(model_states, analyses):
                state.apply_move(analysis.action)
        if opponent_states:
            analyses = select_moves_batched(
                opponent, opponent_states, temperature=opponent_temperature
            )
            for state, analysis in zip(opponent_states, analyses):
                state.apply_move(analysis.action)

    wins = sum(1 for state in states if state.winner == model_player)
    return wins / games


def evaluate_policy(
    game: GomokuGame,
    model_engine: ModelAIEngine,
    games: int,
    *,
    heuristic_temperature: float = 0.0,
    anchor_engine: AIEngine | None = None,
) -> EvalResult:
    """对随机、启发式、以及本次 run 起点策略（锚点）三类对手各评估一次。

    `anchor_engine` 是训练开始那一刻策略的冻结副本，全程不变，且**从不参与训练**。
    启发式对手占了自博弈的大头，只用它选模型等于拿练习册当考卷；与锚点的胜率
    衡量的是「比自己出发时强了多少」，无法通过适应某个固定对手来刷高。
    """
    random_black = play_match(game, model_engine, RandomPlayer(), games=games, model_player=1)
    random_white = play_match(game, model_engine, RandomPlayer(), games=games, model_player=-1)
    heuristic_black = play_match(
        game, model_engine, HeuristicPlayer(), games=games, model_player=1,
        opponent_temperature=heuristic_temperature,
    )
    heuristic_white = play_match(
        game, model_engine, HeuristicPlayer(), games=games, model_player=-1,
        opponent_temperature=heuristic_temperature,
    )

    anchor_black = anchor_white = anchor_mean = math.nan
    if anchor_engine is not None:
        # 双方都是确定性网络时每局会完全相同，用温度给锚点一侧引入多样性。
        anchor_black = play_match(
            game, model_engine, anchor_engine, games=games, model_player=1,
            opponent_temperature=0.35,
        )
        anchor_white = play_match(
            game, model_engine, anchor_engine, games=games, model_player=-1,
            opponent_temperature=0.35,
        )
        anchor_mean = (anchor_black + anchor_white) / 2.0

    return EvalResult(
        win_rate_random=(random_black + random_white) / 2.0,
        win_rate_heuristic=(heuristic_black + heuristic_white) / 2.0,
        win_rate_random_black=random_black,
        win_rate_random_white=random_white,
        win_rate_heuristic_black=heuristic_black,
        win_rate_heuristic_white=heuristic_white,
        win_rate_anchor=anchor_mean,
        win_rate_anchor_black=anchor_black,
        win_rate_anchor_white=anchor_white,
    )
