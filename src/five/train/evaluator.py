from __future__ import annotations

from dataclasses import dataclass

from five.ai.inference import ModelAIEngine
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


def play_match(
    game: GomokuGame,
    model_engine: ModelAIEngine,
    opponent,
    games: int,
    model_player: int,
) -> float:
    wins = 0
    for _ in range(games):
        state: GameState = game.new_game()
        while not state.is_terminal:
            if state.current_player == model_player:
                analysis = model_engine.select_move(state, temperature=0.0)
            else:
                analysis = opponent.select_move(state)
            state.apply_move(analysis.action)
        if state.winner == model_player:
            wins += 1
    return wins / max(games, 1)


def evaluate_policy(game: GomokuGame, model_engine: ModelAIEngine, games: int) -> EvalResult:
    random_black = play_match(game, model_engine, RandomPlayer(), games=games, model_player=1)
    random_white = play_match(game, model_engine, RandomPlayer(), games=games, model_player=-1)
    heuristic_black = play_match(game, model_engine, HeuristicPlayer(), games=games, model_player=1)
    heuristic_white = play_match(game, model_engine, HeuristicPlayer(), games=games, model_player=-1)
    return EvalResult(
        win_rate_random=(random_black + random_white) / 2.0,
        win_rate_heuristic=(heuristic_black + heuristic_white) / 2.0,
        win_rate_random_black=random_black,
        win_rate_random_white=random_white,
        win_rate_heuristic_black=heuristic_black,
        win_rate_heuristic_white=heuristic_white,
    )
