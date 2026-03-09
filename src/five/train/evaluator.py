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


def play_match(game: GomokuGame, model_engine: ModelAIEngine, opponent, games: int) -> float:
    wins = 0
    for _ in range(games):
        state: GameState = game.new_game()
        while not state.is_terminal:
            if state.current_player == 1:
                analysis = model_engine.select_move(state, temperature=0.0)
            else:
                analysis = opponent.select_move(state)
            state.apply_move(analysis.action)
        if state.winner == 1:
            wins += 1
    return wins / max(games, 1)


def evaluate_policy(game: GomokuGame, model_engine: ModelAIEngine, games: int) -> EvalResult:
    return EvalResult(
        win_rate_random=play_match(game, model_engine, RandomPlayer(), games=games),
        win_rate_heuristic=play_match(game, model_engine, HeuristicPlayer(), games=games),
    )
