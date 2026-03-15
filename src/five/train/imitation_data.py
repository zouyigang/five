"""Generate imitation-learning dataset from heuristic self-play.

Usage:
    five-generate --games 20000 --board-size 9 --output data/heuristic_20k.pt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Callable

import torch

from five.ai.encoder import encode_state
from five.ai.players import HeuristicPlayer
from five.common.config import GenerateConfig
from five.common.logging import configure_logging, get_logger
from five.core.game import GomokuGame
from five.core.state import GameState

DEFAULT_GENERATE = GenerateConfig()

LOGGER = get_logger(__name__)


def generate_dataset(
    games: int,
    board_size: int = 9,
    win_length: int = 5,
    on_progress: Callable[[int, int, int, int, int, int, int, int], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
    progress_file: str | None = None,
    games_detail_file: str | None = None,
    output_path: str = "",
) -> dict[str, torch.Tensor]:
    """Play *games* rounds of heuristic-vs-heuristic and collect transitions.

    If on_progress is given, it is called after each game with:
      (games_done, total_games, black_wins, white_wins, draws, total_samples, last_winner, last_moves).
    If stop_check is given and returns True, generation stops and returns data collected so far.

    Returns a dict ready for ``torch.save``::

        {
            "states":      (N, 4, board_size, board_size)  float32
            "actions":     (N,)                            long
            "legal_masks": (N, board_size*board_size)       float32
            "values":      (N,)                            float32   (+1/-1/0)
        }
    """
    game = GomokuGame(board_size=board_size, win_length=win_length)
    heuristic = HeuristicPlayer()

    states: list[torch.Tensor] = []
    actions: list[int] = []
    legal_masks: list[torch.Tensor] = []
    players: list[int] = []

    black_wins = white_wins = draws = 0
    recent_games: list[dict] = []
    if progress_file:
        Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
    if games_detail_file:
        Path(games_detail_file).parent.mkdir(parents=True, exist_ok=True)

    for g in range(1, games + 1):
        state: GameState = game.new_game()
        episode_indices_start = len(states)

        while not state.is_terminal:
            encoded = encode_state(state)
            legal = torch.from_numpy(state.legal_mask()).float()
            # 使用小 temperature 引入随机性，避免两方完全对称导致全部和棋（81 手满盘）
            analysis = heuristic.select_move(state, temperature=0.4)
            action_idx = analysis.action.to_index(board_size)

            states.append(encoded)
            actions.append(action_idx)
            legal_masks.append(legal)
            players.append(state.current_player)

            state.apply_move(analysis.action)

        winner = state.winner
        if winner == 1:
            black_wins += 1
        elif winner == -1:
            white_wins += 1
        else:
            draws += 1

        for idx in range(episode_indices_start, len(states)):
            player = players[idx]
            if winner == 0:
                players[idx] = 0
            elif player == winner:
                players[idx] = 1
            else:
                players[idx] = -1

        num_moves = len(states) - episode_indices_start
        game_actions = actions[episode_indices_start : len(actions)]

        if on_progress is not None:
            on_progress(g, games, black_wins, white_wins, draws, len(states), winner, num_moves)
            time.sleep(0.001)  # 让出 CPU，避免主线程（GUI）卡死

        recent_games.append({"game": g, "winner": winner, "moves": num_moves})
        if len(recent_games) > 500:
            recent_games.pop(0)
        if progress_file:
            payload = {
                "running": True,
                "games_done": g,
                "total_games": games,
                "black_wins": black_wins,
                "white_wins": white_wins,
                "draws": draws,
                "samples": len(states),
                "output_path": output_path,
                "board_size": board_size,
                "recent_games": recent_games,
            }
            Path(progress_file).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if games_detail_file:
            line = json.dumps(
                {"game": g, "winner": winner, "moves": num_moves, "actions": game_actions},
                ensure_ascii=False,
            ) + "\n"
            with open(games_detail_file, "a", encoding="utf-8") as f:
                f.write(line)

        if stop_check is not None and stop_check():
            LOGGER.info("Generation stopped by user at %d/%d games", g, games)
            break
        if g % 2000 == 0 or g == games:
            LOGGER.info(
                "progress %d/%d  black_wins=%d  white_wins=%d  draws=%d  samples=%d",
                g, games, black_wins, white_wins, draws, len(states),
            )

    if progress_file:
        games_done = black_wins + white_wins + draws
        payload = {
            "running": False,
            "games_done": games_done,
            "total_games": games,
            "black_wins": black_wins,
            "white_wins": white_wins,
            "draws": draws,
            "samples": len(states),
            "output_path": output_path,
            "board_size": board_size,
            "recent_games": recent_games,
        }
        Path(progress_file).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    return {
        "states": torch.stack(states),
        "actions": torch.tensor(actions, dtype=torch.long),
        "legal_masks": torch.stack(legal_masks),
        "values": torch.tensor(players, dtype=torch.float32),
        "board_size": torch.tensor(board_size),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    c = DEFAULT_GENERATE
    p = argparse.ArgumentParser(description="Generate heuristic self-play dataset for imitation learning.")
    p.add_argument("--games", type=int, default=c.games, help=f"Number of self-play games (default: {c.games})")
    p.add_argument("--board-size", type=int, default=c.board_size)
    p.add_argument("--win-length", type=int, default=c.win_length)
    p.add_argument("--output", type=str, default=c.output, help=f"Output .pt file path (default: {c.output})")
    p.add_argument(
        "--progress-file",
        type=str,
        default="",
        help="JSON file for GUI progress (default: <output>.progress.json)",
    )
    p.add_argument(
        "--games-detail-file",
        type=str,
        default="",
        help="JSONL file for per-game details (default: <output>.games.jsonl)",
    )
    p.add_argument("--seed", type=int, default=c.seed)
    return p


def main() -> None:
    configure_logging()
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    out = Path(args.output)
    progress_file = args.progress_file or str(out.with_suffix(".progress.json"))
    games_detail_file = args.games_detail_file or str(out.with_suffix(".games.jsonl"))
    LOGGER.info("Generating %d heuristic self-play games (board=%d) ...", args.games, args.board_size)
    dataset = generate_dataset(
        args.games,
        board_size=args.board_size,
        win_length=args.win_length,
        progress_file=progress_file,
        games_detail_file=games_detail_file,
        output_path=args.output,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out)
    LOGGER.info("Dataset saved to %s  (%d samples)", out, dataset["states"].size(0))


if __name__ == "__main__":
    main()
