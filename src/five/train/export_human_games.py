"""Export human-annotated games (e.g. from 人机对弈) to .pt for training.

Only includes games that have at least one move with human_rating set.
Output format matches imitation_data .pt plus a "human_ratings" tensor.

Usage:
    five-export-human-games --run-dir runs/ppo_gomoku_xxx --output data/human_marked.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from five.ai.encoder import encode_state
from five.core.move import Move
from five.core.state import GameState
from five.storage.game_store import GameStore


def export_human_games(
    run_dir: Path,
    output_path: Path,
) -> dict[str, torch.Tensor]:
    """Load games from run_dir/games, keep only those with human_rating, export to .pt."""
    games_dir = run_dir / "games"
    if not games_dir.is_dir():
        return {}
    store = GameStore(games_dir)
    paths = store.list_game_paths()

    states: list[torch.Tensor] = []
    actions: list[int] = []
    legal_masks: list[torch.Tensor] = []
    values: list[float] = []
    human_ratings: list[float] = []

    board_size = 9
    for path in paths:
        record = store.load(path)
        if not any(m.human_rating is not None for m in record.moves):
            continue
        board_size = record.board_size
        state = GameState.new(
            board_size=record.board_size,
            win_length=record.win_length,
        )
        winner = record.winner
        game_states: list[torch.Tensor] = []
        game_actions: list[int] = []
        game_masks: list[torch.Tensor] = []
        game_values: list[float] = []
        game_ratings: list[float] = []
        for move_rec in record.moves:
            move = Move(row=move_rec.row, col=move_rec.col)
            if not state.board.is_legal(move):
                break
            encoded = encode_state(state)
            legal = torch.from_numpy(state.legal_mask()).float()
            action_idx = move.to_index(record.board_size)
            if winner == 0:
                val = 0.0
            elif move_rec.player == winner:
                val = 1.0
            else:
                val = -1.0
            rating = move_rec.human_rating if move_rec.human_rating is not None else float("nan")
            game_states.append(encoded)
            game_actions.append(action_idx)
            game_masks.append(legal)
            game_values.append(val)
            game_ratings.append(rating)
            state.apply_move(move)
        else:
            states.extend(game_states)
            actions.extend(game_actions)
            legal_masks.extend(game_masks)
            values.extend(game_values)
            human_ratings.extend(game_ratings)

    if not states:
        return {}

    return {
        "states": torch.stack(states),
        "actions": torch.tensor(actions, dtype=torch.long),
        "legal_masks": torch.stack(legal_masks),
        "values": torch.tensor(values, dtype=torch.float32),
        "human_ratings": torch.tensor(human_ratings, dtype=torch.float32),
        "board_size": board_size,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export human-annotated games from a run's game_store to .pt for training.",
    )
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the run directory (e.g. runs/ppo_gomoku_xxx) containing a 'games' subdir.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/human_marked.pt",
        help="Output .pt file path (default: data/human_marked.pt)",
    )
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    output_path = Path(args.output)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")
    data = export_human_games(run_dir, output_path)
    if not data:
        print("No games with human_rating found; nothing to export.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    n = data["states"].size(0)
    print(f"Exported {n} samples to {output_path} (board_size={data['board_size']})")


if __name__ == "__main__":
    main()
