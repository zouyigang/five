# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` is the user-facing manual: install steps, every CLI flag, full config tables, and the reward-score table. This file covers what the README does not — the cross-file invariants you can only learn by reading several modules at once.

## Commands

The virtualenv lives at `.venv` (Windows, Python 3.14). Prefix commands with it rather than relying on an activated shell:

```powershell
& E:\PycharmProjects\five\.venv\Scripts\python.exe -m pytest -q
```

```powershell
& E:\PycharmProjects\five\.venv\Scripts\python.exe -m pytest tests/test_reward.py::test_attack_reward_for_open_three_is_positive -v
```

```powershell
& E:\PycharmProjects\five\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

There is no linter or formatter configured. Tests are plain pytest with no fixtures or conftest.

The five CLI entry points (`five-train`, `five-generate`, `five-pretrain`, `five-export-human-games`, `five-gui`) are registered by the editable install; see README for their flags. Training defaults to `--device cuda`; pass `--device cpu` explicitly on machines without a GPU.

## Core conventions

**Player encoding is `+1` = black, `-1` = white, `0` = empty**, uniformly across `Board.grid` (int8), `GameState.current_player`, `MoveRecord.player`, and every reward function's `player` argument. Side-to-move flips via `current_player *= -1` — but only when the game is not terminal, so `current_player` after a winning move still names the winner.

**`Move` ↔ flat action index** conversion (`move.to_index(board_size)`) is what bridges the board and the network's policy head. Board size is a runtime parameter (default 9), never a constant.

## Architecture

### The reward system is the center of gravity

`src/five/train/reward.py` is by far the most-iterated file, and `tests/test_reward.py` + `tests/test_threat_shapes.py` exist to pin its behavior. It is layered:

1. **`analyze_line` → `get_threat_info`** — shape detection radiating from *one* just-played stone. It permits at most **one** empty "jump" cell across both directions of a line, which is how 跳活三/跳四 are recognized. Board edges count as blocked, never as an open end.
2. **`scan_threat_instances` → `_scan_existing_threat_inventory`** — a whole-board scan of the *opponent's* standing threats, returned as **instances**, each carrying its stones, direction, and the cells that would defuse it. A threat is detected once per stone in it, so instances are deduplicated by `(direction, frozenset(stones))`; the count is therefore a count of threats, not of blocking cells. `ThreatInventory` counts instances per category — **not** a 0/1 flag, because binary made "block one of two open threes" score identically to ignoring both (`before` and `after` were both 1, so the difference was 0). Two fields stay binary on purpose: `immediate_win` (inherently yes/no) and the composites `double_four` / `four_three` / `double_three` (they describe a shape the position has, and counting them means nothing). The composites are derived from the instance counts, which is what makes two *plain* live threes register as a double three — under binary both collapsed into `open_three=1` and the sum never reached 2.
3. **`_extract_shape_features` → `_primary_category`** — a move is credited for its **single strongest** shape only, picked by `PRIMARY_CATEGORY_ORDER`; shapes never sum. Tests assert that a double-three does not also collect the open-three reward.
4. **`compute_process_reward_with_details`** — the orchestrator: attack + block − miss − missed-own-win + opening-position, then clipped to `max_process_reward`. `compute_hybrid_reward_with_details` wraps it and adds the terminal win/draw bonus.

**Every reward term must append a `RewardDetail(amount, reason)`.** These details are the only debugging surface — they are rendered live in the GUI 奖励检验 page and stored per-move in `GameRecord`. A term that adjusts the total without a detail is invisible and effectively unreviewable.

**Miss-penalty waivers are a deliberate two-tier hierarchy**, and the distinction is gomoku theory, not tuning:
- `my_strong_attack` (rush four and above) — an absolutely forcing move, single blocking point, opponent cannot counter-attack while responding. Waives miss penalties **entirely**.
- `my_counter_threat` (live three / jump live three) — only relatively forcing; the opponent can answer with a move that blocks *and* counter-attacks. Waives **partially**, scaled by `RewardConfig.counter_threat_waiver_scale` (default 0.5). `restricted_open_three` is deliberately excluded — it cannot become a live four, so it is not a real tempo threat.

**A waived rush four gets its attack reward discounted** by `rush_four_waiver_attack_scale` (default 0.3). The waiver alone is sound gomoku, but a full waiver *plus* full attack credit makes a pointless far-away rush four net positive — free money the policy can farm by spamming forcing moves to postpone defending forever. `_accumulate_miss_penalty` therefore returns `(penalty, waived_penalty)`, and the discount fires only when `waived_penalty > 0` **and** `_primary_category` is exactly `rush_four`. Open four / four-three / double-four / double-three are near-winning and keep full credit.

**Reward functions take the board *before* the move.** Helpers such as `_evaluate_move_features` mutate `board.grid` in place and restore it — never hand them a board that already contains the move.

### Config snapshotting on resume

`RewardConfig` and `ModelConfig` are serialized into every checkpoint's `config` field, and `apply_saved_config` merges them back when resuming via `--checkpoint`. Which side wins is deliberate:

- Keys in `RESUME_SKIP_KEYS` (epochs, device, lr, batch size, heuristic schedule…) — the **current** config wins, because they are the ones exposed as CLI flags.
- `reward` — the **current** `RewardConfig` wins by default, so editing a reward parameter takes effect on resume. `--reward-from-checkpoint` restores the old snapshot instead, for reproducing a historical run.
- Everything else (`gamma`, `entropy_coef`, `clip_epsilon`, `model`…) — the **checkpoint** wins. Changing those defaults and resuming still silently has no effect; add the key to `RESUME_SKIP_KEYS` if it should.

Either way `_log_reward_config_source` logs the effective source plus a field-level diff against the checkpoint, and `_load_checkpoint` rewrites `runs/<run_id>/config.json` after the merge — `create_run` writes that file *before* the checkpoint is loaded, so without the rewrite it would record the pre-merge config. Reward fields absent from an older checkpoint fall back to the dataclass default and are logged as such.

### Learning rate must be re-anchored on resume

`optimizer.load_state_dict` overwrites `lr` *and* `initial_lr` from the checkpoint, and `CosineAnnealingLR` advances recursively from the optimizer's **current** `lr` — constructing it with `last_epoch=N` does not rewrite `lr`, it only sets the position. So a resumed run silently inherits whatever learning rate the checkpoint's optimizer happened to hold.

`_restore_lr_schedule` fixes this by stamping `initial_lr`/`lr` from `config.learning_rate` and the closed-form `cosine_lr_at(...)` value at `last_epoch` before building the scheduler. Without it, resuming from a `five-pretrain` checkpoint starts PPO at the *bottom* of the behavior-cloning cosine (1e-3 × 0.01 = 1e-5, ~35× below the configured 3.5e-4) and stays there, and `--learning-rate` has no effect on any resume. `tests/test_trainer_config.py` pins all three cases: BC resume, override honored, and plain resume continuing its existing curve unchanged.

### Self-play and training loop

`play_self_play_game` drives one game through the `AIEngine` Protocol (`select_move` / `analyze` / `load_checkpoint`). `ModelAIEngine`, `HeuristicPlayer`, and `RandomPlayer` are freely interchangeable behind it — that is how the trainer mixes opponents without special-casing.

**`tracked_players` is what makes curriculum learning work.** When the model plays a heuristic or historical opponent, only the model's own side is appended to the `EpisodeBatch`; the opponent's moves still appear in the `GameRecord` but generate no training transitions. An episode therefore frequently contains only one color's moves.

Rewards are assigned after the game ends (`_apply_hybrid_rewards`), using each transition's stored `board_before`, then back-written into `GameRecord.moves` via `move_record_index`. The outcome tail bonus is skipped for any move flagged `missed_own_win`, so a positive result bonus can never cancel the "could have won and didn't" penalty.

**Only ~2 games per 1000 are written to disk** (`game_index % 1000 in (0, 1)`). Saving two rather than one is intentional: with the default 384 games/epoch, saving a single index would make `(game_index-1) % 384` always odd, so the model would always be white in the persisted opponent games and replay would never show it playing black.

### Two different "best" checkpoints

`best_epoch.py` defines two distinct scoring formulas, shared between the trainer and the GUI metrics panel so the saved checkpoint always matches the green line on the chart:
- `compute_best_epoch` → **`best.pt`** — for *playing*. Dominated by raw heuristic win rate.
- `compute_best_epoch_for_resume` → **`best_for_resume.pt`** — for *continuing training*. Scores training health instead: entropy inside `[1.0, 1.4]`, low value loss, non-declining win-rate trend, no anomalies.

Pick `best_for_resume.pt` for `five-train --checkpoint`, `best.pt` for human-vs-AI play.

### State encoding

`encode_state` produces 4 planes, and planes 0/1 are **relative to the side to move** (own stones, opponent stones), not to black/white. Any change to the plane layout or ordering silently invalidates every existing checkpoint *and* every generated `.pt` dataset — there is no version field to catch it.

### GUI

`FiveApp` constructs all six pages eagerly, then drives polling through `<<NotebookTabChanged>>` → each page's optional `set_active(bool)`. A page that polls must implement `set_active`, start its `after` loop only when active, and cancel the stored after-id when deactivated — otherwise every tab polls in the background forever.

**The GUI never launches training.** `five-generate` / `five-pretrain` / `five-train` run as separate terminal processes that write progress JSON; the corresponding pages only poll and render those files. Default paths are `<output>.progress.json`, `<output>.games.jsonl`, and `<output-dir>/pretrain.progress.json`.

`BAD_MOVE_REASONS` in `gui/bad_move_reasons.py` mirrors the reward function's penalty reasons for the replay page's human-annotation checklist. It is a hand-maintained list, not matched programmatically against `RewardDetail.reason`, so adding a penalty type means updating it manually. Annotations land in `MoveRecord.human_rating` / `human_bad_reasons` and are exported for training by `five-export-human-games`.

## Testing notes

Reward tests build positions with the `_place(Board(...), [(row, col, player), ...])` helper, which writes stones straight onto the grid and bypasses turn alternation — position setup does not need to be a legal move sequence. Pass an explicit `RewardConfig(...)` when a test depends on a specific weight, and set `opening_position_horizon=0` to suppress opening position shaping that would otherwise pollute the total.

GUI widget tests instantiate widgets via `BoardCanvas.__new__(BoardCanvas)` and stub out Tk methods, so handler logic is testable without a display.
