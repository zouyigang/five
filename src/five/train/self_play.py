from __future__ import annotations

from dataclasses import dataclass

import torch

from five.ai.encoder import encode_state
from five.ai.interfaces import AIEngine
from five.ai.interfaces import AnalysisResult
from five.common.config import RewardConfig
from five.core.game import GomokuGame
from five.core.move import Move
from five.storage.schemas import GameRecord, MoveRecord, MoveSummary, RewardDetail
from five.train.dataset import EpisodeBatch, Transition
from five.train.reward import compute_hybrid_reward_with_details, compute_outcome_tail_bonus


@dataclass(slots=True)
class SelfPlayResult:
    episode: EpisodeBatch
    record: GameRecord


def _apply_hybrid_rewards(
    episode: EpisodeBatch,
    winner: int,
    config: RewardConfig | None = None,
) -> list[tuple[float, list[RewardDetail]]]:
    if config is None:
        config = RewardConfig()

    reward_results = []
    total_transitions = len(episode.transitions)
    for index, transition in enumerate(episode.transitions):
        if transition.board_before is not None and transition.move is not None:
            result = compute_hybrid_reward_with_details(
                transition.board_before,
                transition.move,
                transition.player,
                winner,
                config,
            )
            details = [RewardDetail(amount=d.amount, reason=d.reason) for d in result.details]
            transition.reward = result.total_reward
        else:
            transition.reward = 0.0
            details = []

        plies_from_end = total_transitions - index - 1
        tail_bonus = compute_outcome_tail_bonus(transition.player, winner, plies_from_end, config)
        if tail_bonus is not None:
            transition.reward += tail_bonus.amount
            details.append(RewardDetail(amount=tail_bonus.amount, reason=tail_bonus.reason))

        reward_results.append((transition.reward, details))
        transition.done = False
    if episode.transitions:
        episode.transitions[-1].done = True
    return reward_results


def play_self_play_game(
    game: GomokuGame,
    black_engine: AIEngine,
    run_id: str,
    game_index: int,
    checkpoint_name: str | None = None,
    temperature: float = 1.0,
    reward_config: RewardConfig | None = None,
    white_engine: AIEngine | None = None,
    tracked_players: set[int] | None = None,
) -> SelfPlayResult:
    if white_engine is None:
        white_engine = black_engine
    if tracked_players is None:
        tracked_players = {1, -1}

    state = game.new_game()
    episode = EpisodeBatch()
    moves: list[MoveRecord] = []
    while not state.is_terminal:
        acting_player = state.current_player
        acting_engine = black_engine if acting_player == 1 else white_engine
        encoded = encode_state(state)
        analysis: AnalysisResult = acting_engine.select_move(state, temperature=temperature)
        move = analysis.action
        action_index = move.to_index(state.board.size)
        log_prob = float(torch.log(torch.tensor(max(analysis.action_probability, 1e-8))).item())
        board_before = state.board.copy()
        move_record_index = len(moves)
        if acting_player in tracked_players:
            episode.add(
                Transition(
                    state=encoded,
                    action=action_index,
                    old_log_prob=log_prob,
                    reward=0.0,
                    done=False,
                    value=analysis.value_estimate,
                    player=acting_player,
                    legal_mask=torch.from_numpy(state.legal_mask()),
                    board_before=board_before,
                    move=move,
                    move_record_index=move_record_index,
                )
            )
        moves.append(
            MoveRecord(
                move_index=move_record_index + 1,
                player=acting_player,
                row=move.row,
                col=move.col,
                action_probability=analysis.action_probability,
                value_before=analysis.value_estimate,
                legal_count=int(state.legal_mask().sum()),
                policy_topk=[
                    MoveSummary(
                        row=item.move.row,
                        col=item.move.col,
                        score=item.score,
                        visits=item.visits,
                        value=item.value,
                    )
                    for item in analysis.candidates
                ],
            )
        )
        state.apply_move(move)

    reward_results = _apply_hybrid_rewards(episode, state.winner, reward_config)
    for transition, (total_reward, details) in zip(episode.transitions, reward_results):
        if transition.move_record_index is None:
            continue
        if transition.move_record_index < len(moves):
            moves[transition.move_record_index].total_reward = total_reward
            moves[transition.move_record_index].reward_details = details
    result = "draw" if state.winner == 0 else "five_in_a_row"
    record = GameRecord(
        game_id=f"game_{game_index:06d}",
        run_id=run_id,
        board_size=state.board.size,
        win_length=state.board.win_length,
        winner=state.winner,
        total_moves=len(moves),
        black_player="selfplay_model",
        white_player="selfplay_model",
        result=result,
        model_checkpoint=checkpoint_name,
        moves=moves,
    )
    return SelfPlayResult(episode=episode, record=record)
