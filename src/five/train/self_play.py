from __future__ import annotations

from dataclasses import dataclass

import torch

from five.ai.encoder import encode_state
from five.ai.interfaces import AIEngine
from five.ai.interfaces import AnalysisResult, select_moves_batched
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
        missed_own_win = False
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
            missed_own_win = result.missed_own_win
        else:
            transition.reward = 0.0
            details = []

        if not missed_own_win:
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


@dataclass(slots=True)
class SelfPlaySpec:
    """一局待进行的对局配置。"""

    game_index: int
    black_engine: AIEngine
    white_engine: AIEngine | None = None
    tracked_players: set[int] | None = None
    black_player: str = "selfplay_model"
    white_player: str = "selfplay_model"


class _GameRunner:
    """单局的分步状态机：每次只推进一手，便于多局并行凑批。"""

    __slots__ = ("spec", "state", "episode", "moves", "tracked_players", "white_engine")

    def __init__(self, game: GomokuGame, spec: SelfPlaySpec) -> None:
        self.spec = spec
        self.state = game.new_game()
        self.episode = EpisodeBatch()
        self.moves: list[MoveRecord] = []
        self.white_engine = spec.white_engine if spec.white_engine is not None else spec.black_engine
        self.tracked_players = spec.tracked_players if spec.tracked_players is not None else {1, -1}

    @property
    def finished(self) -> bool:
        return self.state.is_terminal

    def acting_engine(self) -> AIEngine:
        return self.spec.black_engine if self.state.current_player == 1 else self.white_engine

    def apply(self, analysis: AnalysisResult) -> None:
        state = self.state
        acting_player = state.current_player
        encoded = encode_state(state)
        move = analysis.action
        action_index = move.to_index(state.board.size)
        log_prob = float(torch.log(torch.tensor(max(analysis.action_probability, 1e-8))).item())
        board_before = state.board.copy()
        move_record_index = len(self.moves)
        if acting_player in self.tracked_players:
            self.episode.add(
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
        self.moves.append(
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

    def build_result(
        self,
        run_id: str,
        checkpoint_name: str | None,
        reward_config: RewardConfig | None,
    ) -> SelfPlayResult:
        state = self.state
        reward_results = _apply_hybrid_rewards(self.episode, state.winner, reward_config)
        for transition, (total_reward, details) in zip(self.episode.transitions, reward_results):
            if transition.move_record_index is None:
                continue
            if transition.move_record_index < len(self.moves):
                self.moves[transition.move_record_index].total_reward = total_reward
                self.moves[transition.move_record_index].reward_details = details
        record = GameRecord(
            game_id=f"game_{self.spec.game_index:06d}",
            run_id=run_id,
            board_size=state.board.size,
            win_length=state.board.win_length,
            winner=state.winner,
            total_moves=len(self.moves),
            black_player=self.spec.black_player,
            white_player=self.spec.white_player,
            result="draw" if state.winner == 0 else "five_in_a_row",
            model_checkpoint=checkpoint_name,
            moves=self.moves,
        )
        return SelfPlayResult(episode=self.episode, record=record)


def play_self_play_games(
    game: GomokuGame,
    specs: list[SelfPlaySpec],
    run_id: str,
    checkpoint_name: str | None = None,
    temperature: float = 1.0,
    reward_config: RewardConfig | None = None,
) -> list[SelfPlayResult]:
    """并行推进一批对局，把同一引擎在同一时刻待决策的局面凑成一次前向。

    按引擎对象身份分组：自博弈局的双方、以及对手局中模型的一方，共用同一个
    ModelAIEngine 实例，因而会合并进同一批；启发式/历史对手各自成组。
    各局长度不同，走完的局自然退出，不影响其余局继续凑批。
    """
    runners = [_GameRunner(game, spec) for spec in specs]

    while True:
        active = [runner for runner in runners if not runner.finished]
        if not active:
            break
        groups: dict[int, list[_GameRunner]] = {}
        for runner in active:
            groups.setdefault(id(runner.acting_engine()), []).append(runner)
        for group in groups.values():
            engine = group[0].acting_engine()
            analyses = select_moves_batched(
                engine, [runner.state for runner in group], temperature=temperature
            )
            for runner, analysis in zip(group, analyses):
                runner.apply(analysis)

    return [runner.build_result(run_id, checkpoint_name, reward_config) for runner in runners]


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
    black_player: str | None = None,
    white_player: str | None = None,
) -> SelfPlayResult:
    """单局版本；走的是与批量版完全相同的代码路径，避免两套逻辑漂移。"""
    spec = SelfPlaySpec(
        game_index=game_index,
        black_engine=black_engine,
        white_engine=white_engine,
        tracked_players=tracked_players,
        black_player=black_player if black_player is not None else "selfplay_model",
        white_player=white_player if white_player is not None else "selfplay_model",
    )
    return play_self_play_games(
        game=game,
        specs=[spec],
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        temperature=temperature,
        reward_config=reward_config,
    )[0]
