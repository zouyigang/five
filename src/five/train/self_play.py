from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass

import torch

from five.ai.encoder import encode_state
from five.ai.interfaces import AIEngine
from five.ai.interfaces import AnalysisResult, select_moves_batched
from five.common.config import RewardConfig
from five.core.board import Board
from five.core.game import GomokuGame
from five.core.move import Move
from five.storage.schemas import GameRecord, MoveRecord, MoveSummary, RewardDetail
from five.train.dataset import EpisodeBatch, Transition
from five.train.reward import compute_hybrid_reward_with_details, compute_outcome_tail_bonus


@dataclass(slots=True)
class SelfPlayResult:
    episode: EpisodeBatch
    record: GameRecord


@dataclass(slots=True)
class RewardTask:
    """一局奖励计算所需的最小载荷：可 pickle，且不含张量。

    奖励计算占自博弈约 90% 的耗时且是纯 CPU 单线程，把它派到进程池是唯一能吃满
    多核的办法。这里只带 9x9 int8 网格和落子信息，不带 4x9x9 状态张量与合法掩码，
    序列化开销可以忽略。
    """

    winner: int
    board_size: int
    win_length: int
    config: RewardConfig
    # 每个 transition 一项：(落子前网格, row, col, player)；缺少局面时网格为 None
    steps: list[tuple[object, int, int, int]]


def compute_episode_rewards(task: RewardTask) -> list[tuple[float, list[RewardDetail]]]:
    """纯函数：算出一局中每个 transition 的 (总奖励, 明细)。

    进程池 worker 与主进程共用这一份实现，两条路径不会漂移。
    """
    results: list[tuple[float, list[RewardDetail]]] = []
    total = len(task.steps)

    # 输家的最后一手：与获胜手拿 +final_win_reward 对称，这里扣 -final_loss_penalty。
    # 必须在 episode 层判定——单步的 (board, move, player, winner) 看不出「这是输家的
    # 最后一次决策」，因为输家永远不是走出终局那一手的人。
    loser_last_index: int | None = None
    if task.winner != 0 and task.config.final_loss_penalty > 0.0:
        for index in reversed(range(total)):
            if task.steps[index][3] != task.winner:
                loser_last_index = index
                break

    for index, (grid, row, col, player) in enumerate(task.steps):
        missed_own_win = False
        if grid is not None:
            board = Board(size=task.board_size, win_length=task.win_length)
            board.grid = grid
            result = compute_hybrid_reward_with_details(
                board, Move(row, col), player, task.winner, task.config
            )
            reward = result.total_reward
            details = [RewardDetail(amount=d.amount, reason=d.reason) for d in result.details]
            missed_own_win = result.missed_own_win
        else:
            reward = 0.0
            details = []

        if index == loser_last_index:
            penalty = -task.config.final_loss_penalty
            reward += penalty
            details.append(RewardDetail(amount=penalty, reason="终局失败惩罚"))

        if not missed_own_win:
            tail_bonus = compute_outcome_tail_bonus(player, task.winner, total - index - 1, task.config)
            if tail_bonus is not None:
                reward += tail_bonus.amount
                details.append(RewardDetail(amount=tail_bonus.amount, reason=tail_bonus.reason))

        results.append((reward, details))
    return results


def build_reward_task(
    episode: EpisodeBatch,
    winner: int,
    board_size: int,
    win_length: int,
    config: RewardConfig,
) -> RewardTask:
    steps: list[tuple[object, int, int, int]] = []
    for transition in episode.transitions:
        if transition.board_before is not None and transition.move is not None:
            steps.append(
                (transition.board_before.grid, transition.move.row, transition.move.col, transition.player)
            )
        else:
            steps.append((None, 0, 0, transition.player))
    return RewardTask(
        winner=winner,
        board_size=board_size,
        win_length=win_length,
        config=config,
        steps=steps,
    )


def _write_back_rewards(
    episode: EpisodeBatch,
    reward_results: list[tuple[float, list[RewardDetail]]],
) -> None:
    for transition, (reward, _details) in zip(episode.transitions, reward_results):
        transition.reward = reward
        transition.done = False
    if episode.transitions:
        episode.transitions[-1].done = True


def _apply_hybrid_rewards(
    episode: EpisodeBatch,
    winner: int,
    config: RewardConfig | None = None,
) -> list[tuple[float, list[RewardDetail]]]:
    if config is None:
        config = RewardConfig()
    board_size = 0
    win_length = 0
    for transition in episode.transitions:
        if transition.board_before is not None:
            board_size = transition.board_before.size
            win_length = transition.board_before.win_length
            break
    task = build_reward_task(episode, winner, board_size, win_length, config)
    reward_results = compute_episode_rewards(task)
    _write_back_rewards(episode, reward_results)
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

    def reward_task(self, reward_config: RewardConfig) -> RewardTask:
        return build_reward_task(
            self.episode,
            self.state.winner,
            self.state.board.size,
            self.state.board.win_length,
            reward_config,
        )

    def build_result(
        self,
        run_id: str,
        checkpoint_name: str | None,
        reward_config: RewardConfig | None,
        reward_results: list[tuple[float, list[RewardDetail]]] | None = None,
    ) -> SelfPlayResult:
        state = self.state
        if reward_results is None:
            reward_results = _apply_hybrid_rewards(self.episode, state.winner, reward_config)
        else:
            _write_back_rewards(self.episode, reward_results)
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
    reward_executor: "Executor | None" = None,
) -> list[SelfPlayResult]:
    """并行推进一批对局，把同一引擎在同一时刻待决策的局面凑成一次前向。

    按引擎对象身份分组：自博弈局的双方、以及对手局中模型的一方，共用同一个
    ModelAIEngine 实例，因而会合并进同一批；启发式/历史对手各自成组。
    各局长度不同，走完的局自然退出，不影响其余局继续凑批。

    `reward_executor` 非空时，各局的奖励计算派发到进程池。奖励是纯 CPU 单线程且
    占自博弈约 90% 的耗时，是唯一能把利用率从 1 核铺到多核的环节；对局之间完全
    独立，因此可以整局为粒度并行。
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

    if reward_executor is None:
        return [runner.build_result(run_id, checkpoint_name, reward_config) for runner in runners]

    config = reward_config if reward_config is not None else RewardConfig()
    tasks = [runner.reward_task(config) for runner in runners]
    # 每局约 100ms，chunksize=1 的调度开销可忽略，换来的是各 worker 负载均衡
    # （对局长度差异很大，成块分发会让拿到长局的 worker 拖住整批）。
    all_rewards = list(reward_executor.map(compute_episode_rewards, tasks, chunksize=1))
    return [
        runner.build_result(run_id, checkpoint_name, config, reward_results=rewards)
        for runner, rewards in zip(runners, all_rewards)
    ]


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
