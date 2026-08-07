from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import torch
from torch import nn

from five.ai.inference import ModelAIEngine
from five.ai.model import PolicyValueNet
from five.ai.players import HeuristicPlayer
from five.common.config import ModelConfig, RewardConfig, TrainingConfig
from five.common.logging import configure_logging, get_logger
from five.common.utils import set_seed, write_json
from five.core.game import GomokuGame
from five.storage.schemas import MetricRecord, ModelRecord
from five.train.dataset import EpisodeBatch
from five.train.evaluator import evaluate_policy
from five.train.best_epoch import compute_best_epoch, compute_best_epoch_for_resume
from five.train.run_manager import RunArtifacts, create_run
from five.train.self_play import SelfPlayResult, SelfPlaySpec, play_self_play_games


LOGGER = get_logger(__name__)


# 续训时保留用户当前设置、不被 checkpoint 覆盖的键。
RESUME_SKIP_KEYS = frozenset(
    {
        "epochs",  # 延长总轮数
        "checkpoint_every",  # checkpoint 保存间隔
        "device",  # 切换 GPU/CPU
        "run_name",  # 新 run 名称
        "runs_dir",  # 输出目录
        "learning_rate",  # --learning-rate 微调
        "batch_size",  # --batch-size
        "self_play_games_per_epoch",  # --games-per-epoch
        "eval_games",  # 评估局数
        "eval_heuristic_temperature",  # 启发式评估温度，影响胜率曲线粒度
        "heuristic_opponent_max_prob",
        "heuristic_start_fraction",
        "heuristic_ramp_fraction",
    }
)


def resolve_opponent_kind(
    roll: float,
    heuristic_prob: float,
    historical_prob: float,
    *,
    has_historical: bool,
) -> str:
    """按一次 [0,1) 采样决定本局对手：'heuristic' / 'historical' / 'self'。

    `historical_prob` 是**非启发式对局中**历史对手所占的比例，不是全局概率。

    按全局概率写会失效：heuristic_prob=0.70 与 historical_prob=0.40 的累计阈值是
    1.10，超过 roll 的上界 1.0，于是所有非启发式对局全部落给历史对手——与当前策略
    的自博弈占比恒为 0，且 historical_prob 取任何 >=0.30 的值都毫无区别。
    改成「剩余部分的比例」后三档恒定和为 1，自博弈始终有份额；在 heuristic_prob=0
    时与原语义完全一致（历史 0.4 / 自博弈 0.6）。
    """
    heuristic_prob = min(max(heuristic_prob, 0.0), 1.0)
    if roll < heuristic_prob:
        return "heuristic"
    if not has_historical:
        return "self"
    remaining = 1.0 - heuristic_prob
    historical_share = remaining * min(max(historical_prob, 0.0), 1.0)
    if roll < heuristic_prob + historical_share:
        return "historical"
    return "self"


def cosine_lr_at(base_lr: float, eta_min: float, t_max: int, position: int) -> float:
    """CosineAnnealingLR 在第 position 步的闭式学习率（position=0 即 base_lr）。"""
    t_max = max(t_max, 1)
    position = min(max(position, 0), t_max)
    return eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * position / t_max)) / 2


def _values_equal(left, right) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-9
    return left == right


def diff_reward_config(saved_reward: dict, reward: RewardConfig) -> tuple[list[str], list[str]]:
    """返回 (与 checkpoint 取值不同的字段, checkpoint 中尚不存在的新增字段)。"""
    changed: list[str] = []
    added: list[str] = []
    for key in getattr(RewardConfig, "__dataclass_fields__", {}):
        current = getattr(reward, key)
        if key not in saved_reward:
            added.append(f"{key}={current}")
        elif not _values_equal(saved_reward[key], current):
            changed.append(f"{key}: {saved_reward[key]} -> {current}")
    return changed, added


def apply_saved_config(
    config: TrainingConfig,
    saved: dict,
    *,
    reward_from_checkpoint: bool = False,
) -> None:
    """把 checkpoint 中的配置合并进 config，RESUME_SKIP_KEYS 里的键保留用户当前设置。

    reward 默认**不**从 checkpoint 恢复：奖励参数是最常迭代的部分，若跟随 checkpoint，
    改完 RewardConfig 再 --checkpoint 续训会静默不生效。需要复现旧 run 时传
    reward_from_checkpoint=True。
    """
    skip_keys = set(RESUME_SKIP_KEYS)
    if not reward_from_checkpoint:
        skip_keys.add("reward")

    for key, value in saved.items():
        if key in skip_keys or not hasattr(config, key):
            continue
        if key == "model" and isinstance(value, dict):
            subset = {k: v for k, v in value.items() if k in getattr(ModelConfig, "__dataclass_fields__", {})}
            setattr(config, key, ModelConfig(**subset))
        elif key == "reward" and isinstance(value, dict):
            subset = {k: v for k, v in value.items() if k in getattr(RewardConfig, "__dataclass_fields__", {})}
            setattr(config, key, RewardConfig(**subset))
        else:
            setattr(config, key, value)


@dataclass(slots=True)
class TrainingBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    legal_masks: torch.Tensor
    old_values: torch.Tensor
    raw_return_mean: float = 0.0
    raw_return_std: float = 0.0
    raw_return_abs_max: float = 0.0
    # 锚点策略在这批局面上的对数概率（T=1，已按合法点掩码）。锚点是冻结的，
    # 整批只需前向一次，不必在每个 minibatch 里重算。
    anchor_log_probs: torch.Tensor | None = None


@dataclass(slots=True)
class _PositionMetricTotals:
    opening_moves: int = 0
    opening_edges: int = 0
    opening_corners: int = 0
    opening_centers: int = 0
    topk_candidates: int = 0
    topk_edges: int = 0

    def merge(self, other: "_PositionMetricTotals") -> None:
        self.opening_moves += other.opening_moves
        self.opening_edges += other.opening_edges
        self.opening_corners += other.opening_corners
        self.opening_centers += other.opening_centers
        self.topk_candidates += other.topk_candidates
        self.topk_edges += other.topk_edges

    @property
    def opening_edge_rate(self) -> float:
        return self.opening_edges / max(self.opening_moves, 1)

    @property
    def opening_corner_rate(self) -> float:
        return self.opening_corners / max(self.opening_moves, 1)

    @property
    def opening_center_rate(self) -> float:
        return self.opening_centers / max(self.opening_moves, 1)

    @property
    def policy_topk_edge_rate(self) -> float:
        return self.topk_edges / max(self.topk_candidates, 1)


class PPOTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        checkpoint_path: str | None = None,
        *,
        reward_from_checkpoint: bool = False,
    ) -> None:
        self.config = config
        self.reward_from_checkpoint = reward_from_checkpoint
        self.device = torch.device(config.device)
        self.game = GomokuGame(board_size=config.board_size, win_length=config.win_length)
        self.model = PolicyValueNet(
            board_size=config.board_size,
            channels=config.model.channels,
            blocks=config.model.blocks,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.lr_min,
        )
        self.engine = ModelAIEngine(self.model, device=config.device)
        self.artifacts: RunArtifacts = create_run(config)
        self.historical_opponent_snapshots: list[dict[str, torch.Tensor]] = []
        self._epoch_counter: int = 0
        self._baseline: dict | None = None

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # 锚点对手：本次 run 起点策略的冻结副本。必须在 checkpoint 加载之后再取，
        # 否则锚定的是随机初始化的网络而非实际起点。它从不参与训练，也从不更新。
        self.anchor_engine = self._build_engine_from_state_dict(self._clone_model_state())
        self._reward_executor: ProcessPoolExecutor | None = None

    def _get_reward_executor(self) -> ProcessPoolExecutor | None:
        """惰性创建奖励计算进程池，整个 run 复用一个（Windows 下进程启动很慢）。"""
        workers = self.config.reward_workers
        if workers == 1:
            return None
        if workers <= 0:
            workers = max(1, (os.cpu_count() or 2) - 2)
        if workers == 1:
            return None
        if self._reward_executor is None:
            # 显式用 spawn：Windows 只有 spawn，显式指定可让 Linux 行为一致，
            # 也避免 fork 出来的子进程继承 CUDA 上下文。
            self._reward_executor = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
            LOGGER.info("Reward computation pool started with %d workers", workers)
        return self._reward_executor

    def _shutdown_reward_executor(self) -> None:
        if self._reward_executor is not None:
            self._reward_executor.shutdown(wait=True)
            self._reward_executor = None

    def _discard_broken_reward_executor(self) -> None:
        """池一旦破损就整个丢弃，下一轮重建。

        BrokenProcessPool 之后该 executor 永久不可用，继续复用会让每一轮都退回内联
        计算（慢 6 倍）。self_play 侧已经保证这一批仍会算出正确结果，这里只负责让
        后续轮次能恢复并行。
        """
        if self._reward_executor is None:
            return
        if getattr(self._reward_executor, "_broken", False):
            LOGGER.warning("Reward worker pool is broken; discarding it so the next epoch rebuilds.")
            try:
                self._reward_executor.shutdown(wait=False)
            except Exception:
                pass
            self._reward_executor = None

    def _get_heuristic_prob(self, epoch: int) -> float:
        start_epoch = int(self.config.epochs * self.config.heuristic_start_fraction)
        ramp_epoch = int(self.config.epochs * self.config.heuristic_ramp_fraction)
        if epoch < start_epoch:
            return 0.0
        if ramp_epoch <= start_epoch:
            return self.config.heuristic_opponent_max_prob
        progress = min((epoch - start_epoch) / (ramp_epoch - start_epoch), 1.0)
        return self.config.heuristic_opponent_max_prob * progress

    def _get_temperature(self, epoch: int) -> float:
        anneal_end = int(self.config.epochs * self.config.temperature_anneal_fraction)
        if epoch >= anneal_end:
            return self.config.temperature_min
        progress = epoch / max(anneal_end, 1)
        return self.config.temperature_init - (self.config.temperature_init - self.config.temperature_min) * progress

    def train(self) -> None:
        try:
            self._train()
        finally:
            # 无论正常结束、异常还是 Ctrl-C，都要收掉进程池，避免留下孤儿进程。
            self._shutdown_reward_executor()

    def _train(self) -> None:
        start_epoch = getattr(self, "_start_epoch", 1)
        for epoch in range(start_epoch, self.config.epochs + 1):
            self._epoch_counter = epoch
            self.model.eval()
            temperature = self._get_temperature(epoch)
            heuristic_prob = self._get_heuristic_prob(epoch)
            historical_opponent = self._sample_historical_opponent()
            heuristic_opponent = HeuristicPlayer()

            batches: list[EpisodeBatch] = []
            total_game_length = 0
            position_metrics = _PositionMetricTotals()
            epoch_opponent_counts = {"heuristic": 0, "historical": 0, "self": 0}
            specs: list[SelfPlaySpec] = []
            for game_offset in range(self.config.self_play_games_per_epoch):
                game_index = (epoch - 1) * self.config.self_play_games_per_epoch + game_offset + 1
                black_engine = self.engine
                white_engine = self.engine
                tracked_players: set[int] | None = None
                opponent_kind = resolve_opponent_kind(
                    random.random(),
                    heuristic_prob,
                    self.config.historical_opponent_prob,
                    has_historical=historical_opponent is not None,
                )
                # 自博弈局不设 tracked_players（双方都记入训练）；对手局只记模型一方。
                if opponent_kind != "self":
                    opponent = heuristic_opponent if opponent_kind == "heuristic" else historical_opponent
                    if game_offset % 2 == 0:
                        white_engine = opponent
                        tracked_players = {1}
                    else:
                        black_engine = opponent
                        tracked_players = {-1}
                epoch_opponent_counts[opponent_kind] += 1
                specs.append(
                    SelfPlaySpec(
                        game_index=game_index,
                        black_engine=black_engine,
                        white_engine=white_engine,
                        tracked_players=tracked_players,
                        black_player="model" if black_engine is self.engine else opponent_kind,
                        white_player="model" if white_engine is self.engine else opponent_kind,
                    )
                )

            # 分批并行推进：同一时刻各局待决策的局面凑成一次前向，避免 batch=1 空转 GPU。
            for chunk_start in range(0, len(specs), self.config.self_play_batch_games):
                chunk = specs[chunk_start : chunk_start + self.config.self_play_batch_games]
                for result in play_self_play_games(
                    game=self.game,
                    specs=chunk,
                    run_id=self.artifacts.run_id,
                    temperature=temperature,
                    reward_config=self.config.reward,
                    reward_executor=self._get_reward_executor(),
                ):
                    batches.append(result.episode)
                    total_game_length += result.record.total_moves
                    position_metrics.merge(self._collect_position_metric_totals(result))
                    # 只保存少量对局到硬盘，减小存储占用。
                    # 每千局保存两盘：1000k 与 1000k+1。若只存 1000k，在默认 games_per_epoch=384 下
                    # (game_index-1)%384 恒为奇数，启发式/历史对局中模型总在白方；多存一盘可覆盖偶数 offset，回放能看到模型执黑。
                    game_index = int(result.record.game_id.removeprefix("game_"))
                    if game_index >= 1000 and game_index % 1000 in (0, 1):
                        self.artifacts.game_store.save(result.record)
            self._discard_broken_reward_executor()
            training_batch = self._flatten_batches(batches)
            # 与本轮自博弈的采样温度保持一致，否则重要性采样比失真。
            stats = self._update_policy(training_batch, temperature=temperature)
            eval_result = evaluate_policy(
                self.game, self.engine, games=self.config.eval_games,
                heuristic_temperature=self.config.eval_heuristic_temperature,
                anchor_engine=self.anchor_engine,
            )
            metric_record = MetricRecord(
                epoch=epoch,
                games=self.config.self_play_games_per_epoch,
                policy_loss=stats.policy_loss,
                value_loss=stats.value_loss,
                entropy=stats.entropy,
                grad_norm=stats.grad_norm,
                return_mean=training_batch.raw_return_mean,
                return_std=training_batch.raw_return_std,
                return_abs_max=training_batch.raw_return_abs_max,
                avg_game_length=total_game_length / max(len(batches), 1),
                eval_win_rate_random=eval_result.win_rate_random,
                eval_win_rate_heuristic=eval_result.win_rate_heuristic,
                eval_win_rate_anchor=eval_result.win_rate_anchor,
                kl_to_anchor=stats.kl_to_anchor,
                opening_edge_rate=position_metrics.opening_edge_rate,
                opening_corner_rate=position_metrics.opening_corner_rate,
                opening_center_rate=position_metrics.opening_center_rate,
                policy_topk_edge_rate=position_metrics.policy_topk_edge_rate,
            )
            self.artifacts.metric_store.append(metric_record)
            checkpoint_payload = {
                "epoch": epoch,
                "config": self.config.to_dict(),
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            }
            model_rec = ModelRecord(
                checkpoint_name="",
                checkpoint_path="",
                epoch=epoch,
                eval_win_rate_random=eval_result.win_rate_random,
                eval_win_rate_heuristic=eval_result.win_rate_heuristic,
            )
            frame = self.artifacts.metric_store.read_frame()
            best_epoch = compute_best_epoch(frame)
            if best_epoch is not None and best_epoch == epoch:
                path = self.artifacts.checkpoint_store.save("best.pt", checkpoint_payload)
                model_rec.checkpoint_name = "best.pt"
                model_rec.checkpoint_path = str(path)
                self.artifacts.model_registry.upsert(model_rec)
                LOGGER.info("Best epoch=%s, saved best.pt", epoch)
            best_for_resume_epoch = compute_best_epoch_for_resume(frame)
            if best_for_resume_epoch is not None and best_for_resume_epoch == epoch:
                path = self.artifacts.checkpoint_store.save("best_for_resume.pt", checkpoint_payload)
                model_rec.checkpoint_name = "best_for_resume.pt"
                model_rec.checkpoint_path = str(path)
                self.artifacts.model_registry.upsert(model_rec)
                LOGGER.info("Best for resume epoch=%s, saved best_for_resume.pt", epoch)
            if self._baseline is not None and best_epoch is not None:
                best_row = frame[frame["epoch"] == best_epoch]
                if not best_row.empty and "eval_win_rate_heuristic" in best_row.columns:
                    current_heuristic = float(best_row["eval_win_rate_heuristic"].iloc[0])
                    delta = current_heuristic - self._baseline["heuristic"]
                    LOGGER.info(
                        "Baseline (epoch %s): heuristic=%.2f | Current best (epoch %s): heuristic=%.2f | Delta=%+.2f",
                        self._baseline["epoch"],
                        self._baseline["heuristic"],
                        best_epoch,
                        current_heuristic,
                        delta,
                    )
            if epoch % self.config.checkpoint_every == 0:
                checkpoint_name = f"epoch_{epoch:03d}.pt"
                path = self.artifacts.checkpoint_store.save(checkpoint_name, checkpoint_payload)
                model_rec.checkpoint_name = checkpoint_name
                model_rec.checkpoint_path = str(path)
                self.artifacts.model_registry.add(model_rec)
            if epoch == self.config.epochs:
                path = self.artifacts.checkpoint_store.save("last.pt", checkpoint_payload)
                model_rec.checkpoint_name = "last.pt"
                model_rec.checkpoint_path = str(path)
                self.artifacts.model_registry.upsert(model_rec)
                LOGGER.info("Last epoch model saved as last.pt")
            self.scheduler.step()
            self._remember_current_policy(epoch)
            LOGGER.info(
                (
                    "epoch=%s policy_loss=%.4f value_loss=%.4f "
                    "entropy=%.4f grad_norm=%.4f "
                    "return_mean=%.4f return_std=%.4f return_abs_max=%.4f "
                    "eval_random=%.2f (b=%.2f w=%.2f) "
                    "eval_heuristic=%.2f (b=%.2f w=%.2f) "
                    "eval_anchor=%.2f (b=%.2f w=%.2f) kl=%.4f "
                    "opening_edge=%.3f opening_corner=%.3f opening_center=%.3f topk_edge=%.3f "
                    "opponents(heur/hist/self)=%d/%d/%d"
                ),
                epoch,
                metric_record.policy_loss,
                metric_record.value_loss,
                metric_record.entropy,
                metric_record.grad_norm,
                metric_record.return_mean,
                metric_record.return_std,
                metric_record.return_abs_max,
                metric_record.eval_win_rate_random,
                eval_result.win_rate_random_black,
                eval_result.win_rate_random_white,
                metric_record.eval_win_rate_heuristic,
                eval_result.win_rate_heuristic_black,
                eval_result.win_rate_heuristic_white,
                metric_record.eval_win_rate_anchor,
                eval_result.win_rate_anchor_black,
                eval_result.win_rate_anchor_white,
                metric_record.kl_to_anchor,
                metric_record.opening_edge_rate,
                metric_record.opening_corner_rate,
                metric_record.opening_center_rate,
                metric_record.policy_topk_edge_rate,
                epoch_opponent_counts["heuristic"],
                epoch_opponent_counts["historical"],
                epoch_opponent_counts["self"],
            )

    def _collect_position_metric_totals(self, result: SelfPlayResult) -> _PositionMetricTotals:
        totals = _PositionMetricTotals()
        horizon = self.config.reward.opening_position_horizon
        if horizon <= 0:
            return totals

        for transition in result.episode.transitions:
            if transition.move_record_index is None:
                continue
            if transition.move_record_index >= len(result.record.moves):
                continue
            move_record = result.record.moves[transition.move_record_index]
            if move_record.move_index > horizon:
                continue

            totals.opening_moves += 1
            if self._is_corner(move_record.row, move_record.col):
                totals.opening_corners += 1
            elif self._is_edge(move_record.row, move_record.col):
                totals.opening_edges += 1
            if self._is_center(move_record.row, move_record.col):
                totals.opening_centers += 1

            for candidate in move_record.policy_topk:
                totals.topk_candidates += 1
                if self._is_border(candidate.row, candidate.col):
                    totals.topk_edges += 1
        return totals

    def _is_center(self, row: int, col: int) -> bool:
        center = (self.config.board_size - 1) / 2.0
        radius = max(1.0, (self.config.board_size - 1) * self.config.reward.opening_center_radius_ratio)
        distance_sq = (row - center) ** 2 + (col - center) ** 2
        return distance_sq <= radius ** 2

    def _is_corner(self, row: int, col: int) -> bool:
        last_index = self.config.board_size - 1
        return (row, col) in {
            (0, 0),
            (0, last_index),
            (last_index, 0),
            (last_index, last_index),
        }

    def _is_edge(self, row: int, col: int) -> bool:
        last_index = self.config.board_size - 1
        if row in (0, last_index) or col in (0, last_index):
            return True
        if row in (1, last_index - 1) or col in (1, last_index - 1):
            return True
        return False

    def _is_border(self, row: int, col: int) -> bool:
        last_index = self.config.board_size - 1
        return row in (0, last_index) or col in (0, last_index) or row in (1, last_index - 1) or col in (1, last_index - 1)

    def _flatten_batches(self, episodes: list[EpisodeBatch]) -> TrainingBatch:
        states = []
        actions = []
        old_log_probs = []
        returns = []
        advantages = []
        legal_masks = []
        old_values = []
        for episode in episodes:
            episode_returns, episode_advantages = episode.compute_returns_and_advantages(
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
            for index, transition in enumerate(episode.transitions):
                states.append(transition.state)
                actions.append(transition.action)
                old_log_probs.append(transition.old_log_prob)
                returns.append(float(episode_returns[index].item()))
                advantages.append(float(episode_advantages[index].item()))
                legal_masks.append(transition.legal_mask)
                old_values.append(transition.value)
        raw_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        raw_mean = float(raw_returns.mean().item()) if raw_returns.numel() else 0.0
        raw_std = float(raw_returns.std(unbiased=False).item()) if raw_returns.numel() else 0.0
        raw_abs_max = float(raw_returns.abs().max().item()) if raw_returns.numel() else 0.0
        ret_std = raw_returns.std() + 1e-8
        normalized_returns = (raw_returns - raw_returns.mean()) / ret_std
        normalized_returns = normalized_returns.clamp(-1.0, 1.0)
        state_tensor = torch.stack(states).to(self.device)
        mask_tensor = torch.stack(legal_masks).to(self.device)
        return TrainingBatch(
            states=state_tensor,
            actions=torch.tensor(actions, dtype=torch.long, device=self.device),
            old_log_probs=torch.tensor(old_log_probs, dtype=torch.float32, device=self.device),
            returns=normalized_returns,
            advantages=torch.tensor(advantages, dtype=torch.float32, device=self.device),
            legal_masks=mask_tensor,
            old_values=torch.tensor(old_values, dtype=torch.float32, device=self.device),
            raw_return_mean=raw_mean,
            raw_return_std=raw_std,
            raw_return_abs_max=raw_abs_max,
            anchor_log_probs=self._anchor_log_probs(state_tensor, mask_tensor),
        )

    @torch.no_grad()
    def _anchor_log_probs(self, states: torch.Tensor, legal_masks: torch.Tensor):
        """锚点策略在这批局面上的 log π（T=1）。kl_coef=0 时不计算。"""
        if self.config.kl_coef <= 0.0 or states.numel() == 0:
            return None
        anchor_model = self.anchor_engine.model
        anchor_model.eval()
        outputs = []
        # 整批一次前向可能超显存，按块走；锚点冻结，无需梯度。
        for start in range(0, states.size(0), 4096):
            logits, _ = anchor_model(states[start : start + 4096])
            masked = logits.masked_fill(legal_masks[start : start + 4096] == 0, -1e9)
            outputs.append(torch.log_softmax(masked, dim=-1))
        return torch.cat(outputs, dim=0)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self._adopt_checkpoint_architecture(checkpoint.get("config"))
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if "config" in checkpoint:
            saved = checkpoint["config"]
            self._log_reward_config_source(saved.get("reward"))
            apply_saved_config(
                self.config, saved, reward_from_checkpoint=self.reward_from_checkpoint
            )
            # create_run 在加载 checkpoint 之前就写过 config.json，那份是合并前的快照。
            # 用合并后的配置覆盖，保证 run 目录记录的就是本次实际生效的配置。
            write_json(self.artifacts.run_dir / "config.json", self.config.to_dict())

        self._start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self._restore_lr_schedule(self._start_epoch - 1)
        LOGGER.info("Checkpoint loaded, resuming from epoch %s", self._start_epoch)
        self.model.eval()

        self._baseline = self._load_baseline_from_checkpoint(checkpoint_path)

    def _adopt_checkpoint_architecture(self, saved: dict | None) -> None:
        """按 checkpoint 记录的结构重建网络，再加载权重。

        网络在 __init__ 里就按当前 config 建好了，而权重要到这里才载入。若 checkpoint
        的通道数/残差块数与当前配置不同（例如默认值调小后去续训旧的 256x16 预训练模型），
        load_state_dict 会直接抛形状不符的错。权重决定结构，因此以 checkpoint 为准重建，
        但必须显式告知——否则用户以为在训小模型，实际训的是大模型。
        """
        if not isinstance(saved, dict):
            return
        saved_model = saved.get("model")
        if not isinstance(saved_model, dict):
            return
        channels = int(saved_model.get("channels", self.config.model.channels))
        blocks = int(saved_model.get("blocks", self.config.model.blocks))
        if channels == self.config.model.channels and blocks == self.config.model.blocks:
            return

        LOGGER.warning(
            "Checkpoint architecture is %dx%d but the current config asks for %dx%d; "
            "rebuilding the network as %dx%d to match the weights. "
            "To actually train the smaller network, re-run pretraining at the new size "
            "instead of resuming this checkpoint.",
            channels,
            blocks,
            self.config.model.channels,
            self.config.model.blocks,
            channels,
            blocks,
        )
        self.config.model = ModelConfig(channels=channels, blocks=blocks)
        self.model = PolicyValueNet(
            board_size=self.config.board_size, channels=channels, blocks=blocks
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.engine = ModelAIEngine(self.model, device=self.config.device)

    def _restore_lr_schedule(self, last_epoch: int) -> None:
        """把学习率重新锚定到当前 config.learning_rate，再从 last_epoch 处接上余弦调度。

        `optimizer.load_state_dict` 会把 checkpoint 里的 lr / initial_lr 一并覆盖回来，而
        CosineAnnealingLR 是按优化器当前 lr 递推的。不重锚会有两个后果：
        1. 从 BC checkpoint 起步时，载入的是预训练余弦的谷底（默认 1e-5），PPO 会一直贴着它跑，
           比配置的 3.5e-4 低一个多数量级；
        2. 续训时 --learning-rate 完全失效，因为它敌不过 checkpoint 里的旧 lr。
        """
        base_lr = self.config.learning_rate
        resumed_lr = cosine_lr_at(base_lr, self.config.lr_min, self.config.epochs, last_epoch)
        for group in self.optimizer.param_groups:
            group["initial_lr"] = base_lr
            group["lr"] = resumed_lr
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.lr_min,
            last_epoch=last_epoch,
        )
        LOGGER.info(
            "LR schedule re-anchored to base=%.3e; epoch %s starts at lr=%.3e",
            base_lr,
            self._start_epoch,
            self.optimizer.param_groups[0]["lr"],
        )

    def _log_reward_config_source(self, saved_reward: dict | None) -> None:
        """把本次续训实际生效的奖励配置及其与 checkpoint 的差异记入日志，避免静默改变训练语义。"""
        if not isinstance(saved_reward, dict):
            LOGGER.info("Checkpoint carries no reward config; using the current RewardConfig.")
            return

        source = "checkpoint" if self.reward_from_checkpoint else "current config"
        LOGGER.info("Reward config source: %s", source)
        changed, added = diff_reward_config(saved_reward, self.config.reward)
        if changed:
            LOGGER.info("  differs from checkpoint (checkpoint -> current): %s", "; ".join(changed))
        if added:
            LOGGER.info("  fields absent from checkpoint, using defaults: %s", "; ".join(added))
        if not changed and not added:
            LOGGER.info("  identical to the checkpoint's reward config.")
        elif self.reward_from_checkpoint and changed:
            LOGGER.warning(
                "  --reward-from-checkpoint is set: the differences above are overridden by the checkpoint."
            )

    def _load_baseline_from_checkpoint(self, checkpoint_path: str) -> dict | None:
        """从 checkpoint 所在 run 的 metrics.csv 读取基线，写入新 run 的 baseline.json。"""
        old_run_dir = Path(checkpoint_path).resolve().parent.parent
        old_metrics_path = old_run_dir / "metrics.csv"
        if not old_metrics_path.exists():
            return None
        try:
            frame = pd.read_csv(old_metrics_path)
        except Exception:
            return None
        if "epoch" not in frame.columns or "eval_win_rate_heuristic" not in frame.columns:
            return None
        baseline_epoch = int(self._start_epoch - 1)
        row = frame[frame["epoch"] == baseline_epoch]
        if row.empty:
            return None
        heuristic = float(row["eval_win_rate_heuristic"].iloc[0])
        random_wr = float(row["eval_win_rate_random"].iloc[0]) if "eval_win_rate_random" in frame.columns else 0.0
        baseline = {"epoch": baseline_epoch, "heuristic": heuristic, "random": random_wr}
        baseline_path = self.artifacts.run_dir / "baseline.json"
        with baseline_path.open("w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)
        LOGGER.info("Baseline recorded: epoch=%s heuristic=%.2f", baseline_epoch, heuristic)
        return baseline

    def _update_policy(self, batch: TrainingBatch, temperature: float = 1.0):
        """PPO 更新。

        `temperature` 必须与自博弈采样时用的温度相同：`old_log_prob` 记录的是采样
        分布 softmax(logits / T) 下的概率，若这里按 T=1 求新策略概率，重要性采样比
        exp(new - old) 会系统性偏离 1（实测 T=0.35 时均值 0.95、14.6% 的样本一进来
        就落在裁剪区间外），PPO 的裁剪语义随之失效。
        """
        if batch.states.size(0) == 0:
            return _LossStats(policy_loss=0.0, value_loss=0.0, entropy=0.0, grad_norm=0.0)
        inverse_temperature = 1.0 / max(temperature, 1e-3)

        self.model.train()
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss_stats = _LossStats(policy_loss=0.0, value_loss=0.0, entropy=0.0, grad_norm=0.0)
        sample_count = batch.states.size(0)
        num_batches = 0
        for _ in range(self.config.updates_per_epoch):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.config.batch_size):
                batch_indices = permutation[start : start + self.config.batch_size]
                states = batch.states[batch_indices]
                actions = batch.actions[batch_indices]
                old_log_probs = batch.old_log_probs[batch_indices]
                returns = batch.returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                legal_masks = batch.legal_masks[batch_indices]
                batch_old_values = batch.old_values[batch_indices]

                logits, values = self.model(states)
                masked_raw = logits.masked_fill(legal_masks == 0, -1e9)
                masked_logits = masked_raw * inverse_temperature
                log_probs = torch.log_softmax(masked_logits, dim=-1)
                probs = torch.softmax(masked_logits, dim=-1)
                chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                ratios = (chosen_log_probs - old_log_probs).exp()
                unclipped = ratios * batch_advantages
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values,
                    -self.config.value_clip_epsilon,
                    self.config.value_clip_epsilon,
                )
                value_loss_unclipped = (values - returns) ** 2
                value_loss_clipped = (value_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy = -(probs * log_probs).sum(dim=-1).mean()

                # 对锚点的正向 KL：KL(π_anchor ‖ π)。选正向而非反向是因为它是
                # 「覆盖型」的——锚点把 0.8 的概率压在唯一挡点、而新策略只给 0.02 时
                # 会被重罚，正好对应我们要防止的失效模式（策略散开、丢掉尖峰）。
                # 用 T=1 的原始分布，约束的是实际对弈的策略而非探索用的采样分布。
                kl_penalty = torch.zeros((), device=self.device)
                if self.config.kl_coef > 0.0 and batch.anchor_log_probs is not None:
                    anchor_log_probs = batch.anchor_log_probs[batch_indices]
                    raw_log_probs = torch.log_softmax(masked_raw, dim=-1)
                    kl_penalty = (
                        anchor_log_probs.exp() * (anchor_log_probs - raw_log_probs)
                    ).sum(dim=-1).mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                    + self.config.kl_coef * kl_penalty
                )
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()
                loss_stats.policy_loss += float(policy_loss.item())
                loss_stats.value_loss += float(value_loss.item())
                loss_stats.entropy += float(entropy.item())
                loss_stats.grad_norm += float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                loss_stats.kl_to_anchor += float(kl_penalty.item())
                num_batches += 1
        if num_batches > 0:
            loss_stats.policy_loss /= num_batches
            loss_stats.value_loss /= num_batches
            loss_stats.entropy /= num_batches
            loss_stats.grad_norm /= num_batches
            loss_stats.kl_to_anchor /= num_batches
        self.model.eval()
        return loss_stats

    def _build_engine_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> ModelAIEngine:
        opponent_model = PolicyValueNet(
            board_size=self.config.board_size,
            channels=self.config.model.channels,
            blocks=self.config.model.blocks,
        ).to(self.device)
        opponent_model.load_state_dict(state_dict)
        return ModelAIEngine(opponent_model, device=self.config.device)

    def _clone_model_state(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def _remember_current_policy(self, epoch: int) -> None:
        if self.config.opponent_pool_size <= 0:
            return
        if epoch % self.config.opponent_snapshot_interval != 0:
            return
        self.historical_opponent_snapshots.append(self._clone_model_state())
        if len(self.historical_opponent_snapshots) > self.config.opponent_pool_size:
            self.historical_opponent_snapshots.pop(0)

    def _sample_historical_opponent(self) -> ModelAIEngine | None:
        if not self.historical_opponent_snapshots:
            return None
        snapshot = random.choice(self.historical_opponent_snapshots)
        return self._build_engine_from_state_dict(snapshot)


@dataclass(slots=True)
class _LossStats:
    policy_loss: float
    value_loss: float
    entropy: float
    grad_norm: float
    kl_to_anchor: float = 0.0


def build_arg_parser() -> argparse.ArgumentParser:
    d = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train a Gomoku PPO self-play model.")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--games-per-epoch", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=768, help="Batch size for training")
    parser.add_argument("--run-name", type=str, default="ppo_gomoku_5080")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to resume training")
    parser.add_argument(
        "--reward-from-checkpoint",
        action="store_true",
        help=(
            "Restore RewardConfig from the checkpoint instead of the current config. "
            "Use it to reproduce an old run; by default reward changes take effect on resume."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate (default: 3.5e-4)")
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        help=f"Conv channels; must match the checkpoint being resumed (default: {d.model.channels})",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=None,
        help=f"Residual blocks; must match the checkpoint being resumed (default: {d.model.blocks})",
    )
    parser.add_argument(
        "--reward-workers",
        type=int,
        default=None,
        help=(
            "Worker processes for reward computation, which is ~90%% of self-play time "
            f"(0 = auto, 1 = inline). Default: {d.reward_workers}"
        ),
    )
    parser.add_argument(
        "--self-play-batch-games",
        type=int,
        default=None,
        help=(
            "Games advanced in parallel so their positions share one forward pass "
            f"(1 = sequential). Default: {d.self_play_batch_games}"
        ),
    )
    parser.add_argument(
        "--heuristic-max-prob",
        type=float,
        default=None,
        help=(
            "Per-game probability of sampling heuristic opponent at schedule peak "
            f"(default: {d.heuristic_opponent_max_prob})"
        ),
    )
    parser.add_argument(
        "--heuristic-start-fraction",
        type=float,
        default=None,
        help=(
            "Start ramp after this fraction of total epochs (0 = from epoch 1). "
            f"Default: {d.heuristic_start_fraction}"
        ),
    )
    parser.add_argument(
        "--heuristic-ramp-fraction",
        type=float,
        default=None,
        help=(
            "Linear ramp reaches peak at this epoch fraction (from start). "
            f"Default: {d.heuristic_ramp_fraction}"
        ),
    )
    return parser


def main() -> None:
    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args()
    config = TrainingConfig(
        board_size=args.board_size,
        epochs=args.epochs,
        self_play_games_per_epoch=args.games_per_epoch,
        batch_size=args.batch_size,
        run_name=args.run_name,
        device=args.device,
    )
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.channels is not None:
        config.model.channels = args.channels
    if args.blocks is not None:
        config.model.blocks = args.blocks
    if args.self_play_batch_games is not None:
        config.self_play_batch_games = max(1, args.self_play_batch_games)
    if args.reward_workers is not None:
        config.reward_workers = max(0, args.reward_workers)
    if args.heuristic_max_prob is not None:
        config.heuristic_opponent_max_prob = args.heuristic_max_prob
    if args.heuristic_start_fraction is not None:
        config.heuristic_start_fraction = args.heuristic_start_fraction
    if args.heuristic_ramp_fraction is not None:
        config.heuristic_ramp_fraction = args.heuristic_ramp_fraction
    set_seed(config.seed)
    trainer = PPOTrainer(
        config,
        checkpoint_path=args.checkpoint,
        reward_from_checkpoint=args.reward_from_checkpoint,
    )
    trainer.train()


if __name__ == "__main__":
    main()