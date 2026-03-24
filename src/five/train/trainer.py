from __future__ import annotations

import argparse
import json
import random
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
from five.common.utils import set_seed
from five.core.game import GomokuGame
from five.storage.schemas import MetricRecord, ModelRecord
from five.train.dataset import EpisodeBatch
from five.train.evaluator import evaluate_policy
from five.train.best_epoch import compute_best_epoch, compute_best_epoch_for_resume
from five.train.run_manager import RunArtifacts, create_run
from five.train.self_play import SelfPlayResult, play_self_play_game


LOGGER = get_logger(__name__)


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
    def __init__(self, config: TrainingConfig, checkpoint_path: str | None = None) -> None:
        self.config = config
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
            for game_offset in range(self.config.self_play_games_per_epoch):
                game_index = (epoch - 1) * self.config.self_play_games_per_epoch + game_offset + 1
                black_engine = self.engine
                white_engine = self.engine
                tracked_players: set[int] | None = None
                roll = random.random()
                if roll < heuristic_prob:
                    if game_offset % 2 == 0:
                        white_engine = heuristic_opponent
                        tracked_players = {1}
                    else:
                        black_engine = heuristic_opponent
                        tracked_players = {-1}
                elif (
                    historical_opponent is not None
                    and roll < heuristic_prob + self.config.historical_opponent_prob
                ):
                    if game_offset % 2 == 0:
                        white_engine = historical_opponent
                        tracked_players = {1}
                    else:
                        black_engine = historical_opponent
                        tracked_players = {-1}
                black_player_name = (
                    "model"
                    if black_engine is self.engine
                    else "heuristic"
                    if black_engine is heuristic_opponent
                    else "historical"
                )
                white_player_name = (
                    "model"
                    if white_engine is self.engine
                    else "heuristic"
                    if white_engine is heuristic_opponent
                    else "historical"
                )
                result = play_self_play_game(
                    game=self.game,
                    black_engine=black_engine,
                    white_engine=white_engine,
                    run_id=self.artifacts.run_id,
                    game_index=game_index,
                    temperature=temperature,
                    reward_config=self.config.reward,
                    tracked_players=tracked_players,
                    black_player=black_player_name,
                    white_player=white_player_name,
                )
                batches.append(result.episode)
                total_game_length += result.record.total_moves
                position_metrics.merge(self._collect_position_metric_totals(result))
                # 只保存少量对局到硬盘，减小存储占用。
                # 每千局保存两盘：1000k 与 1000k+1。若只存 1000k，在默认 games_per_epoch=384 下
                # (game_index-1)%384 恒为奇数，启发式/历史对局中模型总在白方；多存一盘可覆盖偶数 offset，回放能看到模型执黑。
                if game_index >= 1000 and game_index % 1000 in (0, 1):
                    self.artifacts.game_store.save(result.record)
            training_batch = self._flatten_batches(batches)
            stats = self._update_policy(training_batch)
            eval_result = evaluate_policy(
                self.game, self.engine, games=self.config.eval_games,
                heuristic_temperature=self.config.eval_heuristic_temperature,
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
                    "opening_edge=%.3f opening_corner=%.3f opening_center=%.3f topk_edge=%.3f"
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
                metric_record.opening_edge_rate,
                metric_record.opening_corner_rate,
                metric_record.opening_center_rate,
                metric_record.policy_topk_edge_rate,
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
        return TrainingBatch(
            states=torch.stack(states).to(self.device),
            actions=torch.tensor(actions, dtype=torch.long, device=self.device),
            old_log_probs=torch.tensor(old_log_probs, dtype=torch.float32, device=self.device),
            returns=normalized_returns,
            advantages=torch.tensor(advantages, dtype=torch.float32, device=self.device),
            legal_masks=torch.stack(legal_masks).to(self.device),
            old_values=torch.tensor(old_values, dtype=torch.float32, device=self.device),
            raw_return_mean=raw_mean,
            raw_return_std=raw_std,
            raw_return_abs_max=raw_abs_max,
        )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if "config" in checkpoint:
            saved = checkpoint["config"]
            # 继续训练时保留用户配置，不随 checkpoint 覆盖
            skip_keys = {
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
            for key, value in saved.items():
                if key in skip_keys or not hasattr(self.config, key):
                    continue
                if key == "model" and isinstance(value, dict):
                    subset = {k: v for k, v in value.items() if k in getattr(ModelConfig, "__dataclass_fields__", {})}
                    setattr(self.config, key, ModelConfig(**subset))
                elif key == "reward" and isinstance(value, dict):
                    subset = {k: v for k, v in value.items() if k in getattr(RewardConfig, "__dataclass_fields__", {})}
                    setattr(self.config, key, RewardConfig(**subset))
                else:
                    setattr(self.config, key, value)

        self._start_epoch = int(checkpoint.get("epoch", 0)) + 1
        last_epoch = self._start_epoch - 1
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs, eta_min=self.config.lr_min, last_epoch=last_epoch
        )
        LOGGER.info("Checkpoint loaded, resuming from epoch %s", self._start_epoch)
        self.model.eval()

        self._baseline = self._load_baseline_from_checkpoint(checkpoint_path)

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

    def _update_policy(self, batch: TrainingBatch):
        if batch.states.size(0) == 0:
            return _LossStats(policy_loss=0.0, value_loss=0.0, entropy=0.0, grad_norm=0.0)

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
                masked_logits = logits.masked_fill(legal_masks == 0, -1e9)
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
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()
                loss_stats.policy_loss += float(policy_loss.item())
                loss_stats.value_loss += float(value_loss.item())
                loss_stats.entropy += float(entropy.item())
                loss_stats.grad_norm += float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                num_batches += 1
        if num_batches > 0:
            loss_stats.policy_loss /= num_batches
            loss_stats.value_loss /= num_batches
            loss_stats.entropy /= num_batches
            loss_stats.grad_norm /= num_batches
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
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate (default: 3.5e-4)")
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
    if args.heuristic_max_prob is not None:
        config.heuristic_opponent_max_prob = args.heuristic_max_prob
    if args.heuristic_start_fraction is not None:
        config.heuristic_start_fraction = args.heuristic_start_fraction
    if args.heuristic_ramp_fraction is not None:
        config.heuristic_ramp_fraction = args.heuristic_ramp_fraction
    set_seed(config.seed)
    trainer = PPOTrainer(config, checkpoint_path=args.checkpoint)
    trainer.train()


if __name__ == "__main__":
    main()