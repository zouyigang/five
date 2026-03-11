from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import torch
from torch import nn

from five.ai.inference import ModelAIEngine
from five.ai.model import PolicyValueNet
from five.common.config import ModelConfig, RewardConfig, TrainingConfig
from five.common.logging import configure_logging, get_logger
from five.common.utils import set_seed
from five.core.game import GomokuGame
from five.storage.schemas import MetricRecord, ModelRecord
from five.train.dataset import EpisodeBatch
from five.train.evaluator import evaluate_policy
from five.train.run_manager import RunArtifacts, create_run
from five.train.self_play import play_self_play_game


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class TrainingBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    legal_masks: torch.Tensor

    @property
    def return_mean(self) -> float:
        return float(self.returns.mean().item()) if self.returns.numel() else 0.0

    @property
    def return_std(self) -> float:
        return float(self.returns.std(unbiased=False).item()) if self.returns.numel() else 0.0

    @property
    def return_abs_max(self) -> float:
        return float(self.returns.abs().max().item()) if self.returns.numel() else 0.0


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
        self.engine = ModelAIEngine(self.model, device=config.device)
        self.artifacts: RunArtifacts = create_run(config)
        self.historical_opponent_snapshots: list[dict[str, torch.Tensor]] = []
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def train(self) -> None:
        for epoch in range(1, self.config.epochs + 1):
            self.model.eval()
            # Temperature annealing: 从1.0降到0.3
            progress = min(epoch / self.config.epochs, 1.0)
            temperature = 1.0 - (1.0 - 0.3) * progress
            historical_opponent = self._sample_historical_opponent()

            batches: list[EpisodeBatch] = []
            total_game_length = 0
            for game_offset in range(self.config.self_play_games_per_epoch):
                game_index = (epoch - 1) * self.config.self_play_games_per_epoch + game_offset + 1
                use_historical_opponent = (
                    historical_opponent is not None
                    and random.random() < self.config.historical_opponent_prob
                )
                black_engine = self.engine
                white_engine = self.engine
                tracked_players: set[int] | None = None
                if use_historical_opponent:
                    if game_offset % 2 == 0:
                        white_engine = historical_opponent
                        tracked_players = {1}
                    else:
                        black_engine = historical_opponent
                        tracked_players = {-1}
                result = play_self_play_game(
                    game=self.game,
                    black_engine=black_engine,
                    white_engine=white_engine,
                    run_id=self.artifacts.run_id,
                    game_index=game_index,
                    temperature=temperature,
                    reward_config=self.config.reward,
                    tracked_players=tracked_players,
                )
                batches.append(result.episode)
                total_game_length += result.record.total_moves
                # 只保存少量对局到硬盘，减小存储占用。
                # 这里改为每 1000 局保存一局，其余仅用于训练。
                if game_index % 1000 == 0:
                    self.artifacts.game_store.save(result.record)
            training_batch = self._flatten_batches(batches)
            stats = self._update_policy(training_batch)
            eval_result = evaluate_policy(self.game, self.engine, games=self.config.eval_games)
            metric_record = MetricRecord(
                epoch=epoch,
                games=self.config.self_play_games_per_epoch,
                policy_loss=stats.policy_loss,
                value_loss=stats.value_loss,
                entropy=stats.entropy,
                grad_norm=stats.grad_norm,
                return_mean=training_batch.return_mean,
                return_std=training_batch.return_std,
                return_abs_max=training_batch.return_abs_max,
                avg_game_length=total_game_length / max(len(batches), 1),
                eval_win_rate_random=eval_result.win_rate_random,
                eval_win_rate_heuristic=eval_result.win_rate_heuristic,
            )
            self.artifacts.metric_store.append(metric_record)
            if epoch % self.config.checkpoint_every == 0:
                checkpoint_name = f"epoch_{epoch:03d}.pt"
                path = self.artifacts.checkpoint_store.save(
                    checkpoint_name,
                    {
                        "epoch": epoch,
                        "config": self.config.to_dict(),
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                    },
                )
                self.artifacts.model_registry.add(
                    ModelRecord(
                        checkpoint_name=checkpoint_name,
                        checkpoint_path=str(path),
                        epoch=epoch,
                        eval_win_rate_random=eval_result.win_rate_random,
                        eval_win_rate_heuristic=eval_result.win_rate_heuristic,
                    )
                )
            self._remember_current_policy()
            LOGGER.info(
                (
                    "epoch=%s policy_loss=%.4f value_loss=%.4f "
                    "entropy=%.4f grad_norm=%.4f "
                    "return_mean=%.4f return_std=%.4f return_abs_max=%.4f "
                    "eval_random=%.2f (b=%.2f w=%.2f) "
                    "eval_heuristic=%.2f (b=%.2f w=%.2f)"
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
            )

    def _flatten_batches(self, episodes: list[EpisodeBatch]) -> TrainingBatch:
        states = []
        actions = []
        old_log_probs = []
        returns = []
        advantages = []
        legal_masks = []
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
        return TrainingBatch(
            states=torch.stack(states).to(self.device),
            actions=torch.tensor(actions, dtype=torch.long, device=self.device),
            old_log_probs=torch.tensor(old_log_probs, dtype=torch.float32, device=self.device),
            returns=torch.tensor(returns, dtype=torch.float32, device=self.device),
            advantages=torch.tensor(advantages, dtype=torch.float32, device=self.device),
            legal_masks=torch.stack(legal_masks).to(self.device),
        )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model and optimizer state from checkpoint."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        # Update config if present (nested model/reward must be restored as dataclasses)
        if "config" in checkpoint:
            saved = checkpoint["config"]
            for key, value in saved.items():
                if not hasattr(self.config, key):
                    continue
                if key == "model" and isinstance(value, dict):
                    subset = {k: v for k, v in value.items() if k in getattr(ModelConfig, "__dataclass_fields__", {})}
                    setattr(self.config, key, ModelConfig(**subset))
                elif key == "reward" and isinstance(value, dict):
                    subset = {k: v for k, v in value.items() if k in getattr(RewardConfig, "__dataclass_fields__", {})}
                    setattr(self.config, key, RewardConfig(**subset))
                else:
                    setattr(self.config, key, value)
        # Resume epoch count
        self._start_epoch = int(checkpoint.get("epoch", 0)) + 1
        logger.info("Checkpoint loaded successfully, resuming from epoch %s", self._start_epoch)
        self.model.eval()

    def _update_policy(self, batch: TrainingBatch):
        if batch.states.size(0) == 0:
            return _LossStats(policy_loss=0.0, value_loss=0.0, entropy=0.0, grad_norm=0.0)

        self.model.train()
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss_stats = _LossStats(policy_loss=0.0, value_loss=0.0, entropy=0.0, grad_norm=0.0)
        sample_count = batch.states.size(0)
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
                value_loss = nn.functional.huber_loss(values, returns, delta=1.0)
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
        denominator = self.config.updates_per_epoch * max(sample_count // self.config.batch_size, 1)
        loss_stats.policy_loss /= denominator
        loss_stats.value_loss /= denominator
        loss_stats.entropy /= denominator
        loss_stats.grad_norm /= denominator
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

    def _remember_current_policy(self) -> None:
        if self.config.opponent_pool_size <= 0:
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
    parser = argparse.ArgumentParser(description="Train a Gomoku PPO self-play model.")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--games-per-epoch", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--run-name", type=str, default="ppo_gomoku")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to resume training")
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
    set_seed(config.seed)
    trainer = PPOTrainer(config, checkpoint_path=args.checkpoint)
    trainer.train()


if __name__ == "__main__":
    main()