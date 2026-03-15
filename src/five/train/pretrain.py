"""Behavioral Cloning pre-training for PolicyValueNet.

Trains the policy head via cross-entropy against heuristic expert actions
and the value head via MSE against game outcomes, producing a checkpoint
that can be loaded by ``five-train --checkpoint ...`` for PPO fine-tuning.

Usage:
    five-pretrain --dataset data/heuristic_20k.pt --epochs 15 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from five.ai.model import PolicyValueNet
from five.common.config import ModelConfig, PretrainConfig, TrainingConfig
from five.common.logging import configure_logging, get_logger
from five.common.utils import ensure_dir, set_seed

DEFAULT_PRETRAIN = PretrainConfig()

LOGGER = get_logger(__name__)


def _load_dataset(path: str, device: torch.device) -> TensorDataset:
    data = torch.load(path, map_location="cpu", weights_only=False)
    states = data["states"]
    actions = data["actions"]
    legal_masks = data["legal_masks"]
    values = data["values"]
    return TensorDataset(states, actions, legal_masks, values)


def pretrain(
    dataset_path: str,
    *,
    board_size: int = 9,
    channels: int = 256,
    blocks: int = 16,
    epochs: int = 15,
    batch_size: int = 1024,
    lr: float = 1e-3,
    value_coef: float = 0.5,
    device_str: str = "cuda",
    output_dir: str = "pretrain_output",
    seed: int = 42,
    on_epoch_end: Callable[[int, int, float, float, float, float], None] | None = None,
    progress_file: str | None = None,
) -> Path:
    """Run behavioral cloning and return path to the saved checkpoint.

    If on_epoch_end is given, it is called after each epoch with:
      (epoch, total_epochs, policy_loss, value_loss, accuracy_pct, lr).
    """
    set_seed(seed)
    device = torch.device(device_str)

    dataset = _load_dataset(dataset_path, device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    LOGGER.info("Dataset loaded: %d samples, %d batches/epoch", len(dataset), len(loader))

    model = PolicyValueNet(
        board_size=board_size,
        channels=channels,
        blocks=blocks,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    policy_criterion = nn.CrossEntropyLoss()

    out = ensure_dir(Path(output_dir))
    best_loss = math.inf
    best_path: Path | None = None
    history: list[dict] = []
    if progress_file:
        Path(progress_file).parent.mkdir(parents=True, exist_ok=True)

    total_batches_per_epoch = len(loader)
    for epoch in range(1, epochs + 1):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for batch_states, batch_actions, batch_masks, batch_values in loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_masks = batch_masks.to(device)
            batch_values = batch_values.to(device)

            logits, values = model(batch_states)

            masked_logits = logits.masked_fill(batch_masks == 0, -1e9)
            policy_loss = policy_criterion(masked_logits, batch_actions)

            value_loss = nn.functional.mse_loss(values, batch_values)

            loss = policy_loss + value_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            preds = masked_logits.argmax(dim=-1)
            correct += (preds == batch_actions).sum().item()
            total += batch_actions.size(0)
            num_batches += 1
            if num_batches % 50 == 0 or num_batches == total_batches_per_epoch:
                print(
                    f"\rEpoch {epoch}/{epochs}  Batch {num_batches}/{total_batches_per_epoch}",
                    end="",
                    flush=True,
                )

        print()
        scheduler.step()
        avg_p = total_policy_loss / max(num_batches, 1)
        avg_v = total_value_loss / max(num_batches, 1)
        acc = correct / max(total, 1) * 100.0
        current_lr = optimizer.param_groups[0]["lr"]
        if on_epoch_end is not None:
            on_epoch_end(epoch, epochs, avg_p, avg_v, acc, current_lr)
        LOGGER.info(
            "epoch %d/%d  policy_loss=%.4f  value_loss=%.4f  accuracy=%.2f%%  lr=%.2e",
            epoch, epochs, avg_p, avg_v, acc, current_lr,
        )
        print(
            f"epoch {epoch}/{epochs}  policy_loss={avg_p:.4f}  value_loss={avg_v:.4f}  accuracy={acc:.2f}%  lr={current_lr:.2e}",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()

        history.append({
            "epoch": epoch,
            "policy_loss": avg_p,
            "value_loss": avg_v,
            "accuracy": acc,
            "lr": current_lr,
        })
        if progress_file:
            payload = {
                "running": True,
                "output_dir": output_dir,
                "current_epoch": epoch,
                "total_epochs": epochs,
                "policy_loss": avg_p,
                "value_loss": avg_v,
                "accuracy": acc,
                "lr": current_lr,
                "history": history,
            }
            Path(progress_file).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        combined = avg_p + value_coef * avg_v
        if combined < best_loss:
            best_loss = combined
            best_path = out / "best_bc.pt"
            _save_checkpoint(best_path, model, optimizer, epoch, board_size, channels, blocks)
            LOGGER.info("  -> new best (loss=%.4f), saved to %s", combined, best_path)
            print(f"  -> new best (loss={combined:.4f}), saved to {best_path}", flush=True)

    final_path = out / "final_bc.pt"
    _save_checkpoint(final_path, model, optimizer, epochs, board_size, channels, blocks)
    LOGGER.info("Final checkpoint saved to %s", final_path)
    print(f"Final checkpoint saved to {final_path}", flush=True)
    if progress_file:
        payload = {
            "running": False,
            "output_dir": output_dir,
            "current_epoch": epochs,
            "total_epochs": epochs,
            "history": history,
        }
        Path(progress_file).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return best_path or final_path


def _save_checkpoint(
    path: Path,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    board_size: int,
    channels: int,
    blocks: int,
) -> None:
    config = TrainingConfig(
        board_size=board_size,
        model=ModelConfig(channels=channels, blocks=blocks),
    )
    torch.save(
        {
            "epoch": 0,
            "config": config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "pretrain_epoch": epoch,
        },
        path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    c = DEFAULT_PRETRAIN
    p = argparse.ArgumentParser(description="Behavioral Cloning pre-training for Gomoku PolicyValueNet.")
    p.add_argument("--dataset", type=str, default=c.dataset, help=f"Path to .pt dataset (default: {c.dataset})")
    p.add_argument("--board-size", type=int, default=c.board_size)
    p.add_argument("--channels", type=int, default=c.channels, help="Network channel width (must match PPO config)")
    p.add_argument("--blocks", type=int, default=c.blocks, help="Number of residual blocks (must match PPO config)")
    p.add_argument("--epochs", type=int, default=c.epochs, help=f"Pre-training epochs (default: {c.epochs})")
    p.add_argument("--batch-size", type=int, default=c.batch_size)
    p.add_argument("--lr", type=float, default=c.lr, help=f"Initial learning rate (default: {c.lr})")
    p.add_argument("--value-coef", type=float, default=c.value_coef, help=f"Value loss weight (default: {c.value_coef})")
    p.add_argument("--device", type=str, default=c.device)
    p.add_argument("--output-dir", type=str, default=c.output_dir, help=f"Output directory (default: {c.output_dir})")
    p.add_argument(
        "--progress-file",
        type=str,
        default="",
        help="JSON file for GUI progress (default: <output-dir>/pretrain.progress.json)",
    )
    p.add_argument("--seed", type=int, default=c.seed)
    return p


def main() -> None:
    configure_logging()
    args = build_arg_parser().parse_args()
    progress_file = args.progress_file or str(Path(args.output_dir) / "pretrain.progress.json")
    pretrain(
        dataset_path=args.dataset,
        board_size=args.board_size,
        channels=args.channels,
        blocks=args.blocks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        value_coef=args.value_coef,
        device_str=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
        progress_file=progress_file,
    )


if __name__ == "__main__":
    main()
