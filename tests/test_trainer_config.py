import pandas as pd
import pytest
import torch

from five.ai.model import PolicyValueNet
from five.common.config import ModelConfig, RewardConfig, TrainingConfig
from five.train.best_epoch import compute_best_epoch
from five.train.trainer import (
    RESUME_SKIP_KEYS,
    PPOTrainer,
    apply_saved_config,
    cosine_lr_at,
    diff_reward_config,
    resolve_opponent_kind,
)


def _saved_config(**reward_overrides) -> dict:
    saved = TrainingConfig()
    for key, value in reward_overrides.items():
        setattr(saved.reward, key, value)
    return saved.to_dict()


def test_reward_config_is_not_restored_from_checkpoint_by_default():
    saved = _saved_config(counter_threat_waiver_scale=1.0, open_three_score=99.0)
    config = TrainingConfig()

    apply_saved_config(config, saved)

    # 用户当前的 RewardConfig 必须原样生效，否则调完奖励参数续训会静默不起作用。
    assert config.reward.counter_threat_waiver_scale == RewardConfig().counter_threat_waiver_scale
    assert config.reward.open_three_score == RewardConfig().open_three_score


def test_reward_config_is_restored_when_explicitly_requested():
    saved = _saved_config(counter_threat_waiver_scale=1.0, open_three_score=99.0)
    config = TrainingConfig()

    apply_saved_config(config, saved, reward_from_checkpoint=True)

    assert config.reward.counter_threat_waiver_scale == 1.0
    assert config.reward.open_three_score == 99.0


def test_non_reward_keys_still_follow_the_checkpoint():
    saved = TrainingConfig().to_dict()
    saved["entropy_coef"] = 0.123
    saved["model"]["channels"] = 32
    config = TrainingConfig()

    apply_saved_config(config, saved)

    assert config.entropy_coef == 0.123
    assert config.model.channels == 32


def test_skip_keys_keep_user_settings_on_resume():
    saved = TrainingConfig().to_dict()
    for key in RESUME_SKIP_KEYS:
        assert key in saved, f"{key} 不在 TrainingConfig 里，skip_keys 已过期"
    saved["epochs"] = 10
    saved["device"] = "cpu"
    config = TrainingConfig(epochs=999, device="cuda")

    apply_saved_config(config, saved)

    assert config.epochs == 999
    assert config.device == "cuda"


def test_reward_fields_missing_from_old_checkpoint_fall_back_to_defaults():
    saved = TrainingConfig().to_dict()
    del saved["reward"]["counter_threat_waiver_scale"]
    config = TrainingConfig()

    apply_saved_config(config, saved, reward_from_checkpoint=True)

    assert config.reward.counter_threat_waiver_scale == RewardConfig().counter_threat_waiver_scale


def test_diff_reward_config_reports_changed_and_added_fields():
    saved = TrainingConfig().to_dict()["reward"]
    saved["open_three_score"] = 99.0
    del saved["counter_threat_waiver_scale"]

    changed, added = diff_reward_config(saved, RewardConfig())

    assert any("open_three_score" in item for item in changed)
    assert any("counter_threat_waiver_scale" in item for item in added)


def test_diff_reward_config_is_empty_for_identical_configs():
    saved = TrainingConfig().to_dict()["reward"]

    changed, added = diff_reward_config(saved, RewardConfig())

    assert changed == []
    assert added == []


def _tiny_model() -> PolicyValueNet:
    return PolicyValueNet(board_size=9, channels=8, blocks=1)


def _run_cosine(
    base_lr: float, eta_min: float, t_max: int, steps: int
) -> tuple[PolicyValueNet, torch.optim.Optimizer]:
    """跑完 steps 步余弦退火，返回配对的 (model, optimizer)。"""
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    for _ in range(steps):
        optimizer.step()
        scheduler.step()
    return model, optimizer


def _save_checkpoint(path, optimizer, model, epoch: int) -> None:
    config = TrainingConfig(board_size=9, model=ModelConfig(channels=8, blocks=1))
    torch.save(
        {
            "epoch": epoch,
            "config": config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


@pytest.mark.parametrize("steps", [0, 1, 37, 600])
def test_cosine_lr_at_matches_torch_scheduler(steps):
    base_lr, eta_min, t_max = 3.5e-4, 1.5e-5, 600
    _, optimizer = _run_cosine(base_lr, eta_min, t_max, steps)

    expected = optimizer.param_groups[0]["lr"]

    assert cosine_lr_at(base_lr, eta_min, t_max, steps) == pytest.approx(expected, rel=1e-9)


def test_resuming_from_bc_checkpoint_restores_configured_learning_rate(tmp_path):
    # five-pretrain 存档时 lr 停在预训练余弦的谷底，且 checkpoint 里 epoch=0。
    model, optimizer = _run_cosine(1e-3, 1e-5, 30, 30)
    assert optimizer.param_groups[0]["lr"] < 2e-5
    checkpoint = tmp_path / "best_bc.pt"
    _save_checkpoint(checkpoint, optimizer, model, epoch=0)

    config = TrainingConfig(
        board_size=9, device="cpu", runs_dir=str(tmp_path / "runs"),
        model=ModelConfig(channels=8, blocks=1),
    )
    trainer = PPOTrainer(config, checkpoint_path=str(checkpoint))

    # PPO 必须从自己配置的学习率起步，而不是继承预训练余弦的谷底。
    assert trainer._start_epoch == 1
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(config.learning_rate, rel=1e-9)
    assert trainer.optimizer.param_groups[0]["initial_lr"] == pytest.approx(config.learning_rate)


def test_resume_honours_learning_rate_override(tmp_path):
    model, optimizer = _run_cosine(3.5e-4, 1.5e-5, 600, 100)
    checkpoint = tmp_path / "epoch_100.pt"
    _save_checkpoint(checkpoint, optimizer, model, epoch=100)

    config = TrainingConfig(
        board_size=9, device="cpu", runs_dir=str(tmp_path / "runs"),
        model=ModelConfig(channels=8, blocks=1),
    )
    config.learning_rate = 1e-4
    trainer = PPOTrainer(config, checkpoint_path=str(checkpoint))

    expected = cosine_lr_at(1e-4, config.lr_min, config.epochs, 100)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-9)


def test_resume_without_override_keeps_the_existing_curve(tmp_path):
    model, optimizer = _run_cosine(3.5e-4, 1.5e-5, 600, 100)
    expected = cosine_lr_at(3.5e-4, 1.5e-5, 600, 100)
    checkpoint = tmp_path / "epoch_100.pt"
    _save_checkpoint(checkpoint, optimizer, model, epoch=100)

    config = TrainingConfig(
        board_size=9, device="cpu", runs_dir=str(tmp_path / "runs"),
        model=ModelConfig(channels=8, blocks=1),
    )
    trainer = PPOTrainer(config, checkpoint_path=str(checkpoint))

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-9)


def test_historical_opponent_prob_no_longer_starves_self_play():
    """回归：0.70 + 0.40 的全局阈值是 1.10 > roll 上界，曾导致自博弈占比恒为 0。"""
    kinds = [
        resolve_opponent_kind(roll / 1000, 0.70, 0.40, has_historical=True)
        for roll in range(1000)
    ]

    assert kinds.count("heuristic") == 700
    # 剩余 30% 中 40% 给历史对手，其余留给当前策略自博弈
    assert kinds.count("historical") == 120
    assert kinds.count("self") == 180


def test_opponent_split_matches_legacy_semantics_when_no_heuristic():
    """启发式占比为 0 时（训练早期），与原本的 0.4/0.6 语义完全一致。"""
    kinds = [
        resolve_opponent_kind(roll / 1000, 0.0, 0.40, has_historical=True)
        for roll in range(1000)
    ]

    assert kinds.count("heuristic") == 0
    assert kinds.count("historical") == 400
    assert kinds.count("self") == 600


def test_empty_opponent_pool_falls_back_to_self_play():
    assert resolve_opponent_kind(0.99, 0.70, 0.40, has_historical=False) == "self"
    assert resolve_opponent_kind(0.10, 0.70, 0.40, has_historical=False) == "heuristic"


def test_self_play_share_stays_positive_across_the_whole_heuristic_ramp():
    """任何启发式占比下都必须给当前策略留出份额，否则「自博弈」名不副实。"""
    for percent in range(0, 100):
        heuristic_prob = percent / 100
        kinds = [
            resolve_opponent_kind(roll / 1000, heuristic_prob, 0.40, has_historical=True)
            for roll in range(1000)
        ]
        assert kinds.count("self") > 0, f"self-play starved at heuristic_prob={heuristic_prob}"


def test_historical_prob_of_one_still_leaves_no_self_play_only_when_asked():
    """historical_prob=1.0 表示「非启发式局全给历史对手」，这是显式选择。"""
    kinds = [
        resolve_opponent_kind(roll / 1000, 0.50, 1.0, has_historical=True)
        for roll in range(1000)
    ]

    assert kinds.count("self") == 0
    assert kinds.count("historical") == 500


def _metrics_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_best_epoch_falls_back_to_legacy_weights_without_anchor_column():
    """旧 run 的 metrics.csv 没有锚点列，绿线位置必须与从前一致。"""
    frame = _metrics_frame([
        {"epoch": 1, "eval_win_rate_heuristic": 0.9, "eval_win_rate_random": 0.5,
         "value_loss": 0.1, "entropy": 1.2},
        {"epoch": 2, "eval_win_rate_heuristic": 0.2, "eval_win_rate_random": 1.0,
         "value_loss": 0.1, "entropy": 1.2},
    ])

    # 5*0.9 + 2*0.5 = 5.5  >  5*0.2 + 2*1.0 = 3.0
    assert compute_best_epoch(frame) == 1


def test_anchor_win_rate_can_outweigh_the_training_opponent():
    """专门适应启发式、但相对起点毫无长进的检查点，不应再被选为 best。"""
    frame = _metrics_frame([
        # 启发式胜率高，但打不过自己的起点（没有真实进步）
        {"epoch": 1, "eval_win_rate_heuristic": 0.95, "eval_win_rate_random": 0.9,
         "eval_win_rate_anchor": 0.30, "value_loss": 0.1, "entropy": 1.2},
        # 启发式胜率略低，但明显强于起点
        {"epoch": 2, "eval_win_rate_heuristic": 0.75, "eval_win_rate_random": 0.9,
         "eval_win_rate_anchor": 0.85, "value_loss": 0.1, "entropy": 1.2},
    ])

    # 3*0.95 + 3*0.30 + 0.9 = 4.65   vs   3*0.75 + 3*0.85 + 0.9 = 5.70
    assert compute_best_epoch(frame) == 2


def test_anchor_column_present_but_empty_is_treated_as_absent():
    frame = _metrics_frame([
        {"epoch": 1, "eval_win_rate_heuristic": 0.9, "eval_win_rate_random": 0.5,
         "eval_win_rate_anchor": float("nan"), "value_loss": 0.1, "entropy": 1.2},
        {"epoch": 2, "eval_win_rate_heuristic": 0.2, "eval_win_rate_random": 1.0,
         "eval_win_rate_anchor": float("nan"), "value_loss": 0.1, "entropy": 1.2},
    ])

    assert compute_best_epoch(frame) == 1


def test_resuming_with_epochs_below_checkpoint_raises_instead_of_doing_nothing(tmp_path):
    """回归：epochs 是总轮数，小于 checkpoint 轮数时循环为空，此前会静默「成功」。"""
    model, optimizer = _run_cosine(3.5e-4, 1.5e-5, 600, 80)
    checkpoint = tmp_path / "epoch_080.pt"
    _save_checkpoint(checkpoint, optimizer, model, epoch=80)

    config = TrainingConfig(
        board_size=9, device="cpu", epochs=60, runs_dir=str(tmp_path / "runs"),
        model=ModelConfig(channels=8, blocks=1),
    )
    trainer = PPOTrainer(config, checkpoint_path=str(checkpoint))

    with pytest.raises(ValueError, match="总轮数"):
        trainer.train()
