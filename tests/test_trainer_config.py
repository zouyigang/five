import pytest
import torch

from five.ai.model import PolicyValueNet
from five.common.config import ModelConfig, RewardConfig, TrainingConfig
from five.train.trainer import (
    RESUME_SKIP_KEYS,
    PPOTrainer,
    apply_saved_config,
    cosine_lr_at,
    diff_reward_config,
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
