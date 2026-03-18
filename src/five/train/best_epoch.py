"""与训练图绿线（best epoch）一致的 best 判定逻辑，供 trainer 与 metrics_panel 共用。"""

from __future__ import annotations

import pandas as pd


def _rolling_window(frame: pd.DataFrame) -> int:
    return max(5, min(25, len(frame) // 20 if len(frame) >= 20 else len(frame)))


def compute_best_epoch(frame: pd.DataFrame) -> int | None:
    """计算 best epoch，优先原始启发式胜率，适合实际对弈模型选择。

    评分公式：
        score = heuristic_raw × 5
              + random_raw × 2
              - (value_loss / vl_max) × 0.5
              - k × |entropy - entropy_rolling_mean|

    优先级：
    1. 原始启发式胜率（权重最高）
    2. 原始随机胜率（次高）
    3. 价值损失（越小越好）
    4. 策略熵（偏离滚动均值越多惩罚越大，防止坍塌或过散）
    """
    if frame.empty:
        return None
    for col in ("epoch", "eval_win_rate_heuristic", "eval_win_rate_random"):
        if col not in frame.columns:
            return None
    epoch = pd.to_numeric(frame["epoch"], errors="coerce")
    heuristic = pd.to_numeric(frame["eval_win_rate_heuristic"], errors="coerce")
    random_wr = pd.to_numeric(frame["eval_win_rate_random"], errors="coerce")
    value_loss = pd.to_numeric(frame["value_loss"], errors="coerce") if "value_loss" in frame.columns else pd.Series([float("inf")] * len(frame))
    entropy = pd.to_numeric(frame["entropy"], errors="coerce") if "entropy" in frame.columns else pd.Series([0.0] * len(frame))
    if heuristic.notna().sum() == 0 or random_wr.notna().sum() == 0:
        return None

    h_raw = heuristic.fillna(-1.0)
    r_raw = random_wr.fillna(-1.0)
    vl_raw = value_loss.fillna(float("inf"))
    ent = entropy.fillna(0.0)

    window = _rolling_window(frame)
    ent_rolling_mean = ent.rolling(window=window, min_periods=1).mean()
    vl_max = max(vl_raw.max(), 1e-8)

    k = 0.3
    entropy_penalty = k * (ent - ent_rolling_mean).abs()

    scores = (
        h_raw * 5.0
        + r_raw * 2.0
        - (vl_raw / vl_max) * 0.5
        - entropy_penalty
    )
    best_idx = int(scores.idxmax())
    return int(epoch.iloc[best_idx])


def compute_best_epoch_for_resume(frame: pd.DataFrame) -> int | None:
    """计算适合继续训练的 best epoch，关注训练健康度和潜力。

    评分公式：
        score = entropy_bonus
              - value_loss_penalty
              + trend_bonus
              - anomaly_penalty
              + min_heuristic_bonus

    维度：
    1. 策略熵：目标区间 [1.0, 1.4]，区间内正分，过低（坍塌）或过高（过散）扣分
    2. 价值损失：越小越好
    3. 胜率趋势：稳定或上升给正分，下滑扣分
    4. 训练稳定性：异常点大幅扣分
    5. 基础胜率：启发式胜率 > 阈值才考虑
    """
    if frame.empty:
        return None
    for col in ("epoch", "eval_win_rate_heuristic", "eval_win_rate_random"):
        if col not in frame.columns:
            return None
    epoch = pd.to_numeric(frame["epoch"], errors="coerce")
    heuristic = pd.to_numeric(frame["eval_win_rate_heuristic"], errors="coerce")
    random_wr = pd.to_numeric(frame["eval_win_rate_random"], errors="coerce")
    value_loss = pd.to_numeric(frame["value_loss"], errors="coerce") if "value_loss" in frame.columns else pd.Series([float("inf")] * len(frame))
    entropy = pd.to_numeric(frame["entropy"], errors="coerce") if "entropy" in frame.columns else pd.Series([0.0] * len(frame))
    if heuristic.notna().sum() == 0 or random_wr.notna().sum() == 0:
        return None

    h_raw = heuristic.fillna(-1.0)
    r_raw = random_wr.fillna(-1.0)
    vl_raw = value_loss.fillna(float("inf"))
    ent = entropy.fillna(0.0)

    window = _rolling_window(frame)
    h_rolling_mean = h_raw.rolling(window=window, min_periods=1).mean()
    vl_max = max(vl_raw.max(), 1e-8)

    entropy_low, entropy_high = 1.0, 1.4
    entropy_bonus = pd.Series(0.0, index=frame.index)
    in_range = (ent >= entropy_low) & (ent <= entropy_high)
    entropy_bonus[in_range] = 1.0
    below_range = ent < entropy_low
    above_range = ent > entropy_high
    entropy_bonus[below_range] = -1.0 * (entropy_low - ent[below_range])
    entropy_bonus[above_range] = -0.5 * (ent[above_range] - entropy_high)

    vl_penalty = (vl_raw / vl_max) * 2.0

    trend_bonus = pd.Series(0.0, index=frame.index)
    stable_or_rising = h_raw >= h_rolling_mean * 0.95
    trend_bonus[stable_or_rising] = 1.0
    declining = h_raw < h_rolling_mean * 0.90
    trend_bonus[declining] = -1.0

    anomaly_penalty = pd.Series(0.0, index=frame.index)
    if "is_anomaly" in frame.columns:
        is_anomaly = frame["is_anomaly"].fillna(False).astype(bool)
        anomaly_penalty[is_anomaly] = 5.0

    min_heuristic_threshold = 0.3
    min_heuristic_bonus = pd.Series(0.0, index=frame.index)
    above_threshold = h_raw >= min_heuristic_threshold
    min_heuristic_bonus[above_threshold] = 0.5
    below_threshold = h_raw < min_heuristic_threshold
    min_heuristic_bonus[below_threshold] = -2.0

    scores = (
        entropy_bonus
        - vl_penalty
        + trend_bonus
        - anomaly_penalty
        + min_heuristic_bonus
    )
    best_idx = int(scores.idxmax())
    return int(epoch.iloc[best_idx])
