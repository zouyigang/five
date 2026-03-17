"""与训练图绿线（best epoch）一致的 best 判定逻辑，供 trainer 与 metrics_panel 共用。"""

from __future__ import annotations

import pandas as pd


def _rolling_window(frame: pd.DataFrame) -> int:
    return max(5, min(25, len(frame) // 20 if len(frame) >= 20 else len(frame)))


def compute_best_epoch(frame: pd.DataFrame) -> int | None:
    """使用与 PPO 微调图绿线相同的公式计算 best epoch。"""
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
    avg_length = pd.to_numeric(frame["avg_game_length"], errors="coerce") if "avg_game_length" in frame.columns else pd.Series([81.0] * len(frame))
    if heuristic.notna().sum() == 0 or random_wr.notna().sum() == 0:
        return None

    window = _rolling_window(frame)
    h_smooth = heuristic.rolling(window=window, min_periods=1).mean().fillna(-1.0)
    r_smooth = random_wr.rolling(window=window, min_periods=1).mean().fillna(-1.0)
    vl_smooth = value_loss.rolling(window=window, min_periods=1).mean().fillna(float("inf"))
    ent = entropy.fillna(0.0) if entropy is not None else pd.Series([0.0] * len(frame))
    length = avg_length.fillna(81.0) if avg_length is not None else pd.Series([81.0] * len(frame))

    ent_max = max(ent.max(), 1e-8)
    length_max = max(length.max(), 1.0)
    vl_max = max(vl_smooth.max(), 1e-8)

    scores = (
        h_smooth * 4.0
        + r_smooth * 2.0
        + (ent / ent_max) * 1.0
        - (vl_smooth / vl_max) * 0.5
        - (length / length_max) * 0.5
    )
    best_idx = int(scores.idxmax())
    return int(epoch.iloc[best_idx])
