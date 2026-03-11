# Five – 五子棋强化学习系统

一个从零开始的五子棋强化学习项目，支持：
- 自博弈训练（PPO/Actor-Critic）
- 训练过程可视化（损失曲线、胜率、对局长度等）
- 每一局对局的逐手回放与步级详情
- 人机对弈（使用训练好的模型与人类对弈）
- 奖励检验（落子即算奖励明细：进攻/防守/漏防/错失赢棋点等）

## 功能概览

- **自博弈训练**
  - 使用 PPO 风格的自博弈训练，从随机策略开始学习。
  - 支持配置棋盘大小（默认 `9x9`）和训练轮数。
  - 训练过程中自动生成 checkpoint 和评估指标。
  - **课程学习**：从中期开始逐步引入启发式对手（HeuristicPlayer），打破纯自博弈的策略趋同，提升对“会防守、会进攻”对手的泛化能力。

- **训练可视化**
  - 每个训练 run 在 `runs/<run_id>/` 下生成：
    - `metrics.csv`：每个 epoch 的训练损失、熵、平均对局长度、对基线对手的胜率等。
    - `games/*.json`：每一局的完整对局记录（包含每步的落点、价值估计、候选落点 Top-K 等）。
  - Tkinter GUI 中提供：
    - 训练监控页（折线图实时展示训练指标，并标出 **best epoch** 与 **anomaly epochs**，便于选 checkpoint 与排查异常）。
    - 对局回放页（逐手回放，查看当步候选落点热力信息和价值估计）

- **奖励检验**
  - GUI 中的奖励检验页可以：
    - 在空盘或任意局面下点击落子，即时查看该步的奖励明细。
    - 明细包含：形成活四/冲四/活三/跳活三/眠三、延缓对方活四一手、封堵对方冲四/活三等、未化解对方威胁、错失直接获胜落点等；终局步会显示完整奖励（含终局获胜奖励）。
    - 支持新对局、回退；回退后当前玩家会正确恢复，不会多走一步。

- **人机对弈**
  - GUI 中的人机对弈页可以：
    - 选择某个训练 run 下的模型 checkpoint。
    - 选择人类执黑/执白。
    - 选择 AI “固定 / 稳健 / 标准 / 探索”四档思考模式：
      - **固定**：温度 0，始终选择当前概率最大的落点（完全确定）。
      - **稳健**：温度约 0.1，少量随机性，更偏向高概率落点。
      - **标准**：温度约 0.2，中等随机性，兼顾稳健与探索。
      - **探索**：温度约 0.6，更大随机性，更像“试招”。
  - 对弈结束后自动生成一局 `GameRecord`，可在回放页复盘。

## 安装与环境

### 1. 基础环境

- Python 版本：**3.11+**（建议与 `pyproject.toml` 中保持一致）
- 推荐使用虚拟环境（如 `venv`、`conda` 等）。

```bash
cd E:\PycharmProjects\five
python -m venv .venv
.venv\Scripts\activate
```

### 2. 安装依赖（推荐可编辑安装）

```bash
pip install -e .
```

这会安装：
- `torch`（用于模型和训练）
- `numpy`、`pandas`
- `matplotlib`（用于 GUI 中的训练曲线）

并注册两个命令入口：
- `five-train`
- `five-gui`

## 快速开始

### 1. 运行一次最小训练

在项目根目录执行（默认在 GPU 上训练）：

```bash
five-train --board-size 9 --epochs 10 --games-per-epoch 24 --run-name ppo_gomoku
```

说明（所有命令行参数及默认值）：
- `--board-size`：棋盘大小，默认 `9`。首版建议用 `9`，收敛更快，便于观察。
- `--epochs`：训练轮数，默认 `1000`。
- `--games-per-epoch`：每个 epoch 自博弈对局数，默认 `128`。
- `--batch-size`：训练批量大小，默认 `256`。较大的批量大小可以更好地利用 GPU 并行计算能力。
- `--run-name`：本次训练 run 的前缀名，默认 `"ppo_gomoku"`，会体现在目录名中。
- `--device`：训练设备，默认 `"cuda"`；如果有 GPU，并且安装了对应的 CUDA 版本 PyTorch，会自动使用 GPU；如果需要使用 CPU，可以填 `"cpu"`，比如：
- `--checkpoint`：从指定的检查点文件继续训练，默认 `None`。用于恢复之前的训练进度，比如：

  ```bash
  # 使用第一块显卡
  five-train --board-size 9 --epochs 100 --games-per-epoch 64 --run-name ppo_gpu --device cuda

  # 使用第 0 块显卡（和上面等价）
  five-train --device cuda:0

  # 显式指定在 CPU 上训练（覆盖默认的 cuda）
  five-train --device cpu

  # 从检查点继续训练
  five-train --checkpoint runs/ppo_gomoku_20260307_153000/checkpoints/epoch_050.pt --epochs 150
  ```

更多训练与课程学习相关参数（学习率调度、熵系数、对手池、启发式对手比例等）见 `src/five/common/config.py` 中的 `TrainingConfig` 与下方「训练配置与课程学习」小节。

训练结束后，会在 `runs/` 下看到类似：

```text
runs/
  ppo_gomoku_20260307_153000/
    config.json
    metrics.csv
    games/
      game_000001.json
      game_000002.json
      ...
    checkpoints/
      epoch_001.pt
      epoch_002.pt
      ...
    models.json
```

### 2. 启动可视化 GUI

```bash
five-gui
```

GUI 包含四个标签页：
- **训练监控**：选择某个 run 后，实时显示：
  - policy/value loss
  - entropy
  - 平均对局长度
  - 对随机/启发式对手的胜率
- **对局回放**：
  - 先选择 run，再选择具体的 game。
  - 左侧棋盘可以逐手播放、回退。
  - 右侧步级详情展示该步的：执子方、落点、动作概率、价值估计、候选落点 Top-K 列表。
- **人机对弈**：
  - 在“运行”下拉框中选中某个训练 run，在“模型”下拉框中选择某个 checkpoint（如 `epoch_010.pt`）。
  - 选择“人类先手”与否，以及 AI 思考强度；在棋盘上点击落子与模型对弈。
  - 对弈结束后，本局会以 `human_<timestamp>.json` 形式保存在该 run 的 `games/` 目录下。
- **奖励检验**：
  - 在棋盘上落子即可查看该步总奖励与明细（形成活三/眠三、延缓对方活四一手、封堵对方冲四/活三、未阻止对方活四保持双赢点、未化解对方冲四、未压制对方活三、错失直接获胜落点等）；终局步显示完整奖励。
  - 支持新对局、回退；回退后当前玩家与棋盘状态一致。

> 注意：GUI 使用 Tkinter + matplotlib，推荐在本地桌面环境下运行，而不是无头服务器。

## 项目结构

核心目录结构如下：

```text
src/five/
  core/        # 棋盘、规则、对局状态与回放
  ai/          # 策略/价值网络、推理接口、玩家封装（留有 MCTS 升级接口）
  train/       # 自博弈、PPO 训练循环、评估与 run 管理
  storage/     # GameRecord/MetricRecord/Checkpoint/ModelRegistry 持久化
  gui/         # Tkinter GUI：训练监控、对局回放、人机对弈
  common/      # 配置、日志与通用工具
```

与计划中的模块一一对应：
- `core`：完全不依赖训练/GUI，只关心五子棋规则与状态。
- `ai`：`PolicyValueNet`、`ModelAIEngine`、玩家封装、未来的 `MCTSPlayer` 升级点。
- `train`：`self_play.py`、`trainer.py`、`evaluator.py` 等实现 PPO 自博弈训练闭环。
- `storage`：`schemas.py` 中定义 `GameRecord` / `MoveRecord` 等结构，并由 `game_store.py` / `metric_store.py` / `checkpoint_store.py` / `model_registry.py` 负责读写。
- `gui`：`app.py` 启动应用，`pages/` 下为训练监控、对局回放、人机对弈、奖励检验四个页面，`widgets/` 下为棋盘画布、训练指标面板和步级详情面板。

## 训练数据与对局记录格式概览

- **训练指标**：`runs/<run_id>/metrics.csv`
  - 每行一轮（epoch），包含：`epoch`、`games`、`policy_loss`、`value_loss`、`entropy`、`grad_norm`、`return_mean`、`return_std`、`return_abs_max`、`avg_game_length`、`eval_win_rate_random`、`eval_win_rate_heuristic`。其中 `return_*` 为归一化前的原始回报统计。

- **对局记录**：`runs/<run_id>/games/<game_id>.json`
  - 顶层记录对局元信息：棋盘大小、胜负结果、总步数、使用的模型 checkpoint 等。
  - `moves[]` 数组中每一项为一个 `MoveRecord`：
    - `move_index`、`player`（1: 黑棋, -1: 白棋）、`row`、`col`
    - `action_probability`、`value_before`、`legal_count`
    - `policy_topk`：若干 `MoveSummary`（候选落点的行列、得分等）
    - `total_reward`、`reward_details`：该步奖励总分与明细（形成活三、延缓对方活四一手、封堵对方冲四、未化解对方威胁、错失直接获胜落点等）

这些记录既被训练过程消费，也被 GUI 用于对局回放与步级可视化。

## 训练配置与课程学习

- **PPO 与优化**
  - 每 epoch 对当前 batch 的更新次数 `updates_per_epoch = 4`，折扣因子 `gamma = 0.97`，熵系数 `entropy_coef = 0.05`。
  - 学习率采用余弦退火（`CosineAnnealingLR`），从 `learning_rate` 降至 `lr_min`。
  - 价值头输出经 `Tanh` 约束到 [-1,1]，训练时对 returns 做 z-score 归一化并裁剪到 [-1,1]；value loss 使用 PPO 风格的 value clipping（`value_clip_epsilon = 0.2`）。

- **探索与对手**
  - 采样温度从 `temperature_init`（1.0）线性退火到 `temperature_min`（0.5），在总 epoch 的 `temperature_anneal_fraction`（0.8）内完成。
  - 历史对手池：每 `opponent_snapshot_interval` 个 epoch 保存当前策略快照，池大小 `opponent_pool_size`；每局以 `historical_opponent_prob` 概率使用历史对手，否则自博弈。
  - **课程学习（Heuristic 对手）**：
    - `heuristic_opponent_max_prob`（默认 0.25）：最终约 25% 对局使用启发式对手。
    - `heuristic_start_fraction`（默认 0.15）：在训练进度 15% 时开始引入（如 1000 epoch 则从第 150 个 epoch 起）。
    - `heuristic_ramp_fraction`（默认 0.5）：在训练进度 50% 时达到最大比例（如第 500 个 epoch 时 25%）。
  - 每局对手选择顺序：先按当前 epoch 的 heuristic 概率判定是否使用 HeuristicPlayer；否则再按历史对手概率判定是否使用历史快照；否则为纯自博弈。使用 Heuristic 或历史对手时只收集**当前模型一方**的 transitions。

- **评估与指标**
  - 评估对手：随机对手（RandomPlayer）与启发式对手（HeuristicPlayer）。HeuristicPlayer 使用棋形评分（成五/活四/冲四/活三/眠三/活二等）并兼顾己方进攻与对方威胁，强度明显高于纯随机。
  - `metrics.csv` 中的 `return_mean` / `return_std` / `return_abs_max` 为**归一化前的原始回报**统计，便于观察真实奖励尺度与异常检测。

项目根目录下 `tests/` 包含：
- `test_core.py`：棋盘、规则、回放等核心逻辑。
- `test_reward.py`：奖励计算与自博弈奖励回填（进攻/防守/漏防/错失赢棋点、终局 tail bonus）。
- `test_threat_shapes.py`：棋形检测自测（成五、活四、冲四、活三、眠三、跳活三、跳活四、跳冲四，边线视为被堵、贴边判为眠三/冲四，及 open_ends 不重复计数等）。

## 奖励系统

训练使用 PPO 塑形奖励 + 终局奖励，引导模型既重视当前棋形又重视胜负。设计原则：**赢棋奖励为最大正向信号**，过程惩罚适度，避免“消极防守”压倒“主动赢棋”。单步奖励由 **进攻奖励**、**防守奖励**、**漏防惩罚**、**错失己方赢棋点惩罚**、**开局位置塑形** 组成，并做裁剪；终局仅在获胜步给获胜奖励，并在最后若干步叠加衰减的胜负回传（outcome tail bonus）。

### 一、进攻奖励（本手形成的棋形）

按本手落子后**新形成**的棋形给分，统一乘以系数 `attack_scale`（默认 0.03）。若一手同时命中多个描述，当前实现按**最强主棋形**计分，不再把基础项与复合项整包叠加；例如“双活三”只记“双活三”，“冲四活三/四三”只记“四三”。棋形基础分（未乘系数前）如下：

| 棋形 | 基础分 | 说明 |
|------|--------|------|
| 成五 | 100 | 本手直接获胜（实际由终局奖励体现） |
| 活四 | 45 | 四子连、两端空 |
| 双四 | 55 | 一手形成两个四 |
| 冲四活三/四三 | 40 | 一手同时形成四与活三 |
| 双活三 | 35 | 一手形成两个活三 |
| 冲四/跳四 | 20 | 四子连或跳四、一端堵 |
| 活三 | 10 | 三子连、两端空 |
| 跳活三 | 7 | 三子跳、两端空 |
| 眠三 | 3 | 三子连、一端堵 |

**棋形端点约定**：**棋盘边线视为被堵**。即“两端空”指两端均为空格（未抵边），若一端是空格、一端贴边，则判为眠三/冲四等“一端堵”棋形，不判为活三/活四。

实际奖励 = 主棋形基础分 × 数量 × `attack_scale`，再与过程奖励上下限裁剪。

### 二、防守奖励（化解对方**当前盘面已有**的威胁）

只统计对方**已经摆在盘上**的威胁（活四、冲四/跳四、活三、跳活三等），不统计“对方下一手才能形成的威胁”。本手若落在可化解这些威胁的格点上，按化解的**棋形类型**与权重 × `block_scale`（默认 0.025）加分。**活四**一手无法真正封住（对方另一端仍可成五），因此不按“封堵”给高分，而是单独给 **延缓对方活四一手**（`delay_open_four_reward`，默认 +0.1）；堵完后若对方仍保留冲四/跳四制胜点，会叠加“未化解对方冲四/跳四”惩罚。

### 三、漏防惩罚（对方已有威胁未化解）

若对方当前盘面存在活四、冲四、活三等威胁，本手**没有**落在可化解这些威胁的位置，则按未化解的威胁类型扣分。当前默认：`未阻止对方活四保持双赢点 = -0.6`、`未化解对方冲四/跳四 = -0.4`、`未压制对方活三 = -0.1`、`未压制对方跳活三 = -0.05`。若本手只堵住活四一端（活四降为冲四），仍扣“未化解对方冲四/跳四”。惩罚幅度小于赢棋奖励（`final_win_reward = 3.0`），避免模型过度偏向消极防守。

### 四、错失己方赢棋点惩罚

若当前盘面存在**己方**直接成五的落点，但本手**没有**下在其中任一格，则触发 **错失直接获胜落点** 惩罚（默认 -0.8）。保证“该赢不赢”时会被惩罚，但不会大于赢棋奖励。

### 五、开局位置塑形

为减少模型在开局前几手把子随意下到边线或角落，奖励函数加入轻量开局位置先验：

- 仅在**前 12 手**内生效（`opening_position_horizon`，默认 12）。
- **中心趋向**：按到中心距离给出 `opening_center_bonus × centrality²`（默认中心奖励 0.08）。
- **边线**落子惩罚（默认 `opening_edge_penalty = -0.12`），**角落**惩罚更强（默认 `opening_corner_penalty = -0.25`）。
- 中心区域按棋盘尺寸比例定义（`opening_center_radius_ratio`）。

### 六、终局与结果回传

- **终局**：仅在**本手为致胜一手**时，给 `final_win_reward`（默认 **3.0**）；平局可设 `draw_reward`（默认 0）。赢棋为最大正向信号。
- **结果回传**：在**最后若干步**（`outcome_horizon`，默认 12 手）内，按与终局的距离做衰减的胜负 bonus（`outcome_tail_bonus` 默认 0.5，`outcome_decay` 默认 0.92），赢家为正、输家为负。

过程奖励按 `max_process_reward`（默认 1.5）裁剪；加上终局后总奖励按 `max_total_reward`（默认 4.0）裁剪。

## 训练曲线图解读

在 GUI 的“训练监控”页中，会展示一组更适合诊断训练稳定性的曲线图；每张图都同时显示**原始曲线**（浅色）和**滚动均值**（深色），并在所有子图上标出 **best epoch**（绿色虚线）与 **anomaly epochs**（红色虚线），便于挑选 checkpoint 与排查异常。

- **Best epoch**：按综合评分选出。对评估胜率做滑动平滑后，结合启发式胜率、随机胜率、熵（越高越好）、value loss（越低越好）、平均对局长度（越短越好）加权打分，取得分最高的 epoch。避免被单次评估波动误导。
- **Anomaly epochs**：标记可能异常的 epoch，包括：value loss / grad norm / policy loss 突刺；平均对局长度异常变长或变短；熵坍塌（相对滚动中位数骤降）；对随机或启发式胜率大幅回落；以及熵、随机胜率、value loss 的持续恶化趋势。按异常权重排序后取前若干项显示。

曲线图主要包括：

- **Policy Loss**
  - 反映 PPO 策略更新幅度；长期稳定通常表现为围绕某个区间温和波动，而不是持续放大。

- **Value Loss**
  - 反映价值头对回报的拟合误差；若突然跳升并长时间维持高位，通常意味着价值学习与采样分布发生了失配。

- **Entropy（策略熵）**
  - 衡量策略随机性；训练早期希望较高，中后期应缓慢下降而非骤降到极低。

- **Avg Game Length（平均对局长度）**
  - 反映对局形态变化；若从较短对局突然跃迁到很长，常意味着策略发生漂移、双方都不再稳定找到关键杀法。

- **Eval Win Rate（评估胜率）**
  - 同时显示对随机对手与启发式对手的胜率，是最直观的外部强度指标。

- **Grad Norm**
  - 反映一次参数更新前梯度的整体规模；若频繁出现尖峰，通常是训练不稳定的前兆。

- **Return Mean / Std**
  - 用于观察每个 epoch 采样回报的中心和波动范围；`return_std` 若突然放大，往往会带来 value loss 抖动。

- **Return Abs Max**
  - 观察单个 epoch 中回报的极端值大小；若持续抬升，说明奖励尺度或自博弈分布可能正在失控。

综合来看：
- **Policy / Value / Grad Norm** 主要用于判断 PPO 更新是否稳定；
- **Entropy + Avg Game Length** 用于观察策略是否在合理探索并形成组织化攻防；
- **Eval Win Rate + Return 统计** 则有助于识别“表面还能下，但训练分布已经开始漂”的情况。

## 后续升级方向（AlphaZero-lite）

当前版本使用 PPO 自博弈作为教学友好型基线，已经预留了升级路径：
- `five.ai.interfaces.AIEngine`
- `five.ai.mcts.MCTSPlayer`

在保持 `core`、`storage`、`gui` 不变的前提下，可以在 `ai/train` 中加入：
- MCTS 搜索器（基于 `PolicyValueNet` 提供先验和价值评估）。
- AlphaZero 风格的自博弈数据生成与训练 loop。

这将显著提升模型棋力，同时继续复用现有的对局记录与可视化能力。