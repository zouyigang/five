# Five – 五子棋强化学习系统

一个从零开始的五子棋强化学习项目，支持：
- **行为克隆预训练**（从启发式专家策略中模仿学习，提供强初始策略）
- **PPO 自博弈训练**（在预训练基础上持续强化学习）
- **训练过程可视化**（损失曲线、胜率、对局长度等 10+ 指标实时监控）
- **每一局对局的逐手回放与步级详情**
- **人机对弈**（使用训练好的模型与人类对弈）
- **奖励检验**（落子即算奖励明细：进攻/防守/漏防/错失赢棋点等）

## 功能概览

### 自博弈训练
- 使用 PPO 风格的自博弈训练，可从零开始或从预训练模型继续。
- 支持配置棋盘大小（默认 `9×9`）和训练轮数。
- 训练过程中自动生成 checkpoint 和评估指标。
- **课程学习**：从中期开始逐步引入启发式对手（HeuristicPlayer），打破纯自博弈的策略趋同，提升对"会防守、会进攻"对手的泛化能力。

### 行为克隆预训练（Behavioral Cloning）
- 先用启发式对手（HeuristicPlayer）自博弈生成数万局专家数据。
- 用交叉熵损失训练策略头（模仿专家落子）+ MSE 损失训练价值头（拟合胜负结果）。
- 产出的 checkpoint 与 PPO 格式一致，可直接用 `five-train --checkpoint` 加载后做 PPO 微调。
- **好处**：跳过"从随机策略摸索"的低效阶段，模型一开始就具备基本的攻防意识和中心开局偏好。

### 训练可视化
- 每个训练 run 在 `runs/<run_id>/` 下生成：
  - `metrics.csv`：每个 epoch 的训练损失、熵、平均对局长度、对基线对手的胜率等。
  - `games/*.json`：每一局的完整对局记录（含每步的落点、价值估计、候选落点 Top-K 等）。
- Tkinter GUI 中提供：
  - PPO 微调页（折线图实时展示训练指标，标出 **best epoch** 与 **anomaly epochs**）。
  - 对局回放页（逐手回放，查看当步候选落点热力信息和价值估计）

### 奖励检验
- GUI 中的奖励检验页可以：
  - 在空盘或任意局面下点击落子，即时查看该步的奖励明细。
  - 明细包含：形成活四/冲四/活三/跳活三/眠三、封堵对方活四/冲四/活三等、未化解对方威胁、错失直接获胜落点等；终局步会显示完整奖励（含终局获胜奖励）。
  - 支持新对局、按钮回退和棋盘右键快速回退；回退后当前玩家会正确恢复。

### 人机对弈
- GUI 中的人机对弈页可以：
  - 选择某个训练 run 下的模型 checkpoint。
  - 选择人类执黑/执白。
  - 选择 AI 思考模式：**固定**（温度 0）/ **稳健**（温度 0.1）/ **标准**（温度 0.2）/ **探索**（温度 0.6）。
  - 对弈结束后自动生成 `GameRecord`，可在回放页复盘。

## 安装与环境

### 1. 基础环境

- Python 版本：**3.11+**
- 推荐使用独立虚拟环境（`venv`、`conda` 等）。
- 本项目已在 Windows 11、Python 3.14.6、RTX 5080、PyTorch 2.13.0+cu130 下验证。

创建虚拟环境前先运行 `python --version`，确认当前解释器符合要求。下面的 `python -m venv .venv` **只在 `.venv` 不存在时执行一次**；如果 `.venv` 已存在，请跳过创建命令并直接激活。不要使用另一个 Python 版本覆盖已有环境。

Windows PowerShell：

```powershell
cd E:\PycharmProjects\five
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
```

Windows CMD：

```cmd
cd /d E:\PycharmProjects\five
if not exist .venv python -m venv .venv
.\.venv\Scripts\activate.bat
```

> `Activate.ps1` 只能在 PowerShell 中执行；CMD 请使用 `activate.bat`。激活成功后，命令提示符前会显示 `(.venv)`。

Linux/macOS：

```bash
[ -d .venv ] || python -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖（推荐可编辑安装）

安装项目和测试依赖：

```powershell
python -m pip install -e ".[test]"
```

PyPI 默认安装的 PyTorch 可能不包含 CUDA。使用 NVIDIA GPU 训练时，请按显卡驱动和操作系统选择
[PyTorch 官方安装源](https://pytorch.org/get-started/locally/)。本机已验证的 CUDA 13.0 安装命令为：

```powershell
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130
```

安装后验证环境：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m pytest -q
```

CPU 环境可以正常运行 GUI、测试和小规模训练，但训练命令必须显式传入 `--device cpu`。项目默认训练设备为 `cuda`。

安装会注册五个命令行入口：

| 命令 | 说明 |
|------|------|
| `five-train` | PPO 自博弈训练 |
| `five-generate` | 生成启发式自博弈数据集 |
| `five-pretrain` | 行为克隆预训练 |
| `five-export-human-games` | 导出带人工标注的对局训练数据 |
| `five-gui` | 启动 GUI（数据生成 / 行为克隆预训练 / PPO 微调 / 回放 / 人机对弈 / 奖励检验） |

核心依赖：`torch>=2.2`、`numpy>=1.26`、`pandas>=2.2`、`matplotlib>=3.8`。

## 快速开始

以下命令假设已经进入项目根目录并激活虚拟环境，无需在每个步骤中重复激活。

### 方式 A：从零开始 PPO 训练

```bash
five-train --board-size 9 --epochs 600 --games-per-epoch 384 --run-name ppo_gomoku
```

### 方式 B：推荐流程 — 行为克隆预训练 + PPO 微调

推荐使用三阶段训练，模型棋力提升更快：

**第 1 步：生成启发式自博弈数据**

```bash
five-generate --games 50000 --board-size 9 --output data/heuristic_50k.pt
```

HeuristicPlayer 会自动对弈 50000 局，每步根据棋形评分选择落子，通常可生成数百万条 `(state, action, value)` 样本。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--games` | 50000 | 自博弈局数 |
| `--board-size` | 9 | 棋盘大小 |
| `--win-length` | 5 | 连珠数 |
| `--output` | `data/heuristic_50k.pt` | 输出文件路径 |
| `--seed` | 42 | 随机种子 |

**第 2 步：行为克隆预训练**

```bash
five-pretrain --dataset data/heuristic_50k.pt --epochs 30 --lr 1e-3 --output-dir pretrain_output
```

用交叉熵损失模仿启发式专家的落子选择，同时用 MSE 拟合胜负结果。训练结束后在 `pretrain_output/` 生成 `best_bc.pt`（验证最优）和 `final_bc.pt`（最终轮次）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | *必填* | 第 1 步生成的 `.pt` 文件 |
| `--board-size` | 9 | 棋盘大小 |
| `--channels` | 256 | 网络通道数（须与 PPO 配置一致） |
| `--blocks` | 16 | 残差块数量（须与 PPO 配置一致） |
| `--epochs` | 30 | 预训练轮数 |
| `--batch-size` | 1024 | 批大小 |
| `--lr` | 1e-3 | 初始学习率 |
| `--value-coef` | 0.5 | 价值损失权重 |
| `--device` | `cuda` | 训练设备 |
| `--output-dir` | `pretrain_output` | 输出目录 |

**第 3 步：PPO 微调**

```bash
five-train --checkpoint pretrain_output/best_bc.pt --epochs 600 --run-name ppo_from_bc
```

从预训练的模型继续 PPO 自博弈训练，保留行为克隆学到的攻防意识，再通过强化学习持续提升。

### 启动 GUI

```bash
five-gui
```

GUI 包含六个标签页：
- **数据生成**：读取生成进度文件，查看胜负统计和逐局棋谱。
- **行为克隆预训练**：读取预训练进度文件，展示损失、准确率和学习率曲线。
- **PPO 微调**：选择某个 run 后实时展示 policy/value loss、entropy、平均对局长度、评估胜率等 10 项指标。
- **对局回放**：选择 run 和 game，逐手播放，右侧展示步级详情（执子方、落点、动作概率、价值估计、候选落点 Top-K、奖励明细）。
- **人机对弈**：选择 checkpoint、执子颜色和 AI 思考模式，与训练好的模型对弈。
- **奖励检验**：落子即算，实时查看单步奖励明细。

> 注意：GUI 使用 Tkinter + matplotlib，推荐在本地桌面环境下运行。

## 项目结构

```text
five/
├── pyproject.toml          # 包配置与 CLI 入口注册
├── README.md
├── src/five/
│   ├── core/               # 棋盘、规则、对局状态与回放
│   │   ├── board.py         # Board 类：棋盘表示、合法手生成、胜负判定
│   │   ├── game.py          # GomokuGame：创建/推进对局状态
│   │   ├── move.py          # Move 数据类：(row, col) ↔ index 转换
│   │   ├── rules.py         # 方向常量、边界检查
│   │   ├── state.py         # GameState：完整对局状态（含历史、当前玩家、胜者）
│   │   └── replay.py        # 对局回放工具
│   ├── ai/                  # 策略/价值网络、推理接口、玩家封装
│   │   ├── model.py         # PolicyValueNet：残差卷积网络（策略头 + 价值头）
│   │   ├── encoder.py       # encode_state：4 通道状态编码
│   │   ├── inference.py     # ModelAIEngine：模型推理封装
│   │   ├── interfaces.py    # AIEngine Protocol、AnalysisResult 等接口
│   │   ├── players.py       # HeuristicPlayer、RandomPlayer、EnginePlayer
│   │   └── mcts.py          # MCTS 预留接口（AlphaZero 升级路径）
│   ├── train/               # 训练管线
│   │   ├── imitation_data.py # 启发式自博弈数据生成（five-generate）
│   │   ├── pretrain.py      # 行为克隆预训练（five-pretrain）
│   │   ├── trainer.py       # PPOTrainer：PPO 自博弈训练循环（five-train）
│   │   ├── self_play.py     # play_self_play_game：对局执行与奖励回填
│   │   ├── reward.py        # 奖励函数：进攻/防守/漏防/错失赢棋点/开局位置
│   │   ├── evaluator.py     # evaluate_policy：模型对随机/启发式对手胜率评估
│   │   ├── dataset.py       # Transition、EpisodeBatch 数据结构
│   │   ├── events.py        # TrainEvent 事件定义
│   │   └── run_manager.py   # RunArtifacts：run 目录、存储管理
│   ├── storage/             # 持久化
│   │   ├── schemas.py       # GameRecord、MoveRecord、MetricRecord 等数据结构
│   │   ├── game_store.py    # 对局记录读写
│   │   ├── metric_store.py  # 训练指标 CSV 读写
│   │   ├── checkpoint_store.py # 模型 checkpoint 存储
│   │   └── model_registry.py  # 模型注册表
│   ├── gui/                 # Tkinter GUI
│   │   ├── app.py           # FiveApp 主应用
│   │   ├── controllers.py   # GUI 控制逻辑
│   │   ├── viewmodels.py    # 视图模型
│   │   ├── pages/           # 六个功能页
│   │   │   ├── generate_page.py  # 数据生成进度与棋谱
│   │   │   ├── pretrain_page.py  # 行为克隆预训练监控
│   │   │   ├── train_page.py     # PPO 微调监控
│   │   │   ├── replay_page.py    # 对局回放
│   │   │   ├── versus_ai_page.py # 人机对弈
│   │   │   └── reward_test_page.py # 奖励检验
│   │   └── widgets/         # 可复用组件
│   │       ├── board_canvas.py    # 棋盘画布
│   │       ├── metrics_panel.py   # 训练指标面板
│   │       └── move_detail_panel.py # 步级详情面板
│   └── common/              # 通用工具
│       ├── config.py        # RewardConfig、ModelConfig、TrainingConfig、GUIConfig
│       ├── logging.py       # 日志配置
│       └── utils.py         # 工具函数（seed、JSON 读写、时间戳等）
├── tests/
│   ├── test_core.py         # 棋盘/规则/回放测试
│   ├── test_reward.py       # 奖励函数测试
│   └── test_threat_shapes.py # 棋形检测测试
├── data/                    # 预训练数据（生成后出现）
│   └── heuristic_50k.pt
├── pretrain_output/         # 预训练 checkpoint（训练后出现）
│   ├── best_bc.pt
│   └── final_bc.pt
└── runs/                    # PPO 训练产物
    └── <run_id>/
        ├── config.json
        ├── metrics.csv
        ├── models.json
        ├── games/
        │   └── game_*.json
        └── checkpoints/
            └── epoch_*.pt
```

## 完整训练流程

```text
┌─────────────────────────────────────────────────────────┐
│ 第 1 阶段：数据生成                                       │
│                                                         │
│   five-generate --games 50000                           │
│       HeuristicPlayer vs HeuristicPlayer                │
│       输出: data/heuristic_50k.pt                        │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 第 2 阶段：行为克隆预训练                                  │
│                                                         │
│   five-pretrain --dataset data/heuristic_50k.pt         │
│       CrossEntropy(policy) + MSE(value)                 │
│       输出: pretrain_output/best_bc.pt                   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 第 3 阶段：PPO 强化学习微调                                │
│                                                         │
│   five-train --checkpoint pretrain_output/best_bc.pt    │
│       自博弈 + 课程学习（历史对手 + 启发式对手）             │
│       输出: runs/<run_id>/checkpoints/epoch_*.pt         │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 使用与评估                                               │
│                                                         │
│   five-gui                                              │
│       数据生成 │ 预训练 │ PPO 微调 │ 回放 │ 人机 │ 奖励检验    │
└─────────────────────────────────────────────────────────┘
```

## 训练配置详解

所有配置集中在 `src/five/common/config.py`，分为六个 dataclass：

### RewardConfig — 奖励函数参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `attack_scale` | 0.03 | 进攻奖励全局缩放系数 |
| `block_scale` | 0.035 | 防守奖励全局缩放系数 |
| `max_process_reward` | 2.5 | 单步过程奖励裁剪上限 |
| `max_total_reward` | 5.0 | 总奖励裁剪上限 |
| `opening_position_horizon` | 36 | 开局位置塑形生效步数 |
| `opening_center_bonus` | 0.12 | 开局中心落子奖励 |
| `opening_edge_penalty` | 0.8 | 开局边线落子惩罚 |
| `opening_corner_penalty` | 1.2 | 开局角落落子惩罚 |
| `opening_center_radius_ratio` | 0.28 | 中心区域半径比例 |
| `edge_shape_decay` | 0.4 | 边线棋形价值折减 |
| `corner_shape_decay` | 0.25 | 角落棋形价值折减 |
| `final_win_reward` | 3.0 | 获胜奖励 |
| `outcome_tail_bonus` | 0.3 | 终局结果回传基础值 |
| `outcome_decay` | 0.85 | 终局回传衰减系数 |
| `outcome_horizon` | 6 | 终局回传覆盖步数 |
| `miss_immediate_win_penalty` | 2.8 | 未阻止对方制胜手惩罚 |
| `miss_own_immediate_win_penalty` | 2.5 | 错失己方直接获胜惩罚 |
| `miss_own_open_four_penalty` | 2.0 | 错失己方形成活四机会的惩罚 |
| `miss_open_three_penalty` | 2.2 | 未压制对方一步成活四机会的惩罚 |
| `miss_jump_open_three_penalty` | 2.0 | 未压制对方跳活三机会的惩罚 |
| `miss_one_move_four_three_penalty` | 2.2 | 未阻止对方一步形成四三的惩罚 |
| `miss_one_move_double_three_penalty` | 2.0 | 未阻止对方一步形成双活三的惩罚 |
| `counter_threat_waiver_scale` | 0.5 | 本手成活三/跳活三时，对「一手成四三/双活三」漏防惩罚的折减系数（1.0 = 不折减） |
| `rush_four_waiver_attack_scale` | 0.3 | 冲四靠强攻豁免免掉漏防惩罚时，其进攻分的折减系数（1.0 = 不折减） |

### ModelConfig — 网络架构

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `channels` | 256 | 卷积通道数 |
| `blocks` | 16 | 残差块数量 |

### TrainingConfig — PPO 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `board_size` | 9 | 棋盘大小 |
| `win_length` | 5 | 连珠数 |
| `self_play_games_per_epoch` | 384 | 每轮自博弈局数 |
| `epochs` | 600 | 总训练轮数 |
| `batch_size` | 768 | PPO 更新批大小 |
| `updates_per_epoch` | 6 | 每轮 PPO 更新次数 |
| `learning_rate` | 3.5e-4 | 学习率 |
| `lr_min` | 1.5e-5 | 余弦退火最低学习率 |
| `grad_clip_norm` | 1.5 | 梯度裁剪 |
| `gamma` | 0.97 | GAE 折扣因子 |
| `gae_lambda` | 0.95 | GAE λ |
| `clip_epsilon` | 0.25 | PPO clip 范围 |
| `value_clip_epsilon` | 0.25 | Value clip 范围 |
| `value_coef` | 0.6 | Value 损失系数 |
| `entropy_coef` | 0.03 | 熵正则系数 |
| `temperature_init` | 1.3 | 初始采样温度 |
| `temperature_min` | 0.35 | 最低采样温度 |
| `historical_opponent_prob` | 0.4 | 历史对手概率 |
| `opponent_pool_size` | 80 | 对手池大小 |
| `heuristic_opponent_max_prob` | 0.70 | 启发式对手在调度峰值时的单局采样概率 |
| `heuristic_start_fraction` | 0.0 | 启发式占比开始爬升前跳过的轮次比例（0 表示从第 1 轮就有） |
| `heuristic_ramp_fraction` | 0.08 | 从零爬升到峰值所经历的轮次比例 |
| `eval_games` | 48 | 每轮评估局数 |
| `checkpoint_every` | 20 | checkpoint 保存间隔 |

### GUIConfig — 界面参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_width` | 1300 | 窗口宽度 |
| `window_height` | 800 | 窗口高度 |
| `poll_interval_ms` | 2000 | 轮询间隔 |

### GenerateConfig — 启发式数据生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `games` | 50000 | 自博弈局数 |
| `board_size` | 9 | 棋盘大小 |
| `win_length` | 5 | 连珠数 |
| `output` | `data/heuristic_50k.pt` | 数据集输出路径 |
| `seed` | 42 | 随机种子 |

### PretrainConfig — 行为克隆预训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dataset` | `data/heuristic_50k.pt` | 输入数据集 |
| `channels` | 256 | 卷积通道数 |
| `blocks` | 16 | 残差块数量 |
| `epochs` | 30 | 训练轮数 |
| `batch_size` | 1024 | 批大小 |
| `lr` | 1e-3 | 初始学习率 |
| `device` | `cuda` | 训练设备 |

## 命令行参考

### `five-train` — PPO 训练

```bash
five-train [--board-size 9] [--epochs 600] [--games-per-epoch 384]
           [--batch-size 768] [--run-name ppo_gomoku]
           [--device cuda] [--checkpoint PATH]
           [--heuristic-max-prob 0.70] [--heuristic-start-fraction 0]
           [--heuristic-ramp-fraction 0.08]
```

### `five-generate` — 生成启发式数据

在终端后台运行生成，GUI 的「数据生成」页会按进度文件实时刷新进度，并支持分页查看每局详情。

```bash
five-generate [--games 50000] [--board-size 9] [--win-length 5]
              [--output data/heuristic_50k.pt]
              [--progress-file PATH] [--games-detail-file PATH] [--seed 42]
```

- 未指定时，进度写入 `<output>.progress.json`，每局详情追加到 `<output>.games.jsonl`。GUI 默认读取与 `config` 中 `output` 对应的进度文件路径。

### `five-pretrain` — 行为克隆预训练

在终端后台运行预训练，GUI 的「行为克隆预训练」页会按进度文件实时刷新指标与训练曲线。

```bash
five-pretrain --dataset PATH [--board-size 9] [--channels 256] [--blocks 16]
              [--epochs 30] [--batch-size 1024] [--lr 1e-3]
              [--value-coef 0.5] [--device cuda] [--output-dir pretrain_output]
              [--progress-file PATH]
```

- 未指定时，进度写入 `<output-dir>/pretrain.progress.json`。GUI 默认读取 `config` 中 `output_dir` 下的该文件。

### `five-gui` — 启动 GUI

```bash
five-gui
```

### `five-export-human-games` — 导出人工标注数据

```bash
five-export-human-games --run-dir runs/<run_id> [--output data/human_marked.pt]
```

## 训练数据与对局记录格式

- **训练指标**：`runs/<run_id>/metrics.csv`
  - 每行一轮（epoch），包含：`epoch`、`games`、`policy_loss`、`value_loss`、`entropy`、`grad_norm`、`return_mean`、`return_std`、`return_abs_max`、`avg_game_length`、`eval_win_rate_random`、`eval_win_rate_heuristic`、`opening_edge_rate`、`opening_corner_rate`、`opening_center_rate`、`policy_topk_edge_rate`。

- **对局记录**：`runs/<run_id>/games/<game_id>.json`
  - 顶层字段：棋盘大小、胜负结果、总步数、黑/白方类型（model/heuristic/historical）、模型 checkpoint 等。
  - `moves[]` 数组每项为 `MoveRecord`：`move_index`、`player`（1=黑, -1=白）、`row`、`col`、`action_probability`、`value_before`、`legal_count`、`policy_topk`、`total_reward`、`reward_details`。

## 奖励系统

训练使用 PPO 塑形奖励 + 终局奖励，引导模型既重视当前棋形又重视胜负。设计原则：**赢棋奖励为最大正向信号**，过程惩罚适度，避免"消极防守"压倒"主动赢棋"。

### 进攻奖励（本手形成的棋形）

按本手落子后**新形成**的棋形给分，乘以 `attack_scale`。按**最强主棋形**计分：

| 棋形 | 基础分 | 说明 |
|------|--------|------|
| 成五 | 100 | 直接获胜（由终局奖励体现） |
| 活四 | 65 | 四子连、两端空 |
| 双四 | 75 | 一手形成两个四 |
| 冲四活三/四三 | 60 | 同时形成四与活三 |
| 双活三 | 55 | 一手形成两个活三 |
| 冲四/跳四 | 20 | 四子连或跳四、一端堵 |
| 活三 | 10 | 三子连、两端空 |
| 跳活三 | 7 | 三子跳、两端空 |
| 眠三 | 3 | 三子连、一端堵 |

**棋盘边线视为被堵**。两端空指两端均为空格且未抵边。

### 防守奖励

只统计对方**已有**的威胁。本手落在可化解威胁的格点上，按棋形类型 × `block_scale` 加分（含封堵对方活四，与其它封堵类一致）。

威胁**按实例计数**，而非"有/无"的布尔标志。同一条线上的同一组棋子只算一个威胁（扫描时会被其中每颗子各识别一次，按"方向 + 棋子集合"归并去重）。因此：

- 对方有两个不相交的活三时，堵掉其中一个能拿到一份封堵奖励，而不是因为"堵完仍有活三"就一分不给；
- 一手同时化解两个威胁，按 `x2` 计分。

### 漏防惩罚

对方已有活四/冲四/活三等威胁，本手未落在化解位置则扣分，扣分额**随未化解的威胁个数线性放大**（未堵两个活三扣两份）。惩罚幅度小于赢棋奖励，避免过度消极防守。

**豁免层级**（区别来自棋理，不是调参）：

| 本手棋型 | 先手性质 | 对漏防惩罚的作用 |
|---|---|---|
| 冲四及以上（`my_strong_attack`） | 绝对先手：唯一挡点，对方必应且无法反击 | **全额豁免** |
| 活三 / 跳活三（`my_counter_threat`） | 相对先手：对方可用一手兼具封堵与反击化解 | 按 `counter_threat_waiver_scale`（0.5）**部分豁免** |

仅能成冲四的活三不计入反击先手——它无法成活四，构不成真正的牵制。

豁免生效时，**光杆冲四的进攻分按 `rush_four_waiver_attack_scale`（0.3）折减**。豁免本身成立，但若同时全额保留进攻分，"远处随手冲四 + 免责"的净收益为正，会成为刷分燃料：模型可以攒一批无用冲四轮流点，每手净赚，把真正的防守决策无限延后。活四/双四/四三/双活三本身近乎制胜，不折减。

### 错失己方赢棋点惩罚

盘面存在**己方**直接成五的落点却未下在其中，触发惩罚。能赢不赢时，同时压制该步的进攻奖励和终局回传 bonus。

### 开局位置塑形

- 仅前 `opening_position_horizon`（36）手内生效。
- 中心趋向奖励 + 边线/角落惩罚（分两层：最外圈全额、次外圈半额）。
- 当存在战术漏防时，中心奖励被压制，仅保留边线/角落惩罚叠加。

### 终局与结果回传

- 致胜一手给 `final_win_reward`（3.0）。
- 最后 `outcome_horizon`（6）步内按距终局的距离做衰减的胜负 bonus。
- 错失直接获胜的步跳过 outcome tail bonus，避免正奖励冲抵惩罚。

### 奖励明细

每一项奖惩都会追加一条 `RewardDetail(amount, reason)`，**明细金额之和恒等于该步总分**。这条不变量由模糊测试覆盖，是奖励检验页与回放页步级详情可信的前提——总分被调整却不写明细（或写了未参与总分的明细）会让审计结果失真。开局位置分在对手有威胁时会被压制，此时明细记录的是**压制后**的金额。

## 训练曲线图解读

GUI「PPO 微调」页展示原始曲线（浅色）+ 滚动均值（深色），标出 **best epoch**（绿色虚线）与 **anomaly epochs**（红色虚线）。

- **Policy Loss**：PPO 策略更新幅度；稳定训练应围绕某区间温和波动。
- **Value Loss**：价值头拟合误差；突升并维持高位意味着价值学习与分布失配。
- **Entropy**：策略随机性；早期高，中后期缓慢下降。
- **Avg Game Length**：对局形态变化；突然跃迁常意味策略漂移。
- **Eval Win Rate**：对随机/启发式对手胜率，最直观的强度指标。
- **Grad Norm**：梯度规模；频繁尖峰是不稳定前兆。
- **Return Mean/Std/Abs Max**：回报统计，用于观察奖励尺度与异常。
- **Opening Position Rates**：开局边线/角落/中心比例。
- **Policy Top-K Edge Rate**：策略候选落点中边线占比。

## 测试

```bash
# 首次安装测试依赖
python -m pip install -e ".[test]"

# 运行全部测试
python -m pytest -q

# 单独运行奖励测试
python -m pytest tests/test_reward.py -v
```

## 后续升级方向（AlphaZero-lite）

当前版本预留了 MCTS 升级路径：
- `five.ai.interfaces.AIEngine` — 统一推理接口
- `five.ai.mcts.MCTSPlayer` — MCTS 预留接口

在保持 `core`、`storage`、`gui` 不变的前提下，可加入基于 `PolicyValueNet` 的 MCTS 搜索器和 AlphaZero 风格训练循环。
