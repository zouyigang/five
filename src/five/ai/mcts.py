"""PUCT 蒙特卡洛树搜索，作为标准 AIEngine 接入。

与网络直接出手的区别：网络只有「直觉」，无法验证「这手之后会怎样」；搜索能实际
走下去看结果，因此能算出网络看不出的杀棋与必挡。

**跨对局批量求解是可行性的前提。** 单局串行搜索每次模拟都要一次 batch=1 前向
（实测 3.45ms），64 次模拟就是 220ms/手，完全不可用。这里让所有对局的搜索树
同步推进：每一轮模拟中各棵树各自下降到叶子，把所有叶子凑成一批一次前向，再统一
回传。于是 N 局 × S 次模拟只需 S 次 batch=N 的前向。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from five.ai.encoder import encode_state
from five.ai.interfaces import AnalysisResult, CandidateMove
from five.core.move import Move
from five.core.state import GameState


@dataclass(slots=True)
class MCTSConfig:
    simulations: int = 64
    c_puct: float = 1.25
    # 自博弈时在根节点混入 Dirichlet 噪声以保证探索；评估/对弈时设 0 关闭。
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.0
    # 温度采样时只在访问数前 K 的着法中抽取；0 = 不限制。
    # 模拟次数不高时访问计数几乎是平的（64 次模拟摊到约 80 个合法点），温度 1.0
    # 采样近似均匀乱选，会抽到角和边这种明显坏手。限制到前 K 可在「搜索认可的着法」
    # 内部取多样性，而不是在全盘乱选。
    sample_top_k: int = 0


class _Node:
    """搜索树节点。

    `value_sum` / `visits` 得到的 Q 始终是**该节点走子方**的视角，与
    `encode_state` 的己方/对方平面一致；父节点看子节点要取负。
    """

    __slots__ = ("prior", "visits", "value_sum", "children")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[int, _Node] = {}

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def _puct_score(parent_visits: int, child: _Node, c_puct: float) -> float:
    # 子节点的 Q 是它自己走子方的视角，父节点选点时要取负。
    exploit = -child.value if child.visits else 0.0
    explore = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
    return exploit + explore


def _terminal_value(state: GameState) -> float:
    """终局叶子的价值，取「本应轮到走子的一方」的视角。

    `GameState.apply_move` 在终局时不翻转 current_player（胜者仍是 current_player），
    所以这里不能直接用它判断视角。约定：有人获胜时，轮到走子的一方必然是输家，
    故为 -1；和棋为 0。这样回传时逐层取负即可自洽。
    """
    return 0.0 if state.winner == 0 else -1.0


class MCTSEngine:
    """用策略/价值网络引导的 PUCT 搜索引擎。"""

    def __init__(self, model, device: str = "cpu", config: MCTSConfig | None = None) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.config = config or MCTSConfig()

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

    # ------------------------------------------------------------------ 网络
    @torch.no_grad()
    def _evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        """一次前向求出整批局面的先验概率与价值（均为走子方视角）。"""
        encoded = torch.stack([encode_state(state) for state in states]).to(self.device)
        masks = torch.from_numpy(np.stack([state.legal_mask() for state in states])).to(self.device)
        logits, values = self.model(encoded)
        priors = torch.softmax(logits.masked_fill(masks == 0, -1e9), dim=-1)
        return priors.cpu().numpy(), values.flatten().cpu().numpy()

    # ------------------------------------------------------------------ 搜索
    def _expand(self, node: _Node, state: GameState, priors: np.ndarray) -> None:
        size = state.board.size
        for move in state.legal_moves():
            index = move.to_index(size)
            node.children[index] = _Node(float(priors[index]))

    def _add_root_noise(self, root: _Node, rng: np.random.Generator) -> None:
        weight = self.config.dirichlet_weight
        if weight <= 0.0 or not root.children:
            return
        actions = list(root.children)
        noise = rng.dirichlet([self.config.dirichlet_alpha] * len(actions))
        for action, sample in zip(actions, noise):
            child = root.children[action]
            child.prior = (1 - weight) * child.prior + weight * float(sample)

    def _run_search(self, states: list[GameState], rng: np.random.Generator) -> list[_Node]:
        """对一批局面同步跑完整轮搜索，返回各自的根节点。"""
        roots = [_Node(prior=1.0) for _ in states]

        # 根节点先统一展开一次，之后每轮模拟只需评估新到达的叶子。
        priors, values = self._evaluate(states)
        for root, state, prior_row, value in zip(roots, states, priors, values):
            self._expand(root, state, prior_row)
            root.visits = 1
            root.value_sum = float(value)
            self._add_root_noise(root, rng)

        for _ in range(self.config.simulations):
            leaf_paths: list[list[_Node]] = []
            leaf_states: list[GameState] = []
            pending: list[int] = []

            for tree_index, (root, state) in enumerate(zip(roots, states)):
                if state.is_terminal:
                    continue
                node = root
                path = [node]
                sim_state = state.copy()
                while node.expanded and not sim_state.is_terminal:
                    action = max(
                        node.children,
                        key=lambda a, n=node: _puct_score(n.visits, n.children[a], self.config.c_puct),
                    )
                    node = node.children[action]
                    sim_state.apply_move(Move.from_index(action, sim_state.board.size))
                    path.append(node)

                if sim_state.is_terminal:
                    self._backup(path, _terminal_value(sim_state))
                    continue
                leaf_paths.append(path)
                leaf_states.append(sim_state)
                pending.append(tree_index)

            if not leaf_states:
                continue
            # 所有树的叶子凑成一批：这是 MCTS 在 GPU 上可行的关键。
            batch_priors, batch_values = self._evaluate(leaf_states)
            for path, leaf_state, prior_row, value in zip(
                leaf_paths, leaf_states, batch_priors, batch_values
            ):
                self._expand(path[-1], leaf_state, prior_row)
                self._backup(path, float(value))

        return roots

    @staticmethod
    def _backup(path: list[_Node], value: float) -> None:
        """自叶子向根回传，逐层取负——相邻层是对立的走子方。"""
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    # ------------------------------------------------------------------ 接口
    def _result_from_root(
        self, root: _Node, state: GameState, temperature: float, rng: np.random.Generator
    ) -> AnalysisResult:
        actions = list(root.children)
        visits = np.array([root.children[a].visits for a in actions], dtype=np.float64)
        if visits.sum() <= 0:  # 极端情况：无可搜索的走子，退回先验
            visits = np.array([root.children[a].prior for a in actions], dtype=np.float64)
        # 采样用的排序分：访问数 + 先验。先验落在 [0,1) 而访问数是整数，所以它只在
        # 访问数持平时起作用。这在开局尤其要紧——分支约 80 个，64 次模拟下前 5 名的
        # 访问数是 [2,2,1,1,1]（仅占 11%），纯按访问排序等于按噪声排序，会把角和边
        # 排进前列；此时网络先验才是唯一有信息的排序依据。
        priors = np.array([root.children[a].prior for a in actions], dtype=np.float64)
        ranking = visits + priors

        if temperature <= 1e-6:
            choice = int(np.argmax(visits))
        else:
            pool = np.arange(len(actions))
            top_k = self.config.sample_top_k
            if top_k > 0 and top_k < len(actions):
                pool = np.argsort(-ranking)[:top_k]
            weights = ranking[pool] ** (1.0 / temperature)
            total = weights.sum()
            weights = weights / total if total > 0 else np.full(len(pool), 1.0 / len(pool))
            choice = int(rng.choice(pool, p=weights))

        probabilities = visits / visits.sum()
        size = state.board.size
        order = np.argsort(-visits)[:5]
        candidates = [
            CandidateMove(
                move=Move.from_index(actions[i], size),
                score=float(probabilities[i]),
                visits=int(root.children[actions[i]].visits),
                value=float(-root.children[actions[i]].value),
            )
            for i in order
        ]
        return AnalysisResult(
            action=Move.from_index(actions[choice], size),
            action_probability=float(probabilities[choice]),
            value_estimate=float(root.value),
            candidates=candidates,
        )

    def select_moves(self, states: list[GameState], temperature: float = 0.0) -> list[AnalysisResult]:
        if not states:
            return []
        rng = np.random.default_rng()
        roots = self._run_search(states, rng)
        return [
            self._result_from_root(root, state, temperature, rng)
            for root, state in zip(roots, states)
        ]

    def select_move(self, state: GameState, temperature: float = 0.0) -> AnalysisResult:
        return self.select_moves([state], temperature=temperature)[0]

    def analyze(self, state: GameState, top_k: int = 5) -> list[CandidateMove]:
        return self.select_move(state, temperature=0.0).candidates[:top_k]
