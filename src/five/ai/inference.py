from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from five.ai.encoder import encode_state
from five.ai.interfaces import AIEngine, AnalysisResult, CandidateMove
from five.ai.model import PolicyValueNet
from five.core.move import Move
from five.core.state import GameState


class ModelAIEngine(AIEngine):
    def __init__(self, model: PolicyValueNet, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        state_dict = payload["model_state"] if "model_state" in payload else payload
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def select_move(self, state: GameState, temperature: float = 0.0) -> AnalysisResult:
        encoded = encode_state(state).unsqueeze(0).to(self.device)
        legal_mask = torch.from_numpy(state.legal_mask()).to(self.device)
        logits, value = self.model(encoded)
        masked_logits = logits.squeeze(0).masked_fill(legal_mask == 0, -1e9)
        probabilities = torch.softmax(masked_logits / max(temperature, 1e-3), dim=-1)
        if temperature <= 1e-6:
            action_index = int(torch.argmax(masked_logits).item())
        else:
            action_index = int(torch.multinomial(probabilities, num_samples=1).item())
        move = Move.from_index(action_index, state.board.size)
        candidates = self._top_candidates(probabilities, state, top_k=5)
        return AnalysisResult(
            action=move,
            action_probability=float(probabilities[action_index].item()),
            value_estimate=float(value.item()),
            candidates=candidates,
        )

    @torch.no_grad()
    def select_moves(
        self,
        states: list[GameState],
        temperature: float = 0.0,
    ) -> list[AnalysisResult]:
        """一次前向处理整批局面；语义与逐个调用 select_move 完全一致。

        自博弈的瓶颈是 batch=1 前向：同一时刻有几百局在等同一个网络，凑成一批
        可以把每局面成本降低一到两个数量级。
        """
        if not states:
            return []

        encoded = torch.stack([encode_state(state) for state in states]).to(self.device)
        masks = torch.from_numpy(np.stack([state.legal_mask() for state in states])).to(self.device)
        logits, values = self.model(encoded)
        masked_logits = logits.masked_fill(masks == 0, -1e9)
        probabilities = torch.softmax(masked_logits / max(temperature, 1e-3), dim=-1)
        if temperature <= 1e-6:
            action_indices = torch.argmax(masked_logits, dim=-1)
        else:
            action_indices = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

        chosen = probabilities.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        action_list = action_indices.tolist()
        probability_list = chosen.tolist()
        value_list = values.flatten().tolist()

        results: list[AnalysisResult] = []
        for index, state in enumerate(states):
            results.append(
                AnalysisResult(
                    action=Move.from_index(int(action_list[index]), state.board.size),
                    action_probability=float(probability_list[index]),
                    value_estimate=float(value_list[index]),
                    candidates=self._top_candidates(probabilities[index], state, top_k=5),
                )
            )
        return results

    @torch.no_grad()
    def analyze(self, state: GameState, top_k: int = 5) -> list[CandidateMove]:
        encoded = encode_state(state).unsqueeze(0).to(self.device)
        legal_mask = torch.from_numpy(state.legal_mask()).to(self.device)
        logits, _ = self.model(encoded)
        masked_logits = logits.squeeze(0).masked_fill(legal_mask == 0, -1e9)
        probabilities = torch.softmax(masked_logits, dim=-1)
        return self._top_candidates(probabilities, state, top_k=top_k)

    def _top_candidates(
        self,
        probabilities: torch.Tensor,
        state: GameState,
        top_k: int,
    ) -> list[CandidateMove]:
        limit = min(top_k, int(state.legal_mask().sum()))
        values, indices = torch.topk(probabilities, k=limit)
        candidates: list[CandidateMove] = []
        for probability, index in zip(values.tolist(), indices.tolist()):
            candidates.append(
                CandidateMove(
                    move=Move.from_index(int(index), state.board.size),
                    score=float(probability),
                )
            )
        return candidates
