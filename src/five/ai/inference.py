from __future__ import annotations

from pathlib import Path

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
