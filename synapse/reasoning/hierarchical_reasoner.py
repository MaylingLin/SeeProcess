"""Hierarchical reasoning, planning, and execution module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionVocabulary:
    """Dynamic vocabulary that assigns ids to action strings."""

    def __init__(self, base_tokens: Optional[Sequence[str]] = None):
        tokens = list(base_tokens) if base_tokens else ["noop"]
        if "noop" not in tokens:
            tokens.insert(0, "noop")
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        for tok in tokens:
            self.add_token(tok)

    def add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.id_to_token)
        self.token_to_id[token] = idx
        self.id_to_token.append(token)
        return idx

    def extend(self, tokens: Sequence[str]) -> None:
        for token in tokens:
            self.add_token(token)

    def encode(self, token: Optional[str]) -> int:
        token = (token or "noop").strip().lower()
        if not token:
            token = "noop"
        return self.token_to_id.get(token, self.token_to_id["noop"])

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.id_to_token)


class DomainAdapter(nn.Module):
    """Maps modality embeddings into a domain-specific action space."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class LowLevelActionPredictor(nn.Module):
    """Predicts concrete actions conditioned on plan embeddings."""

    def __init__(self, state_dim: int, plan_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.plan_proj = nn.Linear(plan_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.guidance = nn.Linear(hidden_dim, state_dim)

    def forward(
        self, state: torch.Tensor, plan_vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.state_proj(state) + self.plan_proj(plan_vector)
        hidden = self.fusion(fused)
        logits = self.policy_head(hidden)
        guided_state = state + self.guidance(hidden)
        return logits, guided_state


class HighLevelPlanner(nn.Module):
    """Produces hierarchical plans from abstract memory + instruction."""

    def __init__(self, context_dim: int, text_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True, dim_feedforward=hidden_dim * 4
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.plan_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self, high_context: torch.Tensor, text_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        context_token = self.context_proj(high_context).unsqueeze(1)
        text_token = self.text_proj(text_features).unsqueeze(1)
        tokens = torch.cat([context_token, text_token], dim=1)
        encoded = self.encoder(tokens)
        plan_vector = self.plan_head(encoded[:, 0])
        return {"plan_vector": plan_vector, "encoded_tokens": encoded}


@dataclass
class ReasonerOutput:
    action_logits: torch.Tensor
    plan_vector: torch.Tensor
    guided_state: torch.Tensor
    value: torch.Tensor
    losses: Dict[str, torch.Tensor]
    total_loss: torch.Tensor


class HierarchicalReasoningExecutor(nn.Module):
    """Combines planning, RL, and behaviour cloning for execution."""

    def __init__(
        self,
        low_dim: int,
        high_dim: int,
        text_dim: int,
        hidden_dim: int = 512,
        action_tokens: Optional[Sequence[str]] = None,
        rl_weight: float = 1.0,
        bc_weight: float = 1.0,
        value_weight: float = 0.5,
    ):
        super().__init__()
        self.low_dim = low_dim
        self.high_dim = high_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.rl_weight = rl_weight
        self.bc_weight = bc_weight
        self.value_weight = value_weight
        self.action_vocab = ActionVocabulary(action_tokens)
        self.planner = HighLevelPlanner(high_dim, text_dim, hidden_dim)
        self.policy = LowLevelActionPredictor(low_dim, hidden_dim, hidden_dim, len(self.action_vocab))
        self.value_head = nn.Linear(hidden_dim, 1)
        self.domain_adapters = nn.ModuleDict()
        self.default_domain = "gui"
        self.register_cross_domain_adapter("gui")
        self.register_cross_domain_adapter("robotics")
        self.register_cross_domain_adapter("hci")

    def register_cross_domain_adapter(
        self, domain: str, hidden_dim: Optional[int] = None
    ) -> None:
        if hidden_dim is None:
            hidden_dim = self.hidden_dim
        adapter = DomainAdapter(self.low_dim, hidden_dim)
        self.domain_adapters[domain] = adapter

    def encode_actions(
        self, ops_batch: Sequence[Sequence[Dict]], domains: Optional[Sequence[str]], device: torch.device
    ) -> torch.Tensor:
        labels: List[int] = []
        for idx, ops in enumerate(ops_batch):
            action_token = "noop"
            if ops:
                for op in ops:
                    if isinstance(op, dict):
                        action_token = str(op.get("type") or op.get("op") or "noop")
                        if action_token:
                            break
            labels.append(self.action_vocab.encode(action_token))
        return torch.tensor(labels, dtype=torch.long, device=device)

    def estimate_advantages(
        self, traj_mask: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_counts = traj_mask.sum(dim=(1, 2))
        norm = float(max(traj_mask.size(1) * traj_mask.size(2), 1))
        returns = step_counts / norm
        advantages = returns - returns.mean()
        return advantages.to(device), returns.to(device)

    def forward(
        self,
        low_level_state: torch.Tensor,
        high_level_context: torch.Tensor,
        text_features: torch.Tensor,
        action_labels: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        domains: Optional[Sequence[str]] = None,
    ) -> ReasonerOutput:
        if domains is None:
            domains = [self.default_domain] * low_level_state.size(0)

        planner_out = self.planner(high_level_context, text_features)
        plan_vec = planner_out["plan_vector"]
        adapted_state = self._apply_domain_adapters(low_level_state, domains)
        logits, guided_state = self.policy(adapted_state, plan_vec)
        value = self.value_head(plan_vec).squeeze(-1)

        total_loss = torch.zeros(1, device=logits.device)
        losses: Dict[str, torch.Tensor] = {}

        if action_labels is not None:
            bc_loss = F.cross_entropy(logits, action_labels)
            losses["bc_loss"] = bc_loss
            total_loss = total_loss + self.bc_weight * bc_loss

        if advantages is not None and action_labels is not None:
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs.gather(1, action_labels.view(-1, 1)).squeeze(1)
            rl_loss = -(advantages.detach() * selected).mean()
            losses["rl_loss"] = rl_loss
            total_loss = total_loss + self.rl_weight * rl_loss

        if returns is not None:
            value_loss = F.mse_loss(value, returns)
            losses["value_loss"] = value_loss
            total_loss = total_loss + self.value_weight * value_loss

        return ReasonerOutput(
            action_logits=logits,
            plan_vector=plan_vec,
            guided_state=guided_state,
            value=value,
            losses=losses,
            total_loss=total_loss.squeeze(0),
        )

    def _apply_domain_adapters(
        self, state: torch.Tensor, domains: Sequence[str]
    ) -> torch.Tensor:
        outputs = torch.zeros_like(state)
        groups: Dict[str, List[int]] = {}
        for idx, domain in enumerate(domains):
            groups.setdefault(domain, []).append(idx)
        for domain, indices in groups.items():
            adapter = self.domain_adapters[domain] if domain in self.domain_adapters else self.domain_adapters[self.default_domain]
            subset = state[indices]
            outputs[indices] = adapter(subset)
        return outputs
