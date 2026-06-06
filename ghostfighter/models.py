from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
from torch import nn

from .config import ACTION_NAMES, STYLE_NAMES


class PolicyNet(nn.Module):
    """Conditional behavior-cloning policy.

    The style id is embedded and concatenated to the simulator observation. The same
    network can act as different ghost fighters: pressure, counter, evasive, or bully.
    """

    def __init__(self, obs_dim: int, num_actions: int = len(ACTION_NAMES), num_styles: int = len(STYLE_NAMES), hidden: int = 192):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.num_styles = int(num_styles)
        self.style_emb = nn.Embedding(num_styles, 16)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 16, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, num_actions),
        )

    def forward(self, obs: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if style.ndim == 0:
            style = style.unsqueeze(0)
        emb = self.style_emb(style.long())
        x = torch.cat([obs.float(), emb], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def act(self, obs, style: int, deterministic: bool = True, temperature: float = 1.0) -> int:
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        style_t = torch.as_tensor([int(style)], dtype=torch.long, device=device)
        logits = self.forward(obs_t, style_t)[0]
        if deterministic:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
        return int(torch.multinomial(probs, 1).item())


@dataclass
class CheckpointMeta:
    obs_dim: int
    action_names: list[str]
    style_names: list[str]
    metrics: Dict[str, Any]


def load_policy_checkpoint(path: str, map_location: str = "cpu") -> tuple[PolicyNet, Dict[str, Any]]:
    torch.set_num_threads(1)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = PolicyNet(
        obs_dim=int(ckpt["obs_dim"]),
        num_actions=len(ckpt.get("action_names", ACTION_NAMES)),
        num_styles=len(ckpt.get("style_names", STYLE_NAMES)),
        hidden=int(ckpt.get("hidden", 192)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
