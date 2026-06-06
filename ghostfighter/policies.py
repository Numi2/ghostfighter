from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import ACTION_NAMES, ACTION_TO_ID, ATTACK_ACTIONS, STYLE_NAMES
from .env import FightEnv, FighterState, rotate_to_local
from .models import PolicyNet


class Policy(Protocol):
    name: str

    def select_action(self, obs: np.ndarray, env: FightEnv, fighter_idx: int) -> int:
        ...


@dataclass
class ScriptedPilot:
    """Expert-ish scripted policy used to create pilot traces.

    Scripts are intentionally imperfect. They produce diverse human-like styles that
    the neural ghost can clone, and the safety layer can improve under stress.
    """

    style: str
    seed: int = 0

    def __post_init__(self) -> None:
        if self.style not in STYLE_NAMES:
            raise ValueError(f"Unknown style {self.style}. Valid styles: {STYLE_NAMES}")
        self.name = f"scripted_{self.style}"
        self.rng = np.random.default_rng(self.seed)

    def select_action(self, obs: np.ndarray, env: FightEnv, fighter_idx: int) -> int:
        own, opp = env._by_idx(fighter_idx)
        if own.fallen:
            return ACTION_TO_ID["recover"]
        if own.health < 18 and own.balance < 0.42:
            return self._choice(["recover", "guard", "step_back"], [0.55, 0.25, 0.20])
        if own.balance < 0.22:
            return self._choice(["recover", "guard", "step_back"], [0.70, 0.18, 0.12])
        if own.stamina < 0.16:
            return self._choice(["guard", "recover", "step_back"], [0.46, 0.34, 0.20])

        rel = opp.pos() - own.pos()
        rel_local = rotate_to_local(rel, own.theta)
        dist = float(np.linalg.norm(rel))
        near_boundary = np.linalg.norm(own.pos()) > env.config.arena_radius * 0.82
        opp_attacking = opp.last_action in ATTACK_ACTIONS and opp.cooldown > 0
        opp_vulnerable = opp.whiffed > 0.2 or opp.balance < 0.38 or opp.cooldown > 2

        if near_boundary and rel_local[0] > -0.1:
            if self.rng.random() < 0.55:
                return self._choice(["sidestep_left", "sidestep_right", "step_forward", "guard"], [0.27, 0.27, 0.30, 0.16])

        if self.style == "pressure":
            return self._pressure(own, opp, dist, opp_attacking, opp_vulnerable)
        if self.style == "counter":
            return self._counter(own, opp, dist, opp_attacking, opp_vulnerable)
        if self.style == "evasive":
            return self._evasive(own, opp, dist, opp_attacking, opp_vulnerable)
        if self.style == "bully":
            return self._bully(own, opp, dist, opp_attacking, opp_vulnerable)
        return ACTION_TO_ID["guard"]

    def _pressure(self, own: FighterState, opp: FighterState, dist: float, opp_attacking: bool, opp_vulnerable: bool) -> int:
        if own.cooldown > 0:
            return self._choice(["guard", "circle_left", "circle_right", "step_forward"], [0.35, 0.22, 0.22, 0.21])
        if dist > 1.35:
            return self._choice(["step_forward", "circle_left", "circle_right", "guard"], [0.68, 0.12, 0.12, 0.08])
        if opp_attacking and own.guard < 0.34:
            return self._choice(["guard", "sidestep_left", "sidestep_right", "jab"], [0.42, 0.20, 0.20, 0.18])
        if dist < 0.66:
            return self._choice(["push", "hook", "step_back", "guard"], [0.36, 0.33, 0.17, 0.14])
        return self._choice(["jab", "cross", "hook", "step_forward", "low_kick"], [0.35, 0.28, 0.16, 0.12, 0.09])

    def _counter(self, own: FighterState, opp: FighterState, dist: float, opp_attacking: bool, opp_vulnerable: bool) -> int:
        if own.cooldown > 0:
            return self._choice(["guard", "step_back", "circle_left", "circle_right"], [0.52, 0.20, 0.14, 0.14])
        if opp_attacking and dist < 1.3:
            return self._choice(["guard", "sidestep_left", "sidestep_right", "cross"], [0.48, 0.18, 0.18, 0.16])
        if opp_vulnerable and dist < 1.15:
            return self._choice(["cross", "hook", "jab", "push"], [0.42, 0.24, 0.22, 0.12])
        if dist < 0.78:
            return self._choice(["step_back", "guard", "push", "sidestep_left"], [0.35, 0.30, 0.20, 0.15])
        if dist > 1.55:
            return self._choice(["guard", "step_forward", "circle_left", "circle_right"], [0.36, 0.30, 0.17, 0.17])
        return self._choice(["guard", "jab", "circle_left", "circle_right", "step_back"], [0.36, 0.20, 0.18, 0.18, 0.08])

    def _evasive(self, own: FighterState, opp: FighterState, dist: float, opp_attacking: bool, opp_vulnerable: bool) -> int:
        if own.cooldown > 0:
            return self._choice(["sidestep_left", "sidestep_right", "circle_left", "circle_right", "guard"], [0.20, 0.20, 0.22, 0.22, 0.16])
        if opp_attacking and dist < 1.4:
            return self._choice(["sidestep_left", "sidestep_right", "step_back", "guard"], [0.31, 0.31, 0.20, 0.18])
        if dist < 0.8:
            return self._choice(["step_back", "sidestep_left", "sidestep_right", "push"], [0.36, 0.24, 0.24, 0.16])
        if opp_vulnerable and 0.85 <= dist <= 1.2:
            return self._choice(["low_kick", "jab", "cross", "circle_left"], [0.34, 0.28, 0.18, 0.20])
        if dist > 1.75:
            return self._choice(["circle_left", "circle_right", "step_forward", "guard"], [0.30, 0.30, 0.26, 0.14])
        return self._choice(["circle_left", "circle_right", "jab", "low_kick", "guard"], [0.26, 0.26, 0.20, 0.16, 0.12])

    def _bully(self, own: FighterState, opp: FighterState, dist: float, opp_attacking: bool, opp_vulnerable: bool) -> int:
        if own.cooldown > 0:
            return self._choice(["guard", "step_forward", "recover", "circle_left"], [0.30, 0.33, 0.20, 0.17])
        if dist > 0.95:
            return self._choice(["step_forward", "step_forward", "jab", "low_kick"], [0.49, 0.21, 0.18, 0.12])
        if own.balance < 0.36:
            return self._choice(["push", "recover", "guard"], [0.34, 0.38, 0.28])
        return self._choice(["push", "hook", "cross", "low_kick", "guard"], [0.33, 0.27, 0.20, 0.12, 0.08])

    def _choice(self, names: list[str], probs: list[float]) -> int:
        probs_np = np.asarray(probs, dtype=np.float64)
        probs_np = probs_np / probs_np.sum()
        name = names[int(self.rng.choice(len(names), p=probs_np))]
        return ACTION_TO_ID[name]


class EnsembleOpponent:
    """Switches among styles between matches for robust evaluation."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.name = "scripted_ensemble"
        self.current = ScriptedPilot("counter", seed=seed)

    def reset(self, episode: int = 0) -> None:
        style = STYLE_NAMES[int(self.rng.integers(0, len(STYLE_NAMES)))]
        self.current = ScriptedPilot(style, seed=int(self.rng.integers(1, 10_000_000)) + episode)

    def select_action(self, obs: np.ndarray, env: FightEnv, fighter_idx: int) -> int:
        return self.current.select_action(obs, env, fighter_idx)


class NeuralGhostPolicy:
    def __init__(
        self,
        model: PolicyNet,
        style_id: int = 0,
        deterministic: bool = True,
        temperature: float = 0.85,
        name: str | None = None,
    ):
        self.model = model
        self.style_id = int(style_id)
        self.deterministic = deterministic
        self.temperature = float(temperature)
        self.name = name or f"ghost_{STYLE_NAMES[self.style_id]}"

    def select_action(self, obs: np.ndarray, env: FightEnv, fighter_idx: int) -> int:
        return self.model.act(obs, self.style_id, deterministic=self.deterministic, temperature=self.temperature)


def action_name(action: int) -> str:
    return ACTION_NAMES[int(action)]
