from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

ACTION_NAMES = [
    "guard",
    "step_forward",
    "step_back",
    "sidestep_left",
    "sidestep_right",
    "circle_left",
    "circle_right",
    "jab",
    "cross",
    "hook",
    "low_kick",
    "push",
    "recover",
]

ACTION_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(ACTION_NAMES)}

ATTACK_ACTIONS = {
    ACTION_TO_ID["jab"],
    ACTION_TO_ID["cross"],
    ACTION_TO_ID["hook"],
    ACTION_TO_ID["low_kick"],
    ACTION_TO_ID["push"],
}

DEFENSIVE_ACTIONS = {
    ACTION_TO_ID["guard"],
    ACTION_TO_ID["step_back"],
    ACTION_TO_ID["recover"],
    ACTION_TO_ID["sidestep_left"],
    ACTION_TO_ID["sidestep_right"],
}

MOVEMENT_ACTIONS = {
    ACTION_TO_ID["step_forward"],
    ACTION_TO_ID["step_back"],
    ACTION_TO_ID["sidestep_left"],
    ACTION_TO_ID["sidestep_right"],
    ACTION_TO_ID["circle_left"],
    ACTION_TO_ID["circle_right"],
}

STYLE_NAMES = ["pressure", "counter", "evasive", "bully"]
STYLE_TO_ID = {name: i for i, name in enumerate(STYLE_NAMES)}

# range, damage, stamina_cost, cooldown_steps, balance_self_cost, balance_impact,
# guard_break, cone_cos, forward_lunge
ATTACK_TABLE: Dict[int, Tuple[float, float, float, int, float, float, float, float, float]] = {
    ACTION_TO_ID["jab"]: (1.18, 4.2, 0.070, 3, 0.020, 0.075, 0.10, 0.47, 0.10),
    ACTION_TO_ID["cross"]: (1.08, 7.0, 0.115, 5, 0.055, 0.140, 0.20, 0.62, 0.17),
    ACTION_TO_ID["hook"]: (0.86, 8.5, 0.135, 6, 0.080, 0.195, 0.31, 0.28, 0.06),
    ACTION_TO_ID["low_kick"]: (1.00, 5.4, 0.125, 6, 0.105, 0.170, 0.05, 0.54, 0.05),
    ACTION_TO_ID["push"]: (0.80, 2.6, 0.095, 5, 0.040, 0.285, 0.42, 0.45, 0.14),
}

@dataclass(frozen=True)
class SimConfig:
    """Parameters for the self-contained combat simulator.

    The simulator intentionally uses high-level humanoid skill tokens rather than raw
    motor torques. This keeps the project runnable on commodity hardware while still
    exercising the core data, policy, evaluation, and safety architecture.
    """

    arena_radius: float = 5.0
    dt: float = 0.10
    max_steps: int = 180
    max_speed: float = 1.65
    turn_rate: float = 0.34
    velocity_decay: float = 0.78
    stamina_recovery: float = 0.020
    guard_decay: float = 0.84
    min_balance: float = -0.35
    fall_balance_threshold: float = 0.03
    recovery_steps: int = 9
    boundary_penalty: float = 0.17
    boundary_score: float = 1.5
    fall_score: float = 11.0
    knockdown_damage: float = 4.0
    ko_health: float = 0.0
    seed: int = 7

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    batch_size: int = 512
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-5
    val_split: float = 0.15
    seed: int = 13

OBS_VERSION = 3
