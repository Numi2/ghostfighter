from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from .config import ACTION_NAMES, ACTION_TO_ID, ATTACK_ACTIONS, ATTACK_TABLE, MOVEMENT_ACTIONS, STYLE_NAMES, STYLE_TO_ID
from .env import FightEnv, rotate_to_local


ATTRIBUTE_NAMES = [
    "engagement_drive",
    "guard_discipline",
    "counter_timing",
    "lateral_mobility",
    "stamina_discipline",
    "boundary_awareness",
    "damage_targeting",
    "risk_tolerance",
    "close_range_pressure",
]


DEFAULT_POLICY_SPEC: Dict[str, Dict[str, list[float]]] = {
    "pressure": {
        "engagement_drive": [0.72, 1.00],
        "guard_discipline": [0.25, 0.55],
        "counter_timing": [0.22, 0.55],
        "lateral_mobility": [0.25, 0.58],
        "stamina_discipline": [0.25, 0.62],
        "boundary_awareness": [0.38, 0.75],
        "damage_targeting": [0.45, 0.82],
        "risk_tolerance": [0.55, 0.90],
        "close_range_pressure": [0.52, 0.88],
    },
    "counter": {
        "engagement_drive": [0.35, 0.72],
        "guard_discipline": [0.62, 0.96],
        "counter_timing": [0.68, 1.00],
        "lateral_mobility": [0.30, 0.70],
        "stamina_discipline": [0.55, 0.92],
        "boundary_awareness": [0.52, 0.90],
        "damage_targeting": [0.42, 0.80],
        "risk_tolerance": [0.20, 0.56],
        "close_range_pressure": [0.25, 0.62],
    },
    "evasive": {
        "engagement_drive": [0.20, 0.55],
        "guard_discipline": [0.45, 0.82],
        "counter_timing": [0.40, 0.82],
        "lateral_mobility": [0.68, 1.00],
        "stamina_discipline": [0.62, 1.00],
        "boundary_awareness": [0.70, 1.00],
        "damage_targeting": [0.25, 0.62],
        "risk_tolerance": [0.12, 0.45],
        "close_range_pressure": [0.08, 0.42],
    },
    "bully": {
        "engagement_drive": [0.62, 0.95],
        "guard_discipline": [0.18, 0.52],
        "counter_timing": [0.18, 0.50],
        "lateral_mobility": [0.15, 0.52],
        "stamina_discipline": [0.18, 0.55],
        "boundary_awareness": [0.32, 0.72],
        "damage_targeting": [0.52, 0.92],
        "risk_tolerance": [0.58, 0.96],
        "close_range_pressure": [0.72, 1.00],
    },
}


@dataclass(frozen=True)
class PolicyAttributes:
    archetype: str
    policy_id: str
    seed: int
    engagement_drive: float
    guard_discipline: float
    counter_timing: float
    lateral_mobility: float
    stamina_discipline: float
    boundary_awareness: float
    damage_targeting: float
    risk_tolerance: float
    close_range_pressure: float

    def vector(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in ATTRIBUTE_NAMES], dtype=np.float32)


class AttributePolicy:
    """Policy generated from user-specified behavior attributes.

    The policy scores high-level actions with interpretable robotics terms rather
    than fixed if/else scripts. Fighting labels are metadata for behavior priors;
    the actual controller is the sampled attribute vector.
    """

    def __init__(self, attributes: PolicyAttributes, lookahead: bool = True):
        self.attributes = attributes
        self.name = f"attribute_{attributes.policy_id}"
        self.lookahead = bool(lookahead)
        self.rng = np.random.default_rng(attributes.seed)

    def select_action(self, obs: np.ndarray, env: FightEnv, fighter_idx: int) -> int:
        own, _ = env._by_idx(fighter_idx)
        if own.fallen:
            return ACTION_TO_ID["recover"]
        actions = list(range(len(ACTION_NAMES)))
        if self.lookahead:
            scored = [(self._lookahead_score(env, fighter_idx, action), action) for action in actions]
        else:
            scored = [(self._score_action(env, fighter_idx, action), action) for action in actions]
        scored.sort(reverse=True, key=lambda item: item[0])
        best_score = scored[0][0]
        near_best = [action for score, action in scored if score >= best_score - 0.025]
        return int(self.rng.choice(near_best))

    def _lookahead_score(self, env: FightEnv, fighter_idx: int, action: int) -> float:
        own, opp = env._by_idx(fighter_idx)
        base = self._score_action(env, fighter_idx, action)
        if own.cooldown > 0 and action in ATTACK_ACTIONS:
            return base - 0.6
        cloned = env.clone()
        opponent_action = _reactive_opponent_action(cloned, 1 - fighter_idx)
        if fighter_idx == 0:
            _, _, reward, _, _, _ = cloned.step(action, opponent_action)
            post = cloned.red
        else:
            _, _, _, reward, _, _ = cloned.step(opponent_action, action)
            post = cloned.blue
        fall_penalty = 1.4 if post.falls > own.falls else 0.0
        balance_delta = post.balance - own.balance
        health_delta = post.health - own.health
        return base + 0.18 * reward + 0.35 * balance_delta + 0.015 * health_delta - fall_penalty

    def _score_action(self, env: FightEnv, fighter_idx: int, action: int) -> float:
        a = self.attributes
        own, opp = env._by_idx(fighter_idx)
        rel = opp.pos() - own.pos()
        rel_local = rotate_to_local(rel, own.theta)
        dist = float(np.linalg.norm(rel))
        boundary_clearance = max(0.0, env.config.arena_radius - float(np.linalg.norm(own.pos()))) / env.config.arena_radius
        opp_attacking = opp.last_action in ATTACK_ACTIONS and opp.cooldown > 0 and dist < 1.35
        opp_vulnerable = opp.whiffed > 0.2 or opp.balance < 0.38 or opp.cooldown > 2
        leg_damage = 0.5 * (own.left_leg_damage + own.right_leg_damage)
        arm_damage = 0.5 * (own.left_arm_damage + own.right_arm_damage)
        score = 0.0

        if action == ACTION_TO_ID["recover"]:
            score += 1.20 * max(0.0, 0.42 - own.balance)
            score += 0.75 * max(0.0, 0.30 - own.stamina) * a.stamina_discipline
            score += 0.35 * leg_damage
        if action == ACTION_TO_ID["guard"]:
            score += 0.50 * a.guard_discipline
            score += 0.90 * a.guard_discipline if opp_attacking else 0.0
            score += 0.35 * max(0.0, 0.42 - own.balance)
        if action == ACTION_TO_ID["step_forward"]:
            score += 0.80 * a.engagement_drive * _distance_need(dist, target=1.05)
            score += 0.35 * a.close_range_pressure if dist > 0.75 else -0.10
            score -= 0.80 * a.boundary_awareness if boundary_clearance < 0.14 and rel_local[0] < 0 else 0.0
        if action == ACTION_TO_ID["step_back"]:
            score += 0.75 * a.guard_discipline if dist < 0.78 else 0.0
            score += 0.55 * a.stamina_discipline if own.stamina < 0.24 else 0.0
            score -= 0.55 * a.boundary_awareness if boundary_clearance < 0.18 else 0.0
        if action in {ACTION_TO_ID["sidestep_left"], ACTION_TO_ID["sidestep_right"], ACTION_TO_ID["circle_left"], ACTION_TO_ID["circle_right"]}:
            score += 0.70 * a.lateral_mobility
            score += 0.45 * a.boundary_awareness if boundary_clearance < 0.24 else 0.0
            score += 0.35 * a.counter_timing if opp_attacking else 0.0
            score -= 0.55 * leg_damage

        if action in ATTACK_ACTIONS:
            params = ATTACK_TABLE[action]
            range_, damage, stamina_cost, cooldown, self_cost, balance_impact, guard_break, cone_cos, _ = params
            in_range = dist <= range_ * 1.12
            aim_quality = rel_local[0] / max(dist, 1e-6)
            score += 0.55 * a.engagement_drive
            score += 0.45 * a.counter_timing if opp_vulnerable else 0.0
            score += 0.08 * damage * a.damage_targeting
            score += 0.22 * guard_break * a.close_range_pressure
            score += 0.55 if in_range and aim_quality > cone_cos - 0.12 else -0.45
            score -= 0.85 * max(0.0, stamina_cost - own.stamina) * a.stamina_discipline
            score -= 1.10 * max(0.0, self_cost + 0.08 * leg_damage - own.balance) * (1.0 - a.risk_tolerance)
            score -= 0.40 * arm_damage if action in {ACTION_TO_ID["cross"], ACTION_TO_ID["hook"], ACTION_TO_ID["push"]} else 0.0
            if action == ACTION_TO_ID["push"]:
                score += 0.75 * a.close_range_pressure if dist < 0.85 else -0.20
            if action == ACTION_TO_ID["low_kick"]:
                score += 0.30 * a.damage_targeting if opp.balance < 0.62 else 0.0

        score -= 0.80 * max(0.0, 0.18 - own.balance) * (1.0 - a.risk_tolerance)
        score -= 0.25 * own.cooldown if action in ATTACK_ACTIONS else 0.0
        score += float(self.rng.normal(0.0, 0.015))
        return float(score)


def load_policy_spec(path: str | Path | None = None) -> Dict[str, Dict[str, list[float]]]:
    if path is None:
        spec = DEFAULT_POLICY_SPEC
    else:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_policy_spec(spec)
    return spec


def sample_attribute_policies(
    spec: Dict[str, Dict[str, list[float]]],
    variants_per_archetype: int = 8,
    seed: int = 101,
) -> list[PolicyAttributes]:
    rng = np.random.default_rng(seed)
    policies: list[PolicyAttributes] = []
    for archetype in STYLE_NAMES:
        ranges = spec[archetype]
        for idx in range(variants_per_archetype):
            attrs = {}
            for name in ATTRIBUTE_NAMES:
                lo, hi = ranges[name]
                attrs[name] = float(rng.uniform(float(lo), float(hi)))
            policies.append(
                PolicyAttributes(
                    archetype=archetype,
                    policy_id=f"{archetype}_{idx:03d}",
                    seed=int(rng.integers(1, 10_000_000)),
                    **attrs,
                )
            )
    return policies


def write_default_policy_spec(path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_POLICY_SPEC, indent=2), encoding="utf-8")
    return str(path)


def policy_attributes_to_rows(policies: Iterable[PolicyAttributes]) -> list[dict[str, object]]:
    rows = []
    for policy in policies:
        row = asdict(policy)
        for name, value in zip(ATTRIBUTE_NAMES, policy.vector()):
            row[name] = float(value)
        rows.append(row)
    return rows


def _validate_policy_spec(spec: Dict[str, Dict[str, list[float]]]) -> None:
    missing_archetypes = [name for name in STYLE_NAMES if name not in spec]
    if missing_archetypes:
        raise ValueError(f"Policy spec missing archetypes: {missing_archetypes}")
    for archetype, ranges in spec.items():
        for name in ATTRIBUTE_NAMES:
            if name not in ranges:
                raise ValueError(f"Policy spec {archetype!r} missing attribute {name!r}")
            value = ranges[name]
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"Policy spec {archetype!r}/{name!r} must be [min, max]")
            lo, hi = float(value[0]), float(value[1])
            if lo < 0.0 or hi > 1.0 or lo > hi:
                raise ValueError(f"Policy spec {archetype!r}/{name!r} must be within 0..1 and min <= max")


def _distance_need(dist: float, target: float) -> float:
    return float(np.clip((dist - target) / 2.0, 0.0, 1.0))


def _reactive_opponent_action(env: FightEnv, fighter_idx: int) -> int:
    own, opp = env._by_idx(fighter_idx)
    if own.fallen:
        return ACTION_TO_ID["recover"]
    dist = float(np.linalg.norm(opp.pos() - own.pos()))
    if own.balance < 0.18:
        return ACTION_TO_ID["recover"]
    if own.stamina < 0.16:
        return ACTION_TO_ID["guard"]
    if dist > 1.25:
        return ACTION_TO_ID["step_forward"]
    if own.cooldown > 0:
        return ACTION_TO_ID["guard"]
    if dist < 0.80:
        return ACTION_TO_ID["push"]
    return ACTION_TO_ID["jab"]
