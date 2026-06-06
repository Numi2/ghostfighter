from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import ACTION_NAMES, ACTION_TO_ID, ATTACK_ACTIONS, ATTACK_TABLE, MOVEMENT_ACTIONS
from .env import FightEnv, FighterState, rotate_to_local, clamp


@dataclass
class SafetyDecision:
    proposed_action: int
    action: int
    risk: float
    overridden: bool
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "proposed_action": ACTION_NAMES[self.proposed_action],
            "action": ACTION_NAMES[self.action],
            "risk": float(self.risk),
            "overridden": bool(self.overridden),
            "reason": self.reason,
        }


class CombatSafetyFirewall:
    """Pre-controller safety gate for high-level humanoid combat actions.

    It estimates whether the requested skill token is likely to cause a fall,
    boundary loss, or actuator-stress event. Unsafe requests are replaced by a
    defensive/recovery action before they reach the simulator controller.
    """

    def __init__(self, threshold: float = 0.62):
        self.threshold = float(threshold)
        self.rejections = 0
        self.total = 0
        self.reason_counts: Dict[str, int] = {}

    def reset(self) -> None:
        self.rejections = 0
        self.total = 0
        self.reason_counts = {}

    def filter(self, env: FightEnv, fighter_idx: int, proposed_action: int) -> SafetyDecision:
        self.total += 1
        risk, reason = self.estimate_risk(env, fighter_idx, proposed_action)
        if risk < self.threshold:
            self._write_trace(env, fighter_idx, proposed_action, proposed_action, risk, False)
            return SafetyDecision(proposed_action, proposed_action, risk, False, reason)
        replacement = self.replacement_action(env, fighter_idx, proposed_action, reason)
        self.rejections += 1
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        self._write_trace(env, fighter_idx, proposed_action, replacement, risk, True)
        return SafetyDecision(proposed_action, replacement, risk, True, reason)

    def _write_trace(self, env: FightEnv, idx: int, proposed: int, action: int, risk: float, overridden: bool) -> None:
        f, _ = env._by_idx(idx)
        f.last_risk = float(risk)
        f.last_override = bool(overridden)
        f.last_proposed_action = int(proposed)

    def estimate_risk(self, env: FightEnv, fighter_idx: int, action: int) -> Tuple[float, str]:
        own, opp = env._by_idx(fighter_idx)
        if own.fallen:
            return (0.92, "fallen_requires_recover") if action != ACTION_TO_ID["recover"] else (0.15, "already_recovering")

        c = env.config
        risk_terms: list[tuple[float, str]] = []
        leg_damage = 0.5 * (own.left_leg_damage + own.right_leg_damage)
        arm_damage = 0.5 * (own.left_arm_damage + own.right_arm_damage)
        rel = opp.pos() - own.pos()
        rel_local = rotate_to_local(rel, own.theta)
        dist = float(np.linalg.norm(rel))
        boundary_clearance = max(0.0, c.arena_radius - float(np.linalg.norm(own.pos()))) / c.arena_radius
        speed = float(np.linalg.norm(own.vel()))
        opp_threat = opp.last_action in ATTACK_ACTIONS and dist < 1.35 and opp.cooldown > 0

        base = 0.05
        # Balance is the dominant safety indicator.
        if own.balance < 0.20:
            risk_terms.append((0.48 + (0.20 - own.balance) * 1.6, "low_balance"))
        elif own.balance < 0.36:
            risk_terms.append((0.25 + (0.36 - own.balance), "fragile_balance"))
        if own.stamina < 0.16 and action in ATTACK_ACTIONS.union(MOVEMENT_ACTIONS):
            risk_terms.append((0.24 + (0.16 - own.stamina), "low_stamina"))
        if boundary_clearance < 0.16 and self._moves_toward_boundary(env, own, action):
            risk_terms.append((0.42 + (0.16 - boundary_clearance), "boundary_escape_needed"))
        if leg_damage > 0.45 and action in {
            ACTION_TO_ID["sidestep_left"],
            ACTION_TO_ID["sidestep_right"],
            ACTION_TO_ID["circle_left"],
            ACTION_TO_ID["circle_right"],
            ACTION_TO_ID["low_kick"],
        }:
            risk_terms.append((0.31 + 0.42 * leg_damage, "leg_damage_mobility_risk"))
        if arm_damage > 0.50 and action in {ACTION_TO_ID["hook"], ACTION_TO_ID["cross"], ACTION_TO_ID["push"]}:
            risk_terms.append((0.16 + 0.28 * arm_damage, "arm_damage_strike_risk"))
        if speed > 1.95 and action in ATTACK_ACTIONS:
            risk_terms.append((0.26 + 0.12 * min(speed, 3.5), "high_momentum_strike"))
        if abs(own.omega) > 2.3 and action not in {ACTION_TO_ID["recover"], ACTION_TO_ID["guard"]}:
            risk_terms.append((0.30 + 0.08 * min(abs(own.omega), 4.5), "angular_instability"))
        if own.cooldown > 0 and action in ATTACK_ACTIONS:
            risk_terms.append((0.32 + 0.04 * own.cooldown, "cooldown_forced_whiff"))
        if opp_threat and action in {ACTION_TO_ID["hook"], ACTION_TO_ID["low_kick"], ACTION_TO_ID["step_forward"]} and own.guard < 0.28:
            risk_terms.append((0.28 + 0.24 * (1.0 - own.guard), "incoming_contact_no_guard"))
        if action in ATTACK_ACTIONS:
            params = ATTACK_TABLE[action]
            _, _, stamina_cost, _, self_cost, _, _, cone_cos, _ = params
            aim_quality = rel_local[0] / max(dist, 1e-6)
            if dist > params[0] * 1.22 or aim_quality < cone_cos - 0.15:
                risk_terms.append((0.16 + self_cost * 3.0, "likely_whiff"))
            if own.balance - self_cost * 1.35 - 0.05 * leg_damage < 0.05:
                risk_terms.append((0.36 + self_cost * 2.0, "strike_breaks_balance"))
            if own.stamina - stamina_cost < 0.05:
                risk_terms.append((0.22 + stamina_cost, "stamina_exhaustion"))

        if action == ACTION_TO_ID["recover"]:
            base -= 0.03
        if action == ACTION_TO_ID["guard"]:
            base -= 0.02

        if not risk_terms:
            return clamp(base, 0.0, 1.0), "nominal"
        # Aggregate conservatively but avoid saturating from a single minor issue.
        risk_terms_sorted = sorted(risk_terms, reverse=True, key=lambda x: x[0])
        risk = base
        for val, _ in risk_terms_sorted[:4]:
            risk = 1.0 - (1.0 - risk) * (1.0 - clamp(val, 0.0, 0.96))
        return clamp(risk, 0.0, 1.0), risk_terms_sorted[0][1]

    def replacement_action(self, env: FightEnv, fighter_idx: int, proposed_action: int, reason: str) -> int:
        own, opp = env._by_idx(fighter_idx)
        if own.fallen:
            return ACTION_TO_ID["recover"]
        boundary_clearance = max(0.0, env.config.arena_radius - float(np.linalg.norm(own.pos()))) / env.config.arena_radius
        rel = opp.pos() - own.pos()
        rel_local = rotate_to_local(rel, own.theta)
        dist = float(np.linalg.norm(rel))
        if reason in {"low_balance", "fragile_balance", "strike_breaks_balance", "angular_instability"}:
            return ACTION_TO_ID["recover"] if own.balance < 0.33 else ACTION_TO_ID["guard"]
        if reason == "boundary_escape_needed":
            # If backed into boundary, move toward opponent/center rather than retreating.
            return ACTION_TO_ID["step_forward"] if rel_local[0] > -0.2 else ACTION_TO_ID["sidestep_left"]
        if reason in {"incoming_contact_no_guard", "cooldown_forced_whiff", "arm_damage_strike_risk"}:
            return ACTION_TO_ID["guard"]
        if reason in {"leg_damage_mobility_risk", "low_stamina", "stamina_exhaustion"}:
            return ACTION_TO_ID["recover"] if dist > 1.0 else ACTION_TO_ID["guard"]
        if reason == "likely_whiff":
            return ACTION_TO_ID["step_forward"] if dist > 1.15 and boundary_clearance > 0.18 else ACTION_TO_ID["guard"]
        if reason == "high_momentum_strike":
            return ACTION_TO_ID["guard"]
        return ACTION_TO_ID["guard"]

    def _moves_toward_boundary(self, env: FightEnv, own: FighterState, action: int) -> bool:
        if action not in MOVEMENT_ACTIONS and action not in ATTACK_ACTIONS:
            return False
        pos = own.pos()
        radius = np.linalg.norm(pos)
        if radius < 1e-6:
            return False
        radial = pos / radius
        heading = np.array([np.cos(own.theta), np.sin(own.theta)], dtype=np.float32)
        left = np.array([-heading[1], heading[0]], dtype=np.float32)
        if action == ACTION_TO_ID["step_forward"]:
            vec = heading
        elif action == ACTION_TO_ID["step_back"]:
            vec = -heading
        elif action in (ACTION_TO_ID["sidestep_left"], ACTION_TO_ID["circle_left"]):
            vec = left
        elif action in (ACTION_TO_ID["sidestep_right"], ACTION_TO_ID["circle_right"]):
            vec = -left
        elif action in ATTACK_ACTIONS:
            vec = heading
        else:
            vec = np.zeros(2, dtype=np.float32)
        return float(np.dot(vec, radial)) > 0.18

    @property
    def rejection_rate(self) -> float:
        return 0.0 if self.total == 0 else self.rejections / self.total
