from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .config import (
    ACTION_NAMES,
    ACTION_TO_ID,
    ATTACK_ACTIONS,
    ATTACK_TABLE,
    MOVEMENT_ACTIONS,
    OBS_VERSION,
    SimConfig,
)


def wrap_angle(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def unit_from_angle(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)


def rotate_to_local(vec: np.ndarray, theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([c * vec[0] + s * vec[1], -s * vec[0] + c * vec[1]], dtype=np.float32)


@dataclass
class FighterState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    theta: float = 0.0
    omega: float = 0.0
    health: float = 100.0
    stamina: float = 1.0
    balance: float = 1.0
    guard: float = 0.25
    cooldown: int = 0
    fallen: bool = False
    recovery_timer: int = 0
    score: float = 0.0
    falls: int = 0
    last_action: int = 0
    landed_hit: float = 0.0
    got_hit: float = 0.0
    blocked: float = 0.0
    whiffed: float = 0.0
    # Damage accumulates from 0 to 1. Higher damage reduces action quality.
    left_arm_damage: float = 0.0
    right_arm_damage: float = 0.0
    left_leg_damage: float = 0.0
    right_leg_damage: float = 0.0
    core_damage: float = 0.0
    # Trace-only fields used for rendering/explanations.
    last_risk: float = 0.0
    last_override: bool = False
    last_proposed_action: int = 0

    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def vel(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=np.float32)

    def damage_vector(self) -> np.ndarray:
        return np.array(
            [
                self.left_arm_damage,
                self.right_arm_damage,
                self.left_leg_damage,
                self.right_leg_damage,
                self.core_damage,
            ],
            dtype=np.float32,
        )

    def copy(self) -> "FighterState":
        return copy.deepcopy(self)


@dataclass
class StepEvent:
    kind: str
    actor: int
    target: int
    action: int
    value: float = 0.0
    text: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "actor": self.actor,
            "target": self.target,
            "action": ACTION_NAMES[self.action],
            "value": self.value,
            "text": self.text,
        }


@dataclass
class MatchTrace:
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    proposed_actions: List[int] = field(default_factory=list)
    styles: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    risks: List[float] = field(default_factory=list)
    overrides: List[bool] = field(default_factory=list)


class FightEnv:
    """A deterministic, high-level humanoid combat simulator.

    The state uses approximate humanoid combat dynamics rather than raw torques:
    balance, cooldown, action range/cone, guard, boundary pressure, actuator damage,
    and knockdowns. It is built to test autonomy architecture quickly: data logging,
    style cloning, safety filtering, and batch evaluation.
    """

    metadata = {"obs_version": OBS_VERSION}

    def __init__(self, config: SimConfig | None = None, seed: int | None = None):
        self.config = config or SimConfig()
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.red: FighterState
        self.blue: FighterState
        self.step_count = 0
        self.done = False
        self.last_events: List[StepEvent] = []
        self.reset()

    @property
    def fighters(self) -> Tuple[FighterState, FighterState]:
        return self.red, self.blue

    def clone(self) -> "FightEnv":
        return copy.deepcopy(self)

    def reset(self, randomize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        c = self.config
        if randomize:
            angle = float(self.rng.uniform(-0.35, 0.35))
            offset = float(self.rng.uniform(-0.25, 0.25))
            sep = float(self.rng.uniform(2.0, 2.7))
        else:
            angle = 0.0
            offset = 0.0
            sep = 2.35
        dx, dy = math.cos(angle) * sep / 2, math.sin(angle) * sep / 2
        ortho = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32) * offset
        self.red = FighterState(x=-dx + float(ortho[0]), y=-dy + float(ortho[1]), theta=angle)
        self.blue = FighterState(x=dx + float(ortho[0]), y=dy + float(ortho[1]), theta=wrap_angle(angle + math.pi))
        # Preserve a tiny amount of population heterogeneity.
        self.red.balance = float(self.rng.uniform(0.82, 1.0)) if randomize else 1.0
        self.blue.balance = float(self.rng.uniform(0.82, 1.0)) if randomize else 1.0
        self.red.stamina = float(self.rng.uniform(0.82, 1.0)) if randomize else 1.0
        self.blue.stamina = float(self.rng.uniform(0.82, 1.0)) if randomize else 1.0
        self.step_count = 0
        self.done = False
        self.last_events = []
        self._face_each_other(force=True)
        return self.observe(0), self.observe(1)

    def observe(self, fighter_idx: int) -> np.ndarray:
        own, opp = self._by_idx(fighter_idx)
        c = self.config
        rel_world = opp.pos() - own.pos()
        rel_local = rotate_to_local(rel_world, own.theta)
        own_v_local = rotate_to_local(own.vel(), own.theta)
        rel_v_local = rotate_to_local(opp.vel() - own.vel(), own.theta)
        dist = float(np.linalg.norm(rel_world))
        bearing = math.atan2(rel_local[1], rel_local[0])
        boundary = self._boundary_features(own)
        opp_boundary = self._boundary_features(opp)
        last_one_hot = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        last_one_hot[int(own.last_action)] = 1.0
        opp_last_one_hot = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        opp_last_one_hot[int(opp.last_action)] = 1.0

        obs = np.concatenate(
            [
                np.array(
                    [
                        own.x / c.arena_radius,
                        own.y / c.arena_radius,
                        own_v_local[0] / c.max_speed,
                        own_v_local[1] / c.max_speed,
                        math.sin(own.theta),
                        math.cos(own.theta),
                        clamp(own.omega / 2.0, -1.0, 1.0),
                        own.health / 100.0,
                        own.stamina,
                        own.balance,
                        own.guard,
                        own.cooldown / 12.0,
                        1.0 if own.fallen else 0.0,
                        own.recovery_timer / max(1, c.recovery_steps),
                        own.landed_hit,
                        own.got_hit,
                        own.blocked,
                        own.whiffed,
                    ],
                    dtype=np.float32,
                ),
                own.damage_vector(),
                last_one_hot,
                np.array(
                    [
                        rel_local[0] / c.arena_radius,
                        rel_local[1] / c.arena_radius,
                        rel_v_local[0] / c.max_speed,
                        rel_v_local[1] / c.max_speed,
                        dist / (2 * c.arena_radius),
                        math.sin(bearing),
                        math.cos(bearing),
                        opp.health / 100.0,
                        opp.stamina,
                        opp.balance,
                        opp.guard,
                        opp.cooldown / 12.0,
                        1.0 if opp.fallen else 0.0,
                        opp.landed_hit,
                        opp.got_hit,
                        opp.blocked,
                        opp.whiffed,
                    ],
                    dtype=np.float32,
                ),
                opp.damage_vector(),
                opp_last_one_hot,
                boundary,
                opp_boundary,
                np.array([self.step_count / c.max_steps], dtype=np.float32),
            ]
        ).astype(np.float32)
        return obs

    @property
    def observation_dim(self) -> int:
        return int(self.observe(0).shape[0])

    def step(self, action_red: int, action_blue: int) -> Tuple[np.ndarray, np.ndarray, float, float, bool, Dict[str, object]]:
        if self.done:
            raise RuntimeError("FightEnv.step() called after the match is done. Call reset().")
        action_red = int(action_red)
        action_blue = int(action_blue)
        if not (0 <= action_red < len(ACTION_NAMES)) or not (0 <= action_blue < len(ACTION_NAMES)):
            raise ValueError("Action id out of range.")

        prev_red_score = self.red.score
        prev_blue_score = self.blue.score
        prev_red_health = self.red.health
        prev_blue_health = self.blue.health
        prev_red_falls = self.red.falls
        prev_blue_falls = self.blue.falls

        self.last_events = []
        self._pre_step_decay(self.red)
        self._pre_step_decay(self.blue)
        self._face_each_other(force=False)

        self._apply_action_setup(self.red, self.blue, action_red)
        self._apply_action_setup(self.blue, self.red, action_blue)

        # Resolve attacks after movement setup, so dodges and lunge distance matter.
        self._resolve_attack(0, action_red)
        self._resolve_attack(1, action_blue)

        self._integrate(self.red)
        self._integrate(self.blue)
        self._check_fall_and_bounds(self.red, self.blue, 0)
        self._check_fall_and_bounds(self.blue, self.red, 1)

        self.red.last_action = action_red
        self.blue.last_action = action_blue
        self.step_count += 1

        terminal = (
            self.step_count >= self.config.max_steps
            or self.red.health <= self.config.ko_health
            or self.blue.health <= self.config.ko_health
        )
        self.done = bool(terminal)

        red_score_delta = self.red.score - prev_red_score
        blue_score_delta = self.blue.score - prev_blue_score
        red_damage_delta = prev_blue_health - self.blue.health
        blue_damage_delta = prev_red_health - self.red.health
        red_fall_delta = self.red.falls - prev_red_falls
        blue_fall_delta = self.blue.falls - prev_blue_falls
        reward_red = (
            0.09 * red_score_delta
            + 0.22 * red_damage_delta
            - 0.18 * blue_damage_delta
            + 1.8 * blue_fall_delta
            - 1.8 * red_fall_delta
            + 0.015 * (self.red.balance - self.blue.balance)
        )
        reward_blue = (
            0.09 * blue_score_delta
            + 0.22 * blue_damage_delta
            - 0.18 * red_damage_delta
            + 1.8 * red_fall_delta
            - 1.8 * blue_fall_delta
            + 0.015 * (self.blue.balance - self.red.balance)
        )
        info = {
            "events": [e.as_dict() for e in self.last_events],
            "red": self.summary(0),
            "blue": self.summary(1),
        }
        return self.observe(0), self.observe(1), float(reward_red), float(reward_blue), self.done, info

    def winner(self) -> int:
        """Return 0 for red, 1 for blue, -1 for draw."""
        red_total = self.red.score + 0.5 * self.red.health + 7.5 * (self.blue.falls - self.red.falls)
        blue_total = self.blue.score + 0.5 * self.blue.health + 7.5 * (self.red.falls - self.blue.falls)
        if abs(red_total - blue_total) < 1.0:
            return -1
        return 0 if red_total > blue_total else 1

    def summary(self, idx: int) -> Dict[str, float | int | bool]:
        f, _ = self._by_idx(idx)
        return {
            "health": round(float(f.health), 3),
            "stamina": round(float(f.stamina), 3),
            "balance": round(float(f.balance), 3),
            "guard": round(float(f.guard), 3),
            "score": round(float(f.score), 3),
            "falls": int(f.falls),
            "fallen": bool(f.fallen),
            "cooldown": int(f.cooldown),
            "last_action": ACTION_NAMES[int(f.last_action)],
        }

    def _by_idx(self, idx: int) -> Tuple[FighterState, FighterState]:
        if idx == 0:
            return self.red, self.blue
        if idx == 1:
            return self.blue, self.red
        raise ValueError("fighter_idx must be 0 or 1")

    def _pre_step_decay(self, f: FighterState) -> None:
        f.landed_hit *= 0.15
        f.got_hit *= 0.15
        f.blocked *= 0.15
        f.whiffed *= 0.15
        f.last_override = False
        f.last_risk = 0.0
        f.guard *= self.config.guard_decay
        f.cooldown = max(0, f.cooldown - 1)
        if not f.fallen:
            leg_damage = 0.5 * (f.left_leg_damage + f.right_leg_damage)
            f.stamina = clamp(f.stamina + self.config.stamina_recovery * (1.0 - 0.35 * leg_damage), 0.0, 1.0)
            f.balance = clamp(f.balance + 0.018 * (1.0 - f.balance) - 0.004 * f.core_damage, self.config.min_balance, 1.0)
        else:
            f.vx *= 0.58
            f.vy *= 0.58
            f.omega *= 0.50
            f.recovery_timer = max(0, f.recovery_timer - 1)
            if f.recovery_timer <= 0:
                f.fallen = False
                f.balance = max(f.balance, 0.40)
                f.guard = max(f.guard, 0.25)

    def _face_each_other(self, force: bool = False) -> None:
        for own, opp in ((self.red, self.blue), (self.blue, self.red)):
            if own.fallen:
                continue
            desired = math.atan2(opp.y - own.y, opp.x - own.x)
            delta = wrap_angle(desired - own.theta)
            leg_damage = 0.5 * (own.left_leg_damage + own.right_leg_damage)
            turn_cap = self.config.turn_rate * (1.0 - 0.45 * leg_damage)
            if force:
                own.theta = desired
                own.omega = 0.0
            else:
                turn = clamp(delta, -turn_cap, turn_cap)
                own.theta = wrap_angle(own.theta + turn)
                own.omega = 0.68 * own.omega + 0.32 * turn / self.config.dt

    def _apply_action_setup(self, own: FighterState, opp: FighterState, action: int) -> None:
        c = self.config
        own.last_proposed_action = action
        if own.fallen:
            if action == ACTION_TO_ID["recover"]:
                own.recovery_timer = max(0, own.recovery_timer - 2)
                own.balance = clamp(own.balance + 0.06, c.min_balance, 0.65)
            return

        if action == ACTION_TO_ID["guard"]:
            own.guard = clamp(own.guard + 0.58, 0.0, 1.0)
            own.balance = clamp(own.balance + 0.030, c.min_balance, 1.0)
            own.stamina = clamp(own.stamina + 0.010, 0.0, 1.0)
        elif action == ACTION_TO_ID["recover"]:
            own.guard = clamp(own.guard + 0.20, 0.0, 1.0)
            own.balance = clamp(own.balance + 0.115 + 0.025 * own.stamina, c.min_balance, 1.0)
            own.stamina = clamp(own.stamina + 0.032, 0.0, 1.0)
            own.vx *= 0.62
            own.vy *= 0.62
        elif action in MOVEMENT_ACTIONS:
            self._movement(own, opp, action)
        elif action in ATTACK_ACTIONS:
            # Small pre-attack body commitment/lunge. Actual hit resolution comes later.
            params = ATTACK_TABLE[action]
            forward_lunge = params[-1]
            heading = unit_from_angle(own.theta)
            stamina_factor = 0.45 + 0.55 * own.stamina
            own.vx += float(heading[0]) * forward_lunge * stamina_factor
            own.vy += float(heading[1]) * forward_lunge * stamina_factor
            own.guard *= 0.45
        else:
            own.balance = clamp(own.balance + 0.006, c.min_balance, 1.0)

    def _movement(self, own: FighterState, opp: FighterState, action: int) -> None:
        c = self.config
        fwd = unit_from_angle(own.theta)
        left = np.array([-fwd[1], fwd[0]], dtype=np.float32)
        dist = float(np.linalg.norm(opp.pos() - own.pos()))
        leg_damage = 0.5 * (own.left_leg_damage + own.right_leg_damage)
        speed = c.max_speed * (0.55 + 0.45 * own.stamina) * (1.0 - 0.55 * leg_damage)
        vec = np.zeros(2, dtype=np.float32)
        balance_cost = 0.014
        stamina_cost = 0.018
        if action == ACTION_TO_ID["step_forward"]:
            vec = fwd * speed
            balance_cost = 0.026 if dist < 0.95 else 0.018
            stamina_cost = 0.028
        elif action == ACTION_TO_ID["step_back"]:
            vec = -fwd * speed * 0.74
            own.guard = clamp(own.guard + 0.10, 0.0, 1.0)
            balance_cost = 0.012
            stamina_cost = 0.014
        elif action == ACTION_TO_ID["sidestep_left"]:
            vec = left * speed * 0.82
            balance_cost = 0.036
            stamina_cost = 0.030
        elif action == ACTION_TO_ID["sidestep_right"]:
            vec = -left * speed * 0.82
            balance_cost = 0.036
            stamina_cost = 0.030
        elif action == ACTION_TO_ID["circle_left"]:
            vec = left * speed * 0.72 + fwd * (0.18 * (1.3 - dist))
            balance_cost = 0.031
            stamina_cost = 0.027
        elif action == ACTION_TO_ID["circle_right"]:
            vec = -left * speed * 0.72 + fwd * (0.18 * (1.3 - dist))
            balance_cost = 0.031
            stamina_cost = 0.027
        own.vx += float(vec[0]) * c.dt
        own.vy += float(vec[1]) * c.dt
        own.stamina = clamp(own.stamina - stamina_cost, 0.0, 1.0)
        own.balance = clamp(own.balance - balance_cost * (0.55 + 0.45 * (1.0 - own.stamina)), c.min_balance, 1.0)

    def _resolve_attack(self, attacker_idx: int, action: int) -> None:
        if action not in ATTACK_ACTIONS:
            return
        attacker, defender = self._by_idx(attacker_idx)
        defender_idx = 1 - attacker_idx
        if attacker.fallen:
            return
        if attacker.cooldown > 0:
            attacker.whiffed = 1.0
            attacker.balance = clamp(attacker.balance - 0.020, self.config.min_balance, 1.0)
            return

        range_, damage, stamina_cost, cooldown, self_cost, balance_impact, guard_break, cone_cos, _ = ATTACK_TABLE[action]
        stamina_factor = clamp(0.35 + 0.65 * attacker.stamina, 0.25, 1.0)
        arm_damage = 0.0
        leg_damage = 0.0
        if action in (ACTION_TO_ID["jab"], ACTION_TO_ID["hook"]):
            arm_damage = attacker.left_arm_damage
        elif action in (ACTION_TO_ID["cross"], ACTION_TO_ID["push"]):
            arm_damage = attacker.right_arm_damage
        elif action == ACTION_TO_ID["low_kick"]:
            leg_damage = 0.5 * (attacker.left_leg_damage + attacker.right_leg_damage)
        capability = clamp(1.0 - 0.50 * arm_damage - 0.55 * leg_damage - 0.18 * attacker.core_damage, 0.25, 1.0)

        rel = defender.pos() - attacker.pos()
        dist = float(np.linalg.norm(rel)) + 1e-6
        direction = rel / dist
        facing = unit_from_angle(attacker.theta)
        aim_cos = float(np.dot(direction, facing))
        lateral_velocity = abs(float(np.cross(np.append(direction, 0.0), np.append(defender.vel(), 0.0))[2]))
        evasion_penalty = 0.03 * lateral_velocity
        hit_quality = aim_cos - cone_cos - evasion_penalty + 0.04 * attacker.stamina
        in_range = dist <= range_ * (0.96 + 0.10 * stamina_factor)
        hit = bool(in_range and hit_quality >= -0.02 and not defender.fallen)

        attacker.stamina = clamp(attacker.stamina - stamina_cost, 0.0, 1.0)
        attacker.balance = clamp(attacker.balance - self_cost * (1.12 - attacker.stamina), self.config.min_balance, 1.0)
        attacker.cooldown = cooldown

        if not hit:
            attacker.whiffed = 1.0
            attacker.balance = clamp(attacker.balance - self_cost * 0.70, self.config.min_balance, 1.0)
            self.last_events.append(
                StepEvent("whiff", attacker_idx, defender_idx, action, 0.0, f"{ACTION_NAMES[action]} missed")
            )
            return

        guard_power = defender.guard
        if defender.last_action == ACTION_TO_ID["guard"]:
            guard_power = clamp(guard_power + 0.28, 0.0, 1.0)
        if defender.last_action in (ACTION_TO_ID["sidestep_left"], ACTION_TO_ID["sidestep_right"], ACTION_TO_ID["circle_left"], ACTION_TO_ID["circle_right"]):
            guard_power = clamp(guard_power + 0.09, 0.0, 1.0)
        block_ratio = clamp(0.08 + 0.70 * guard_power - guard_break, 0.0, 0.82)
        landed_damage = damage * stamina_factor * capability * (1.0 - block_ratio)
        if defender.fallen:
            landed_damage *= 0.25
        defender.health = max(-5.0, defender.health - landed_damage)
        attacker.score += landed_damage
        attacker.landed_hit = 1.0
        defender.got_hit = 1.0
        if block_ratio > 0.42:
            defender.blocked = 1.0
            defender.score += 0.55
        defender.guard = clamp(defender.guard - 0.18 - guard_break * 0.4, 0.0, 1.0)
        defender.balance = clamp(
            defender.balance - balance_impact * (1.0 - 0.45 * block_ratio) * (0.75 + 0.35 * stamina_factor),
            self.config.min_balance,
            1.0,
        )
        # Damage to components matters because it changes future action safety.
        limb_delta = clamp(landed_damage / 120.0, 0.0, 0.12)
        if action == ACTION_TO_ID["low_kick"]:
            if self.rng.random() < 0.5:
                defender.left_leg_damage = clamp(defender.left_leg_damage + limb_delta * 1.7, 0.0, 1.0)
            else:
                defender.right_leg_damage = clamp(defender.right_leg_damage + limb_delta * 1.7, 0.0, 1.0)
        elif action in (ACTION_TO_ID["jab"], ACTION_TO_ID["cross"]):
            defender.core_damage = clamp(defender.core_damage + limb_delta * 0.75, 0.0, 1.0)
        elif action == ACTION_TO_ID["hook"]:
            defender.core_damage = clamp(defender.core_damage + limb_delta * 1.25, 0.0, 1.0)
            defender.left_arm_damage = clamp(defender.left_arm_damage + limb_delta * 0.40, 0.0, 1.0)
            defender.right_arm_damage = clamp(defender.right_arm_damage + limb_delta * 0.40, 0.0, 1.0)
        elif action == ACTION_TO_ID["push"]:
            defender.core_damage = clamp(defender.core_damage + limb_delta * 0.30, 0.0, 1.0)

        knock = direction * balance_impact * (0.7 + 0.6 * (1.0 - block_ratio))
        defender.vx += float(knock[0])
        defender.vy += float(knock[1])
        defender.omega += float(self.rng.normal(0.0, 0.16)) + (0.10 if action == ACTION_TO_ID["hook"] else 0.0)
        self.last_events.append(
            StepEvent("hit", attacker_idx, defender_idx, action, float(landed_damage), f"{ACTION_NAMES[action]} landed")
        )

    def _integrate(self, f: FighterState) -> None:
        c = self.config
        speed = math.hypot(f.vx, f.vy)
        if speed > c.max_speed * 1.8:
            scale = c.max_speed * 1.8 / max(speed, 1e-6)
            f.vx *= scale
            f.vy *= scale
        f.x += f.vx * c.dt
        f.y += f.vy * c.dt
        f.vx *= c.velocity_decay
        f.vy *= c.velocity_decay
        f.theta = wrap_angle(f.theta + f.omega * c.dt * 0.10)
        f.omega *= 0.78
        # Fast lateral motion and spin are dangerous for humanoids under contact.
        dynamic_penalty = 0.006 * min(3.0, speed) + 0.004 * min(4.0, abs(f.omega))
        if not f.fallen:
            f.balance = clamp(f.balance - dynamic_penalty, c.min_balance, 1.0)

    def _check_fall_and_bounds(self, f: FighterState, opp: FighterState, idx: int) -> None:
        c = self.config
        radius = math.hypot(f.x, f.y)
        if radius > c.arena_radius:
            # Push back inside and penalize boundary loss.
            nx, ny = f.x / max(radius, 1e-6), f.y / max(radius, 1e-6)
            over = radius - c.arena_radius
            f.x = nx * c.arena_radius
            f.y = ny * c.arena_radius
            f.vx -= nx * (0.4 + over)
            f.vy -= ny * (0.4 + over)
            f.balance = clamp(f.balance - c.boundary_penalty * (1.0 + over), c.min_balance, 1.0)
            opp.score += c.boundary_score
            self.last_events.append(StepEvent("boundary", 1 - idx, idx, opp.last_action, c.boundary_score, "ring pressure"))
        if f.fallen:
            return
        damage_drag = 0.08 * f.core_damage + 0.06 * (f.left_leg_damage + f.right_leg_damage)
        effective_balance = f.balance - damage_drag
        fall_prob = 0.0
        if effective_balance < c.fall_balance_threshold:
            fall_prob = clamp((c.fall_balance_threshold - effective_balance) * 2.4, 0.0, 0.95)
        if effective_balance < -0.06 or self.rng.random() < fall_prob:
            f.fallen = True
            f.recovery_timer = c.recovery_steps
            f.falls += 1
            f.balance = min(f.balance, -0.05)
            f.health = max(-5.0, f.health - c.knockdown_damage)
            opp.score += c.fall_score
            self.last_events.append(StepEvent("fall", 1 - idx, idx, opp.last_action, c.fall_score, "knockdown"))

    def _boundary_features(self, f: FighterState) -> np.ndarray:
        c = self.config
        pos = f.pos()
        heading = unit_from_angle(f.theta)
        left = np.array([-heading[1], heading[0]], dtype=np.float32)
        # Clearance in four local directions, normalized. Used by policy/safety.
        dirs = [heading, -heading, left, -left]
        out = []
        for d in dirs:
            # Solve ||pos + t d|| = arena_radius for positive t.
            b = 2.0 * float(np.dot(pos, d))
            cc = float(np.dot(pos, pos) - c.arena_radius**2)
            disc = max(0.0, b * b - 4 * cc)
            t = (-b + math.sqrt(disc)) / 2.0
            out.append(clamp(t / c.arena_radius, 0.0, 1.0))
        return np.asarray(out, dtype=np.float32)

    def style_features(self, idx: int) -> Dict[str, float]:
        """Human-readable behavior fingerprint for dashboards."""
        f, _ = self._by_idx(idx)
        return {
            "health": f.health,
            "balance": f.balance,
            "stamina": f.stamina,
            "guard": f.guard,
            "score": f.score,
            "falls": float(f.falls),
            "arm_damage": 0.5 * (f.left_arm_damage + f.right_arm_damage),
            "leg_damage": 0.5 * (f.left_leg_damage + f.right_leg_damage),
            "core_damage": f.core_damage,
        }
