from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict

import numpy as np

from .config import SimConfig
from .env import FightEnv, clamp


@dataclass(frozen=True)
class DomainRandomizationProfile:
    mass_scale: float = 1.0
    inertia_scale: float = 1.0
    friction_scale: float = 1.0
    floor_compliance: float = 0.0
    latency_steps: int = 0
    motor_strength: float = 1.0
    actuator_delay: float = 0.0
    joint_damping: float = 1.0
    imu_noise: float = 0.0
    encoder_noise: float = 0.0
    contact_restitution: float = 0.0
    battery_voltage: float = 1.0
    thermal_limit: float = 1.0
    terrain_roughness: float = 0.0
    external_push: float = 0.0


def sample_domain_randomization(rng: np.random.Generator, intensity: float = 1.0) -> DomainRandomizationProfile:
    """Sample sim-to-real perturbations for the high-level backend.

    The current backend is intentionally compact, so these parameters are projected
    onto equivalent high-level effects: speed, balance recovery, damping, contact
    bounce, state noise, and push impulses. Isaac/MuJoCo backends can consume the
    same profile names directly.
    """
    s = float(np.clip(intensity, 0.0, 1.0))
    return DomainRandomizationProfile(
        mass_scale=float(rng.uniform(1.0 - 0.18 * s, 1.0 + 0.22 * s)),
        inertia_scale=float(rng.uniform(1.0 - 0.20 * s, 1.0 + 0.30 * s)),
        friction_scale=float(rng.uniform(1.0 - 0.35 * s, 1.0 + 0.25 * s)),
        floor_compliance=float(rng.uniform(0.0, 0.35 * s)),
        latency_steps=int(rng.integers(0, 1 + int(round(3 * s)))),
        motor_strength=float(rng.uniform(1.0 - 0.28 * s, 1.0 + 0.15 * s)),
        actuator_delay=float(rng.uniform(0.0, 0.22 * s)),
        joint_damping=float(rng.uniform(1.0 - 0.22 * s, 1.0 + 0.35 * s)),
        imu_noise=float(rng.uniform(0.0, 0.045 * s)),
        encoder_noise=float(rng.uniform(0.0, 0.035 * s)),
        contact_restitution=float(rng.uniform(0.0, 0.30 * s)),
        battery_voltage=float(rng.uniform(1.0 - 0.24 * s, 1.0)),
        thermal_limit=float(rng.uniform(1.0 - 0.20 * s, 1.0)),
        terrain_roughness=float(rng.uniform(0.0, 0.16 * s)),
        external_push=float(rng.uniform(0.0, 0.34 * s)),
    )


def apply_domain_randomization(env: FightEnv, profile: DomainRandomizationProfile) -> None:
    strength = profile.motor_strength * profile.battery_voltage * profile.thermal_limit
    damping = profile.joint_damping * profile.friction_scale
    max_speed = env.config.max_speed * np.clip(strength / max(profile.mass_scale, 1e-6), 0.55, 1.25)
    velocity_decay = np.clip(env.config.velocity_decay * (0.92 + 0.10 * damping) - 0.08 * profile.floor_compliance, 0.58, 0.90)
    stamina_recovery = env.config.stamina_recovery * np.clip(profile.battery_voltage * profile.thermal_limit, 0.60, 1.05)
    turn_rate = env.config.turn_rate * np.clip(strength / max(profile.inertia_scale, 1e-6), 0.55, 1.18)
    boundary_penalty = env.config.boundary_penalty * (1.0 + 0.45 * profile.terrain_roughness + 0.30 * profile.floor_compliance)
    env.config = replace(
        env.config,
        max_speed=float(max_speed),
        velocity_decay=float(velocity_decay),
        stamina_recovery=float(stamina_recovery),
        turn_rate=float(turn_rate),
        boundary_penalty=float(boundary_penalty),
    )
    for fighter in env.fighters:
        fighter.balance = clamp(fighter.balance - 0.10 * profile.terrain_roughness - 0.04 * profile.actuator_delay, env.config.min_balance, 1.0)
        fighter.stamina = clamp(fighter.stamina * (0.92 + 0.08 * profile.battery_voltage), 0.0, 1.0)
        fighter.omega += float(env.rng.normal(0.0, 0.05 * profile.imu_noise))


def apply_observation_noise(obs: np.ndarray, rng: np.random.Generator, profile: DomainRandomizationProfile) -> np.ndarray:
    sigma = profile.imu_noise + profile.encoder_noise
    if sigma <= 0:
        return obs
    return (obs + rng.normal(0.0, sigma, size=obs.shape)).astype(np.float32)


def apply_external_push(env: FightEnv, rng: np.random.Generator, profile: DomainRandomizationProfile) -> None:
    if profile.external_push <= 0 or rng.random() > 0.035:
        return
    target = env.red if rng.random() < 0.5 else env.blue
    angle = float(rng.uniform(-np.pi, np.pi))
    impulse = profile.external_push
    target.vx += float(np.cos(angle) * impulse)
    target.vy += float(np.sin(angle) * impulse)
    target.balance = clamp(target.balance - 0.04 * impulse, env.config.min_balance, 1.0)


def summarize_domain_profiles(profiles: list[DomainRandomizationProfile]) -> Dict[str, object]:
    if not profiles:
        return {"enabled": False}
    rows = [asdict(p) for p in profiles]
    summary: Dict[str, object] = {"enabled": True, "profiles": len(rows)}
    for key in rows[0]:
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {"min": float(vals.min()), "mean": float(vals.mean()), "max": float(vals.max())}
    return summary


def write_domain_randomization_card(path: str | Path, summary: Dict[str, object]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Domain Randomization Card

GhostFighter randomizes high-level equivalents of standard robotics sim-to-real variables: mass, inertia, friction, floor compliance, latency, motor strength, actuator delay, joint damping, IMU noise, encoder noise, contact restitution, battery voltage sag, thermal limits, terrain, and external pushes.

The compact backend projects those variables onto speed, damping, balance recovery, contact instability, sensor noise, and impulse disturbances. Isaac Lab or MuJoCo backends can consume the same profile schema at higher fidelity.

```json
{json.dumps(summary, indent=2)}
```
"""
    path.write_text(text, encoding="utf-8")
    return str(path)
