from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

from .attributes import AttributePolicy
from .domain import apply_domain_randomization, apply_external_push, apply_observation_noise, sample_domain_randomization
from .env import FightEnv, SimConfig
from .rl import NeuralActorPolicy, load_actor_checkpoint
from .selfplay import _make_population


@dataclass(frozen=True)
class RobustnessAblation:
    name: str
    description: str
    domain_randomization: bool = False
    domain_intensity: float = 0.0
    setup: Callable[[FightEnv, np.random.Generator], None] | None = None


def run_robustness_ablations(
    policy_path: str | Path,
    out_dir: str | Path,
    episodes: int = 12,
    seed: int = 2401,
    max_steps: int = 80,
) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_actor_checkpoint(policy_path)
    role_population = _make_population(variants_per_role=1, seed=seed + 11)
    rng = np.random.default_rng(seed)
    rows = []
    ablations = [
        RobustnessAblation("clean", "Nominal simulator with no extra stress."),
        RobustnessAblation("domain_randomized", "Randomized dynamics, sensors, derating, terrain, and pushes.", True, 0.60),
        RobustnessAblation("actuator_degraded", "Asymmetric actuator damage and low motor authority.", True, 0.75, _setup_actuator_degraded),
        RobustnessAblation("latency_boundary", "Boundary pressure with noisy latency-like observations.", True, 0.45, _setup_boundary_pressure),
        RobustnessAblation("recovery_stress", "Low balance recovery under external disturbances.", True, 0.65, _setup_recovery_stress),
    ]
    for ablation in ablations:
        for ep in range(episodes):
            opponent_attrs = role_population[int(rng.integers(0, len(role_population)))]
            result = _run_ablation_match(model, opponent_attrs, ablation, int(rng.integers(1, 10_000_000)), max_steps)
            rows.append({"ablation": ablation.name, "description": ablation.description, "episode": ep, **result})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "robustness_results.csv", index=False)
    summary = _summarize(df, ckpt)
    (out_dir / "robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(out_dir / "ROBUSTNESS_REPORT.md", summary)
    _make_dashboard(out_dir / "robustness_dashboard.png", df)
    return {"summary": summary, "rows": rows}


def _run_ablation_match(model, opponent_attrs, ablation: RobustnessAblation, seed: int, max_steps: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    env = FightEnv(config=SimConfig(max_steps=max_steps, seed=seed), seed=seed)
    obs_red, obs_blue = env.reset(randomize=True)
    profile = None
    if ablation.setup:
        ablation.setup(env, rng)
        obs_red, obs_blue = env.observe(0), env.observe(1)
    if ablation.domain_randomization:
        profile = sample_domain_randomization(rng, ablation.domain_intensity)
        apply_domain_randomization(env, profile)
        obs_red = apply_observation_noise(env.observe(0), rng, profile)
        obs_blue = apply_observation_noise(env.observe(1), rng, profile)
    red = NeuralActorPolicy(model, deterministic=True)
    blue = AttributePolicy(opponent_attrs, lookahead=False)
    boundary_losses = 0
    done = False
    while not done:
        obs_red, obs_blue, _rr, _rb, done, info = env.step(red.select_action(obs_red, env, 0), blue.select_action(obs_blue, env, 1))
        boundary_losses += sum(1 for event in info["events"] if event["kind"] == "boundary" and event["target"] == 0)
        if profile is not None and not done:
            apply_external_push(env, rng, profile)
            obs_red = apply_observation_noise(env.observe(0), rng, profile)
            obs_blue = apply_observation_noise(env.observe(1), rng, profile)
    return {
        "winner": int(env.winner()),
        "win": float(1.0 if env.winner() == 0 else 0.5 if env.winner() == -1 else 0.0),
        "red_score": float(env.red.score),
        "blue_score": float(env.blue.score),
        "red_health": float(env.red.health),
        "blue_health": float(env.blue.health),
        "red_falls": int(env.red.falls),
        "blue_falls": int(env.blue.falls),
        "boundary_losses": int(boundary_losses),
        "steps": int(env.step_count),
    }


def _setup_actuator_degraded(env: FightEnv, rng: np.random.Generator) -> None:
    env.red.left_leg_damage = float(rng.uniform(0.25, 0.55))
    env.red.right_arm_damage = float(rng.uniform(0.18, 0.45))
    env.red.stamina = min(env.red.stamina, 0.68)
    env.red.balance = min(env.red.balance, 0.72)


def _setup_boundary_pressure(env: FightEnv, rng: np.random.Generator) -> None:
    env.red.x = env.config.arena_radius * 0.88
    env.red.y = float(rng.uniform(-0.35, 0.35))
    env.blue.x = env.config.arena_radius * 0.55
    env.blue.y = -env.red.y * 0.5
    env.red.balance = min(env.red.balance, 0.62)
    env._face_each_other(force=True)


def _setup_recovery_stress(env: FightEnv, rng: np.random.Generator) -> None:
    env.red.balance = float(rng.uniform(0.12, 0.28))
    env.red.stamina = float(rng.uniform(0.22, 0.42))
    env.red.omega = float(rng.uniform(-1.2, 1.2))
    env.red.core_damage = float(rng.uniform(0.08, 0.22))


def _summarize(df: pd.DataFrame, ckpt: dict[str, object]) -> Dict[str, object]:
    by_ablation = []
    for name, group in df.groupby("ablation"):
        by_ablation.append(
            {
                "ablation": name,
                "episodes": int(len(group)),
                "win_rate": float(group["win"].mean()),
                "fall_rate": float((group["red_falls"] > 0).mean()),
                "avg_red_falls": float(group["red_falls"].mean()),
                "avg_boundary_losses": float(group["boundary_losses"].mean()),
                "avg_health_margin": float((group["red_health"] - group["blue_health"]).mean()),
            }
        )
    clean = next((row for row in by_ablation if row["ablation"] == "clean"), None)
    worst = min(by_ablation, key=lambda row: row["win_rate"]) if by_ablation else None
    return {
        "policy_metrics": ckpt.get("metrics", {}),
        "ablations": by_ablation,
        "clean_win_rate": clean["win_rate"] if clean else None,
        "worst_ablation": worst,
        "robustness_gap": float((clean["win_rate"] - worst["win_rate"]) if clean and worst else 0.0),
    }


def _write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Robustness Ablation Report",
        "",
        "This report evaluates the PPO policy under nominal, domain-randomized, actuator-degraded, boundary-pressure, and recovery-stress conditions.",
        "",
        "| Ablation | Win Rate | Fall Rate | Boundary Losses | Health Margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["ablations"]:
        lines.append(
            f"| `{row['ablation']}` | {row['win_rate']:.3f} | {row['fall_rate']:.3f} | {row['avg_boundary_losses']:.2f} | {row['avg_health_margin']:.2f} |"
        )
    lines.extend(["", "```json", json.dumps(summary, indent=2), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_dashboard(path: Path, df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    grouped = df.groupby("ablation").agg(
        win_rate=("win", "mean"),
        fall_rate=("red_falls", lambda x: float((x > 0).mean())),
        boundary=("boundary_losses", "mean"),
    )
    fig = plt.figure(figsize=(11, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    grouped[["win_rate", "fall_rate"]].plot(kind="bar", ax=ax1)
    ax1.set_ylim(0, 1)
    ax1.set_title("Robustness win/fall rates")
    grouped["boundary"].plot(kind="bar", ax=ax2)
    ax2.set_title("Average boundary losses")
    ax2.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
