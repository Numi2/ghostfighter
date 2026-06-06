from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import ACTION_NAMES, STYLE_NAMES, ACTION_TO_ID, ATTACK_ACTIONS, MOVEMENT_ACTIONS
from .env import FightEnv, SimConfig
from .models import load_policy_checkpoint
from .policies import EnsembleOpponent, NeuralGhostPolicy, ScriptedPilot
from .safety import CombatSafetyFirewall


@dataclass
class MatchResult:
    episode: int
    style: str
    mode: str
    winner: int
    red_score: float
    blue_score: float
    red_health: float
    blue_health: float
    red_falls: int
    blue_falls: int
    steps: int
    unsafe_rejections: int
    unsafe_rate: float
    avg_risk: float
    red_action_entropy: float
    red_attack_rate: float
    red_guard_rate: float


def run_match(
    red_policy,
    blue_policy,
    seed: int,
    style_name: str,
    mode: str,
    firewall: CombatSafetyFirewall | None = None,
    max_steps: int | None = None,
    collect_trace: bool = False,
    stress_level: float = 0.0,
) -> tuple[MatchResult, list[dict]]:
    env = FightEnv(config=SimConfig(max_steps=max_steps or SimConfig().max_steps, seed=seed), seed=seed)
    obs_red, obs_blue = env.reset(randomize=True)
    stress_rng = np.random.default_rng(seed + 99173)
    if stress_level > 0:
        _apply_initial_stress(env, stress_rng, stress_level)
        obs_red, obs_blue = env.observe(0), env.observe(1)
    if hasattr(blue_policy, "reset"):
        blue_policy.reset(seed)
    if firewall:
        firewall.reset()
    action_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    risk_values: list[float] = []
    trace: list[dict] = []
    done = False
    while not done:
        proposed_red = int(red_policy.select_action(obs_red, env, 0))
        if firewall:
            decision = firewall.filter(env, 0, proposed_red)
            action_red = decision.action
            risk_values.append(decision.risk)
        else:
            decision = None
            action_red = proposed_red
            risk_values.append(0.0)
        action_blue = int(blue_policy.select_action(obs_blue, env, 1))
        action_counts[action_red] += 1
        next_red, next_blue, _r_red, _r_blue, done, info = env.step(action_red, action_blue)
        if stress_level > 0 and not done:
            _apply_transient_stress(env, stress_rng, stress_level)
            next_red, next_blue = env.observe(0), env.observe(1)
        if collect_trace:
            trace.append(
                {
                    "step": env.step_count,
                    "obs_red": obs_red.copy(),
                    "proposed_action": proposed_red,
                    "action_red": action_red,
                    "action_blue": action_blue,
                    "decision": decision.as_dict() if decision else None,
                    "events": info["events"],
                    "red": info["red"],
                    "blue": info["blue"],
                    "env": env.clone(),
                }
            )
        obs_red, obs_blue = next_red, next_blue
    probs = action_counts / max(1, action_counts.sum())
    entropy = float(-(probs[probs > 0] * np.log2(probs[probs > 0])).sum())
    attack_ids = [ACTION_NAMES.index(x) for x in ["jab", "cross", "hook", "low_kick", "push"]]
    attack_rate = float(action_counts[attack_ids].sum() / max(1, action_counts.sum()))
    guard_rate = float(action_counts[ACTION_NAMES.index("guard")] / max(1, action_counts.sum()))
    unsafe_rejections = int(firewall.rejections if firewall else 0)
    unsafe_rate = float(firewall.rejection_rate if firewall else 0.0)
    result = MatchResult(
        episode=seed,
        style=style_name,
        mode=mode,
        winner=env.winner(),
        red_score=float(env.red.score),
        blue_score=float(env.blue.score),
        red_health=float(env.red.health),
        blue_health=float(env.blue.health),
        red_falls=int(env.red.falls),
        blue_falls=int(env.blue.falls),
        steps=int(env.step_count),
        unsafe_rejections=unsafe_rejections,
        unsafe_rate=unsafe_rate,
        avg_risk=float(np.mean(risk_values)) if risk_values else 0.0,
        red_action_entropy=entropy,
        red_attack_rate=attack_rate,
        red_guard_rate=guard_rate,
    )
    return result, trace


def evaluate_policy(
    model_path: str | Path,
    out_dir: str | Path,
    episodes: int = 160,
    seed: int = 222,
    max_steps: int | None = None,
    verbose: bool = False,
    include_stress: bool = False,
) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_policy_checkpoint(str(model_path))
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    matches_per_style = max(1, episodes // len(STYLE_NAMES))
    modes = ["raw", "firewall"]
    if include_stress:
        modes.extend(["raw_stress", "firewall_stress"])
    for style_id, style_name in enumerate(STYLE_NAMES):
        for mode in modes:
            if verbose:
                print(f"evaluating {mode}/{style_name} ({matches_per_style} matches)", flush=True)
            for ep in range(matches_per_style):
                match_seed = int(rng.integers(1, 10_000_000))
                red_policy = NeuralGhostPolicy(model, style_id=style_id, deterministic=True, name=f"ghost_{style_name}")
                blue_policy = EnsembleOpponent(seed=match_seed + 17)
                firewall = CombatSafetyFirewall(threshold=0.62) if "firewall" in mode else None
                stress_level = 1.0 if mode.endswith("stress") else 0.0
                result, _ = run_match(
                    red_policy,
                    blue_policy,
                    seed=match_seed,
                    style_name=style_name,
                    mode=mode,
                    firewall=firewall,
                    max_steps=max_steps,
                    collect_trace=False,
                    stress_level=stress_level,
                )
                rows.append(asdict(result))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "match_results.csv", index=False)
    summary = summarize_results(df)
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return {"summary": summary, "rows": rows, "checkpoint_metrics": ckpt.get("metrics", {})}


def evaluate_scripted_baseline(out_dir: str | Path, episodes: int = 80, seed: int = 333, verbose: bool = False) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    matches_per_style = max(1, episodes // len(STYLE_NAMES))
    for style_name in STYLE_NAMES:
        if verbose:
            print(f"scripted baseline/{style_name} ({matches_per_style} matches)", flush=True)
        for ep in range(matches_per_style):
            match_seed = int(rng.integers(1, 10_000_000))
            red_policy = ScriptedPilot(style_name, seed=match_seed + 5)
            blue_policy = EnsembleOpponent(seed=match_seed + 10)
            result, _ = run_match(
                red_policy,
                blue_policy,
                seed=match_seed,
                style_name=style_name,
                mode="scripted_baseline",
                firewall=None,
            )
            rows.append(asdict(result))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "scripted_baseline.csv", index=False)
    summary = summarize_results(df)
    with open(out_dir / "scripted_baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _apply_initial_stress(env: FightEnv, rng: np.random.Generator, level: float) -> None:
    """Create hardware-stress conditions: partial damage, low battery, bad balance, and boundary pressure."""
    for idx, f in enumerate((env.red, env.blue)):
        asym = 1.0 if idx == 0 else 0.85
        f.balance = float(np.clip(rng.uniform(0.24, 0.68) - 0.08 * level * asym, 0.08, 0.88))
        f.stamina = float(np.clip(rng.uniform(0.26, 0.82) - 0.05 * level, 0.08, 1.0))
        f.left_leg_damage = float(np.clip(rng.uniform(0.0, 0.42) * level * asym, 0.0, 0.75))
        f.right_leg_damage = float(np.clip(rng.uniform(0.0, 0.42) * level * asym, 0.0, 0.75))
        f.left_arm_damage = float(np.clip(rng.uniform(0.0, 0.32) * level, 0.0, 0.65))
        f.right_arm_damage = float(np.clip(rng.uniform(0.0, 0.32) * level, 0.0, 0.65))
        f.core_damage = float(np.clip(rng.uniform(0.0, 0.24) * level, 0.0, 0.55))
    if rng.random() < 0.60:
        # Start red under ring pressure, a common real-match stressor.
        angle = float(rng.uniform(-np.pi, np.pi))
        radius = float(rng.uniform(0.76, 0.93) * env.config.arena_radius)
        env.red.x = np.cos(angle) * radius
        env.red.y = np.sin(angle) * radius
        env.blue.x = np.cos(angle + np.pi) * rng.uniform(0.15, 0.35) * env.config.arena_radius
        env.blue.y = np.sin(angle + np.pi) * rng.uniform(0.15, 0.35) * env.config.arena_radius
    env._face_each_other(force=True)


def _apply_transient_stress(env: FightEnv, rng: np.random.Generator, level: float) -> None:
    """Inject contact/actuator noise that punishes reckless actions more than safe ones."""
    for f in (env.red, env.blue):
        if f.fallen:
            continue
        action = int(f.last_action)
        risky_action = action in ATTACK_ACTIONS or action in MOVEMENT_ACTIONS
        if risky_action:
            f.balance = float(np.clip(f.balance - rng.uniform(0.008, 0.040) * level, env.config.min_balance, 1.0))
            if rng.random() < 0.10 * level:
                impulse = rng.normal(0.0, 0.32, size=2)
                f.vx += float(impulse[0])
                f.vy += float(impulse[1])
                f.omega += float(rng.normal(0.0, 0.65))
                f.balance = float(np.clip(f.balance - rng.uniform(0.025, 0.090) * level, env.config.min_balance, 1.0))
        elif action == ACTION_TO_ID["recover"]:
            f.balance = float(np.clip(f.balance + 0.030 * level, env.config.min_balance, 1.0))
        elif action == ACTION_TO_ID["guard"]:
            f.balance = float(np.clip(f.balance + 0.012 * level, env.config.min_balance, 1.0))


def summarize_results(df: pd.DataFrame) -> Dict[str, object]:
    summaries = []
    group_cols = ["mode", "style"] if "style" in df.columns else ["mode"]
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = dict(zip(group_cols, keys))
        item.update(
            {
                "matches": int(len(group)),
                "win_rate": float((group["winner"] == 0).mean()),
                "draw_rate": float((group["winner"] == -1).mean()),
                "red_fall_rate": float((group["red_falls"] > 0).mean()),
                "avg_red_falls": float(group["red_falls"].mean()),
                "avg_blue_falls": float(group["blue_falls"].mean()),
                "avg_health_margin": float((group["red_health"] - group["blue_health"]).mean()),
                "avg_score_margin": float((group["red_score"] - group["blue_score"]).mean()),
                "avg_unsafe_rate": float(group["unsafe_rate"].mean()),
                "avg_risk": float(group["avg_risk"].mean()),
                "avg_attack_rate": float(group["red_attack_rate"].mean()),
                "avg_guard_rate": float(group["red_guard_rate"].mean()),
                "avg_entropy": float(group["red_action_entropy"].mean()),
            }
        )
        summaries.append(item)
    # Global raw vs firewall comparison.
    global_by_mode = []
    for mode, group in df.groupby("mode"):
        global_by_mode.append(
            {
                "mode": mode,
                "matches": int(len(group)),
                "win_rate": float((group["winner"] == 0).mean()),
                "red_fall_rate": float((group["red_falls"] > 0).mean()),
                "avg_red_falls": float(group["red_falls"].mean()),
                "avg_health_margin": float((group["red_health"] - group["blue_health"]).mean()),
                "avg_score_margin": float((group["red_score"] - group["blue_score"]).mean()),
                "avg_unsafe_rate": float(group["unsafe_rate"].mean()),
                "avg_risk": float(group["avg_risk"].mean()),
            }
        )
    return {"by_mode_style": summaries, "by_mode": global_by_mode}
