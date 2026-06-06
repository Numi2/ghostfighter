from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .attributes import ATTRIBUTE_NAMES, AttributePolicy, PolicyAttributes
from .config import ACTION_NAMES
from .domain import (
    apply_domain_randomization,
    apply_external_push,
    apply_observation_noise,
    sample_domain_randomization,
    summarize_domain_profiles,
    write_domain_randomization_card,
)
from .env import FightEnv, SimConfig


SELFPLAY_ROLES: Dict[str, Dict[str, float]] = {
    "striker": {
        "engagement_drive": 0.86,
        "guard_discipline": 0.38,
        "counter_timing": 0.48,
        "lateral_mobility": 0.46,
        "stamina_discipline": 0.46,
        "boundary_awareness": 0.58,
        "damage_targeting": 0.78,
        "risk_tolerance": 0.72,
        "close_range_pressure": 0.74,
    },
    "defender": {
        "engagement_drive": 0.42,
        "guard_discipline": 0.88,
        "counter_timing": 0.72,
        "lateral_mobility": 0.50,
        "stamina_discipline": 0.82,
        "boundary_awareness": 0.74,
        "damage_targeting": 0.50,
        "risk_tolerance": 0.32,
        "close_range_pressure": 0.38,
    },
    "stabilizer": {
        "engagement_drive": 0.50,
        "guard_discipline": 0.66,
        "counter_timing": 0.52,
        "lateral_mobility": 0.44,
        "stamina_discipline": 0.84,
        "boundary_awareness": 0.86,
        "damage_targeting": 0.44,
        "risk_tolerance": 0.24,
        "close_range_pressure": 0.36,
    },
    "evasive_mover": {
        "engagement_drive": 0.34,
        "guard_discipline": 0.62,
        "counter_timing": 0.66,
        "lateral_mobility": 0.92,
        "stamina_discipline": 0.78,
        "boundary_awareness": 0.92,
        "damage_targeting": 0.40,
        "risk_tolerance": 0.28,
        "close_range_pressure": 0.22,
    },
    "recovery_specialist": {
        "engagement_drive": 0.48,
        "guard_discipline": 0.76,
        "counter_timing": 0.58,
        "lateral_mobility": 0.54,
        "stamina_discipline": 0.92,
        "boundary_awareness": 0.82,
        "damage_targeting": 0.46,
        "risk_tolerance": 0.20,
        "close_range_pressure": 0.30,
    },
}


def run_population_self_play(
    out_dir: str | Path,
    generations: int = 3,
    matches_per_pair: int = 2,
    seed: int = 1601,
    max_steps: int = 90,
    variants_per_role: int = 2,
    domain_randomization: bool = True,
    domain_intensity: float = 0.55,
    verbose: bool = False,
) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    population = _make_population(variants_per_role, seed)
    elo = {p.policy_id: 1000.0 for p in population}
    rows = []
    action_hist = {p.policy_id: np.zeros(len(ACTION_NAMES), dtype=np.int64) for p in population}
    failure_counts = {p.policy_id: {"falls": 0, "boundary_losses": 0, "low_stamina_finishes": 0, "ko_losses": 0} for p in population}
    domain_profiles = []
    match_idx = 0

    for generation in range(1, generations + 1):
        for i, red_attrs in enumerate(population):
            for j, blue_attrs in enumerate(population):
                if i == j:
                    continue
                for rep in range(matches_per_pair):
                    match_seed = int(rng.integers(1, 10_000_000))
                    if verbose:
                        print(f"self-play gen={generation} {red_attrs.policy_id} vs {blue_attrs.policy_id}", flush=True)
                    result = _run_selfplay_match(
                        red_attrs,
                        blue_attrs,
                        match_seed,
                        max_steps,
                        domain_randomization,
                        domain_intensity,
                        action_hist,
                        failure_counts,
                        domain_profiles,
                    )
                    score_red = 0.5 if result["winner"] == -1 else (1.0 if result["winner"] == 0 else 0.0)
                    old_red, old_blue = elo[red_attrs.policy_id], elo[blue_attrs.policy_id]
                    elo[red_attrs.policy_id], elo[blue_attrs.policy_id] = _update_elo(old_red, old_blue, score_red)
                    rows.append(
                        {
                            "generation": generation,
                            "match": match_idx,
                            "red_policy": red_attrs.policy_id,
                            "blue_policy": blue_attrs.policy_id,
                            "red_role": red_attrs.archetype,
                            "blue_role": blue_attrs.archetype,
                            "winner": result["winner"],
                            "red_score": result["red_score"],
                            "blue_score": result["blue_score"],
                            "red_falls": result["red_falls"],
                            "blue_falls": result["blue_falls"],
                            "steps": result["steps"],
                            "red_elo_before": old_red,
                            "blue_elo_before": old_blue,
                            "red_elo_after": elo[red_attrs.policy_id],
                            "blue_elo_after": elo[blue_attrs.policy_id],
                        }
                    )
                    match_idx += 1

    match_df = pd.DataFrame(rows)
    match_df.to_csv(out_dir / "selfplay_matches.csv", index=False)
    population_rows = _population_rows(population, elo, action_hist, failure_counts)
    pd.DataFrame(population_rows).to_csv(out_dir / "population.csv", index=False)
    summary = _summarize_selfplay(match_df, population_rows, action_hist, domain_profiles)
    (out_dir / "selfplay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_selfplay_card(out_dir / "SELF_PLAY_CARD.md", summary)
    _make_selfplay_dashboard(out_dir / "selfplay_dashboard.png", match_df, population_rows)
    if domain_randomization:
        write_domain_randomization_card(out_dir / "DOMAIN_RANDOMIZATION_CARD.md", summary["domain_randomization"])
    return {"summary": summary, "rows": rows, "population": population_rows}


def _make_population(variants_per_role: int, seed: int) -> list[PolicyAttributes]:
    rng = np.random.default_rng(seed)
    population = []
    for role, base in SELFPLAY_ROLES.items():
        for idx in range(variants_per_role):
            attrs = {}
            for name in ATTRIBUTE_NAMES:
                attrs[name] = float(np.clip(base[name] + rng.normal(0.0, 0.055), 0.02, 1.0))
            population.append(
                PolicyAttributes(
                    archetype=role,
                    policy_id=f"{role}_{idx:03d}",
                    seed=int(rng.integers(1, 10_000_000)),
                    **attrs,
                )
            )
    return population


def _run_selfplay_match(
    red_attrs: PolicyAttributes,
    blue_attrs: PolicyAttributes,
    seed: int,
    max_steps: int,
    domain_randomization: bool,
    domain_intensity: float,
    action_hist: dict[str, np.ndarray],
    failure_counts: dict[str, dict[str, int]],
    domain_profiles: list,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    env = FightEnv(config=SimConfig(max_steps=max_steps, seed=seed), seed=seed)
    obs_red, obs_blue = env.reset(randomize=True)
    profile = None
    if domain_randomization:
        profile = sample_domain_randomization(rng, domain_intensity)
        domain_profiles.append(profile)
        apply_domain_randomization(env, profile)
        obs_red = apply_observation_noise(env.observe(0), rng, profile)
        obs_blue = apply_observation_noise(env.observe(1), rng, profile)
    red_policy = AttributePolicy(red_attrs, lookahead=False)
    blue_policy = AttributePolicy(blue_attrs, lookahead=False)
    done = False
    boundary_red = 0
    boundary_blue = 0
    while not done:
        action_red = red_policy.select_action(obs_red, env, 0)
        action_blue = blue_policy.select_action(obs_blue, env, 1)
        action_hist[red_attrs.policy_id][action_red] += 1
        action_hist[blue_attrs.policy_id][action_blue] += 1
        obs_red, obs_blue, _rr, _rb, done, info = env.step(action_red, action_blue)
        for event in info["events"]:
            if event["kind"] == "boundary":
                if event["target"] == 0:
                    boundary_red += 1
                else:
                    boundary_blue += 1
        if profile is not None and not done:
            apply_external_push(env, rng, profile)
            obs_red = apply_observation_noise(env.observe(0), rng, profile)
            obs_blue = apply_observation_noise(env.observe(1), rng, profile)

    if env.red.falls:
        failure_counts[red_attrs.policy_id]["falls"] += int(env.red.falls)
    if env.blue.falls:
        failure_counts[blue_attrs.policy_id]["falls"] += int(env.blue.falls)
    failure_counts[red_attrs.policy_id]["boundary_losses"] += boundary_red
    failure_counts[blue_attrs.policy_id]["boundary_losses"] += boundary_blue
    if env.red.stamina < 0.18:
        failure_counts[red_attrs.policy_id]["low_stamina_finishes"] += 1
    if env.blue.stamina < 0.18:
        failure_counts[blue_attrs.policy_id]["low_stamina_finishes"] += 1
    if env.red.health <= env.config.ko_health:
        failure_counts[red_attrs.policy_id]["ko_losses"] += 1
    if env.blue.health <= env.config.ko_health:
        failure_counts[blue_attrs.policy_id]["ko_losses"] += 1
    return {
        "winner": env.winner(),
        "red_score": float(env.red.score),
        "blue_score": float(env.blue.score),
        "red_falls": int(env.red.falls),
        "blue_falls": int(env.blue.falls),
        "steps": int(env.step_count),
    }


def _update_elo(red_elo: float, blue_elo: float, red_score: float, k: float = 28.0) -> tuple[float, float]:
    expected_red = 1.0 / (1.0 + 10 ** ((blue_elo - red_elo) / 400.0))
    expected_blue = 1.0 - expected_red
    return red_elo + k * (red_score - expected_red), blue_elo + k * ((1.0 - red_score) - expected_blue)


def _population_rows(population, elo, action_hist, failure_counts) -> list[dict[str, object]]:
    rows = []
    for attrs in population:
        hist = action_hist[attrs.policy_id].astype(np.float64)
        probs = hist / max(1.0, hist.sum())
        entropy = float(-(probs[probs > 0] * np.log2(probs[probs > 0])).sum())
        row = asdict(attrs)
        row.update(
            {
                "role": attrs.archetype,
                "elo": float(elo[attrs.policy_id]),
                "action_entropy": entropy,
                "top_action": ACTION_NAMES[int(np.argmax(hist))] if hist.sum() else "",
                **failure_counts[attrs.policy_id],
            }
        )
        rows.append(row)
    return rows


def _summarize_selfplay(match_df: pd.DataFrame, population_rows: list[dict[str, object]], action_hist: dict[str, np.ndarray], domain_profiles: list) -> Dict[str, object]:
    if match_df.empty:
        return {"matches": 0}
    pop_df = pd.DataFrame(population_rows)
    total_matches = int(len(match_df))
    win_rates = []
    for role in pop_df["role"].unique():
        ids = set(pop_df.loc[pop_df["role"] == role, "policy_id"])
        red = match_df[match_df["red_policy"].isin(ids)]
        blue = match_df[match_df["blue_policy"].isin(ids)]
        wins = int((red["winner"] == 0).sum() + (blue["winner"] == 1).sum())
        draws = int((red["winner"] == -1).sum() + (blue["winner"] == -1).sum())
        played = int(len(red) + len(blue))
        win_rates.append({"role": role, "played": played, "win_rate": float((wins + 0.5 * draws) / max(1, played))})
    diversity = _policy_diversity(action_hist)
    best = float(pop_df["elo"].max())
    median = float(pop_df["elo"].median())
    exploitability = best - median
    failure_modes = {
        "falls": int(pop_df["falls"].sum()),
        "boundary_losses": int(pop_df["boundary_losses"].sum()),
        "low_stamina_finishes": int(pop_df["low_stamina_finishes"].sum()),
        "ko_losses": int(pop_df["ko_losses"].sum()),
    }
    return {
        "matches": total_matches,
        "generations": int(match_df["generation"].max()),
        "population_size": int(len(pop_df)),
        "roles": sorted(pop_df["role"].unique().tolist()),
        "elo": {
            "min": float(pop_df["elo"].min()),
            "median": median,
            "max": best,
            "spread": float(pop_df["elo"].max() - pop_df["elo"].min()),
        },
        "exploitability_elo_gap": exploitability,
        "policy_diversity_jsd": diversity,
        "role_win_rates": win_rates,
        "failure_modes": failure_modes,
        "domain_randomization": summarize_domain_profiles(domain_profiles),
    }


def _policy_diversity(action_hist: dict[str, np.ndarray]) -> float:
    probs = []
    for hist in action_hist.values():
        p = hist.astype(np.float64) + 1e-9
        probs.append(p / p.sum())
    if len(probs) < 2:
        return 0.0
    vals = []
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            vals.append(_js_divergence(probs[i], probs[j]))
    return float(np.mean(vals))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def _write_selfplay_card(path: Path, summary: Dict[str, object]) -> None:
    text = f"""# Self-Play Card

GhostFighter runs population-based self-play across five policy roles: striker, defender, stabilizer, evasive mover, and recovery specialist. These are not scripted combat styles; they are adversarial policy priors that fight each other, update Elo-style ratings, and expose exploitability, diversity, and failure modes.

The compact backend is intended for fast curriculum experiments. Serious claims should scale this same interface to a vectorized Isaac Lab backend and validate selected policies in a higher-fidelity MuJoCo backend.

```json
{json.dumps(summary, indent=2)}
```
"""
    path.write_text(text, encoding="utf-8")


def _make_selfplay_dashboard(path: Path, match_df: pd.DataFrame, population_rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    pop = pd.DataFrame(population_rows).sort_values("elo", ascending=False)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)
    ax1.bar(pop["policy_id"], pop["elo"])
    ax1.set_title("Population Elo")
    ax1.tick_params(axis="x", rotation=55)
    ax2.bar(pop["role"], pop["action_entropy"])
    ax2.set_title("Policy action diversity")
    ax2.tick_params(axis="x", rotation=25)
    fail_cols = ["falls", "boundary_losses", "low_stamina_finishes", "ko_losses"]
    failures = pop.groupby("role")[fail_cols].sum()
    failures.plot(kind="bar", stacked=True, ax=ax3)
    ax3.set_title("Failure modes by role")
    ax3.tick_params(axis="x", rotation=20)
    fig.suptitle("GhostFighter population self-play", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=160)
    plt.close(fig)
