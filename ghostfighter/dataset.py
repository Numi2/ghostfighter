from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import ACTION_NAMES, STYLE_NAMES, STYLE_TO_ID, SimConfig
from .env import FightEnv
from .policies import ScriptedPilot
from .attributes import (
    ATTRIBUTE_NAMES,
    AttributePolicy,
    load_policy_spec,
    policy_attributes_to_rows,
    sample_attribute_policies,
    write_default_policy_spec,
)


class FightTraceDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path, allow_pickle=True)
        self.obs = torch.as_tensor(data["obs"], dtype=torch.float32)
        self.actions = torch.as_tensor(data["actions"], dtype=torch.long)
        self.styles = torch.as_tensor(data["styles"], dtype=torch.long)
        self.rewards = torch.as_tensor(data["rewards"], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, idx: int):
        return self.obs[idx], self.styles[idx], self.actions[idx]


def generate_trace_dataset(
    out_path: str | Path,
    episodes_per_style: int = 80,
    seed: int = 101,
    max_steps: int | None = None,
    source: str = "attributes",
    policy_spec: str | Path | None = None,
    variants_per_archetype: int = 8,
) -> Dict[str, object]:
    """Generate Generation Zero traces from attribute policies or scripted baselines.

    Both fighters are logged, so every match contributes two policy views.
    Each trace item is: observation, policy-condition id, action, immediate reward, and episode id.
    """
    if source == "scripted":
        return generate_scripted_trace_dataset(out_path, episodes_per_style, seed, max_steps=max_steps)
    if source == "attributes":
        return generate_attribute_trace_dataset(
            out_path,
            episodes_per_archetype=episodes_per_style,
            seed=seed,
            max_steps=max_steps,
            policy_spec=policy_spec,
            variants_per_archetype=variants_per_archetype,
        )
    raise ValueError("source must be 'attributes' or 'scripted'")


def generate_scripted_trace_dataset(
    out_path: str | Path,
    episodes_per_style: int = 80,
    seed: int = 101,
    max_steps: int | None = None,
) -> Dict[str, object]:
    """Generate legacy scripted baseline traces."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    config = SimConfig(max_steps=max_steps or SimConfig().max_steps, seed=seed)
    obs_buf: list[np.ndarray] = []
    action_buf: list[int] = []
    style_buf: list[int] = []
    reward_buf: list[float] = []
    episode_buf: list[int] = []
    fighter_buf: list[int] = []

    episode = 0
    style_counts = {name: 0 for name in STYLE_NAMES}
    action_counts = {name: 0 for name in ACTION_NAMES}
    total_steps = 0
    for style_name in STYLE_NAMES:
        for _ in range(episodes_per_style):
            env = FightEnv(config=config, seed=int(rng.integers(1, 10_000_000)))
            opp_style = STYLE_NAMES[int(rng.integers(0, len(STYLE_NAMES)))]
            red_policy = ScriptedPilot(style_name, seed=int(rng.integers(1, 10_000_000)))
            blue_policy = ScriptedPilot(opp_style, seed=int(rng.integers(1, 10_000_000)))
            obs_red, obs_blue = env.reset(randomize=True)
            done = False
            while not done:
                action_red = red_policy.select_action(obs_red, env, 0)
                action_blue = blue_policy.select_action(obs_blue, env, 1)
                # Store pre-action observations, which is what a policy sees.
                obs_buf.append(obs_red.copy())
                action_buf.append(action_red)
                style_buf.append(STYLE_TO_ID[style_name])
                episode_buf.append(episode)
                fighter_buf.append(0)
                action_counts[ACTION_NAMES[action_red]] += 1

                obs_buf.append(obs_blue.copy())
                action_buf.append(action_blue)
                style_buf.append(STYLE_TO_ID[opp_style])
                episode_buf.append(episode)
                fighter_buf.append(1)
                action_counts[ACTION_NAMES[action_blue]] += 1

                next_red, next_blue, r_red, r_blue, done, _info = env.step(action_red, action_blue)
                reward_buf.extend([r_red, r_blue])
                obs_red, obs_blue = next_red, next_blue
                total_steps += 1
            style_counts[style_name] += 1
            episode += 1

    obs = np.asarray(obs_buf, dtype=np.float32)
    actions = np.asarray(action_buf, dtype=np.int64)
    styles = np.asarray(style_buf, dtype=np.int64)
    rewards = np.asarray(reward_buf, dtype=np.float32)
    episodes = np.asarray(episode_buf, dtype=np.int64)
    fighters = np.asarray(fighter_buf, dtype=np.int64)
    np.savez_compressed(
        out_path,
        obs=obs,
        actions=actions,
        styles=styles,
        rewards=rewards,
        episodes=episodes,
        fighters=fighters,
        action_names=np.asarray(ACTION_NAMES),
        style_names=np.asarray(STYLE_NAMES),
        obs_dim=np.asarray([obs.shape[1]], dtype=np.int64),
        source_ids=np.zeros_like(actions, dtype=np.int64),
    )
    summary: Dict[str, object] = {
        "path": str(out_path),
        "source": "scripted",
        "episodes": episode,
        "episodes_per_style": episodes_per_style,
        "samples": int(obs.shape[0]),
        "obs_dim": int(obs.shape[1]),
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "style_counts": style_counts,
        "action_counts": action_counts,
        "steps": int(total_steps),
        "seed": seed,
    }
    with open(out_path.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def generate_attribute_trace_dataset(
    out_path: str | Path,
    episodes_per_archetype: int = 80,
    seed: int = 101,
    max_steps: int | None = None,
    policy_spec: str | Path | None = None,
    variants_per_archetype: int = 8,
) -> Dict[str, object]:
    """Generate Generation Zero traces from user-specified policy attributes."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gen0_dir = out_path.parent.parent / "gen0" if out_path.parent.name == "data" else out_path.parent / "gen0"
    gen0_dir.mkdir(parents=True, exist_ok=True)
    spec = load_policy_spec(policy_spec)
    if policy_spec is None:
        write_default_policy_spec(gen0_dir / "policy_specs.resolved.json")
    else:
        (gen0_dir / "policy_specs.resolved.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    policies = sample_attribute_policies(spec, variants_per_archetype=variants_per_archetype, seed=seed)
    policy_by_archetype = {name: [p for p in policies if p.archetype == name] for name in STYLE_NAMES}
    policy_rows = policy_attributes_to_rows(policies)
    pd.DataFrame(policy_rows).to_csv(gen0_dir / "policy_variants.csv", index=False)
    (gen0_dir / "policy_variants.json").write_text(json.dumps(policy_rows, indent=2), encoding="utf-8")

    rng = np.random.default_rng(seed)
    config = SimConfig(max_steps=max_steps or SimConfig().max_steps, seed=seed)
    obs_buf: list[np.ndarray] = []
    action_buf: list[int] = []
    style_buf: list[int] = []
    reward_buf: list[float] = []
    episode_buf: list[int] = []
    fighter_buf: list[int] = []
    policy_id_buf: list[int] = []
    source_id_buf: list[int] = []
    attr_buf: list[np.ndarray] = []
    action_counts = {name: 0 for name in ACTION_NAMES}
    archetype_counts = {name: 0 for name in STYLE_NAMES}
    variant_episode_counts = {p.policy_id: 0 for p in policies}
    policy_index = {p.policy_id: i for i, p in enumerate(policies)}
    episode = 0
    total_steps = 0

    for archetype in STYLE_NAMES:
        variants = policy_by_archetype[archetype]
        for ep in range(episodes_per_archetype):
            red_attrs = variants[ep % len(variants)]
            opp_archetype = STYLE_NAMES[int(rng.integers(0, len(STYLE_NAMES)))]
            opp_attrs = policy_by_archetype[opp_archetype][int(rng.integers(0, len(policy_by_archetype[opp_archetype])))]
            red_policy = AttributePolicy(red_attrs, lookahead=False)
            blue_policy = AttributePolicy(opp_attrs, lookahead=False)
            env = FightEnv(config=config, seed=int(rng.integers(1, 10_000_000)))
            obs_red, obs_blue = env.reset(randomize=True)
            done = False
            while not done:
                action_red = red_policy.select_action(obs_red, env, 0)
                action_blue = blue_policy.select_action(obs_blue, env, 1)
                obs_buf.append(obs_red.copy())
                action_buf.append(action_red)
                style_buf.append(STYLE_TO_ID[red_attrs.archetype])
                episode_buf.append(episode)
                fighter_buf.append(0)
                policy_id_buf.append(policy_index[red_attrs.policy_id])
                source_id_buf.append(1)
                attr_buf.append(red_attrs.vector())
                action_counts[ACTION_NAMES[action_red]] += 1

                obs_buf.append(obs_blue.copy())
                action_buf.append(action_blue)
                style_buf.append(STYLE_TO_ID[opp_attrs.archetype])
                episode_buf.append(episode)
                fighter_buf.append(1)
                policy_id_buf.append(policy_index[opp_attrs.policy_id])
                source_id_buf.append(1)
                attr_buf.append(opp_attrs.vector())
                action_counts[ACTION_NAMES[action_blue]] += 1

                next_red, next_blue, r_red, r_blue, done, _info = env.step(action_red, action_blue)
                reward_buf.extend([r_red, r_blue])
                obs_red, obs_blue = next_red, next_blue
                total_steps += 1
            archetype_counts[archetype] += 1
            variant_episode_counts[red_attrs.policy_id] += 1
            episode += 1

    obs = np.asarray(obs_buf, dtype=np.float32)
    actions = np.asarray(action_buf, dtype=np.int64)
    styles = np.asarray(style_buf, dtype=np.int64)
    rewards = np.asarray(reward_buf, dtype=np.float32)
    episodes = np.asarray(episode_buf, dtype=np.int64)
    fighters = np.asarray(fighter_buf, dtype=np.int64)
    policy_ids = np.asarray(policy_id_buf, dtype=np.int64)
    source_ids = np.asarray(source_id_buf, dtype=np.int64)
    attribute_vectors = np.asarray(attr_buf, dtype=np.float32)
    np.savez_compressed(
        out_path,
        obs=obs,
        actions=actions,
        styles=styles,
        rewards=rewards,
        episodes=episodes,
        fighters=fighters,
        policy_ids=policy_ids,
        source_ids=source_ids,
        attribute_vectors=attribute_vectors,
        attribute_names=np.asarray(ATTRIBUTE_NAMES),
        policy_variant_ids=np.asarray([p.policy_id for p in policies]),
        action_names=np.asarray(ACTION_NAMES),
        style_names=np.asarray(STYLE_NAMES),
        obs_dim=np.asarray([obs.shape[1]], dtype=np.int64),
    )
    probs = np.asarray(list(action_counts.values()), dtype=np.float64)
    probs = probs / max(1.0, probs.sum())
    entropy = float(-(probs[probs > 0] * np.log2(probs[probs > 0])).sum())
    summary: Dict[str, object] = {
        "path": str(out_path),
        "source": "attributes",
        "episodes": episode,
        "episodes_per_archetype": episodes_per_archetype,
        "variants_per_archetype": variants_per_archetype,
        "policy_variants": len(policies),
        "samples": int(obs.shape[0]),
        "obs_dim": int(obs.shape[1]),
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "action_entropy": entropy,
        "archetype_counts": archetype_counts,
        "variant_episode_counts": variant_episode_counts,
        "action_counts": action_counts,
        "attribute_ranges": _observed_attribute_ranges(policies),
        "steps": int(total_steps),
        "seed": seed,
    }
    with open(out_path.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    (gen0_dir / "attribute_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_generation_zero_card(gen0_dir / "GENERATION_ZERO_CARD.md", summary)
    _make_attribute_dashboard(gen0_dir / "attribute_dashboard.png", policy_rows, action_counts)
    return summary


def load_npz_arrays(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["obs"].astype(np.float32), data["styles"].astype(np.int64), data["actions"].astype(np.int64)


def _observed_attribute_ranges(policies) -> Dict[str, list[float]]:
    arr = np.asarray([p.vector() for p in policies], dtype=np.float32)
    return {name: [float(arr[:, i].min()), float(arr[:, i].max())] for i, name in enumerate(ATTRIBUTE_NAMES)}


def _write_generation_zero_card(path: Path, summary: Dict[str, object]) -> None:
    text = f"""# Generation Zero Card

Generation Zero data was produced from user-configurable policy attributes rather than fixed scripted pilots. Each policy archetype defines ranges for behavior attributes such as engagement drive, guard discipline, counter timing, lateral mobility, stamina discipline, boundary awareness, damage targeting, risk tolerance, and close-range pressure.

GhostFighter samples concrete policy variants from those ranges, rolls them out in simulation, and trains the neural ghost policy from the resulting observation/action traces.

```json
{json.dumps(summary, indent=2)}
```
"""
    path.write_text(text, encoding="utf-8")


def _make_attribute_dashboard(path: Path, policy_rows: list[dict[str, object]], action_counts: Dict[str, int]) -> None:
    import matplotlib.pyplot as plt

    df = pd.DataFrame(policy_rows)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)
    for archetype, group in df.groupby("archetype"):
        ax1.scatter(group["engagement_drive"], group["guard_discipline"], label=archetype)
    ax1.set_xlabel("engagement drive")
    ax1.set_ylabel("guard discipline")
    ax1.set_title("Generated policy variants")
    ax1.legend()
    for archetype, group in df.groupby("archetype"):
        ax2.scatter(group["risk_tolerance"], group["boundary_awareness"], label=archetype)
    ax2.set_xlabel("risk tolerance")
    ax2.set_ylabel("boundary awareness")
    ax2.set_title("Safety posture")
    actions = list(action_counts.keys())
    counts = [action_counts[a] for a in actions]
    ax3.bar(actions, counts)
    ax3.set_title("Generation Zero action distribution")
    ax3.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
