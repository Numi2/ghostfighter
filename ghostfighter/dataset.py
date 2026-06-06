from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ACTION_NAMES, STYLE_NAMES, STYLE_TO_ID, SimConfig
from .env import FightEnv
from .policies import ScriptedPilot


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
) -> Dict[str, object]:
    """Generate pilot traces from scripted fight styles.

    Both red and blue pilots are logged, so every match contributes two policy views.
    Each trace item is: observation, style id, action, immediate reward, and episode id.
    """
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
    )
    summary: Dict[str, object] = {
        "path": str(out_path),
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


def load_npz_arrays(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["obs"].astype(np.float32), data["styles"].astype(np.int64), data["actions"].astype(np.int64)
