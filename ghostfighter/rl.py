from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

from .attributes import AttributePolicy
from .config import ACTION_NAMES, STYLE_NAMES
from .domain import apply_domain_randomization, apply_external_push, apply_observation_noise, sample_domain_randomization
from .env import FightEnv, SimConfig
from .selfplay import SELFPLAY_ROLES, _make_population, _update_elo
from .train import set_seeds


class ActorCriticPolicy(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int = len(ACTION_NAMES), hidden: int = 128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, num_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        z = self.encoder(obs.float())
        return self.actor(z), self.critic(z).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        logits, value = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        dist = Categorical(logits=logits[0])
        action = torch.argmax(logits[0]) if deterministic else dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value[0].item())


class NeuralActorPolicy:
    def __init__(self, model: ActorCriticPolicy, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def select_action(self, obs: np.ndarray, env: FightEnv, fighter_idx: int) -> int:
        action, _logp, _value = self.model.act(obs, deterministic=self.deterministic)
        return action


@dataclass
class PPOConfig:
    updates: int = 4
    matches_per_update: int = 8
    max_steps: int = 80
    epochs: int = 3
    batch_size: int = 512
    gamma: float = 0.97
    gae_lambda: float = 0.92
    clip: float = 0.20
    entropy_coef: float = 0.020
    value_coef: float = 0.50
    lr: float = 3.0e-4
    hidden: int = 128
    seed: int = 1801
    snapshot_interval: int = 1
    domain_randomization: bool = True
    domain_intensity: float = 0.45


def train_ppo_self_play(
    out_dir: str | Path,
    config: PPOConfig | None = None,
    verbose: bool = False,
) -> Dict[str, object]:
    config = config or PPOConfig()
    set_seeds(config.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    probe_env = FightEnv(config=SimConfig(max_steps=config.max_steps, seed=config.seed), seed=config.seed)
    obs_dim = probe_env.observation_dim
    model = ActorCriticPolicy(obs_dim=obs_dim, hidden=config.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    rng = np.random.default_rng(config.seed)
    role_population = _make_population(variants_per_role=1, seed=config.seed + 31)
    historical: list[dict[str, object]] = []
    elo = {"ppo_current": 1000.0}
    rows = []

    for update in range(1, config.updates + 1):
        rollout = _collect_ppo_rollout(model, historical, role_population, config, rng, verbose=verbose)
        _ppo_update(model, optimizer, rollout, config)
        score = _update_training_elo(model, historical, role_population, rng, config)
        elo["ppo_current"] = score["elo"]
        if update % config.snapshot_interval == 0:
            snap_name = f"ppo_update_{update:03d}"
            historical.append({"name": snap_name, "state": copy.deepcopy(model.state_dict()), "elo": float(score["elo"])})
            elo[snap_name] = float(score["elo"])
            _save_actor_checkpoint(ckpt_dir / f"{snap_name}.pt", model, config, {"update": update, "elo": score["elo"]})
        row = {
            "update": update,
            "episodes": int(rollout["episodes"]),
            "steps": int(len(rollout["actions"])),
            "mean_return": float(np.mean(rollout["episode_returns"])) if rollout["episode_returns"] else 0.0,
            "mean_length": float(np.mean(rollout["episode_lengths"])) if rollout["episode_lengths"] else 0.0,
            "policy_loss": float(rollout.get("policy_loss", 0.0)),
            "value_loss": float(rollout.get("value_loss", 0.0)),
            "entropy": float(rollout.get("entropy", 0.0)),
            "elo": float(score["elo"]),
            "eval_win_rate": float(score["win_rate"]),
            "eval_fall_rate": float(score["fall_rate"]),
        }
        rows.append(row)
        if verbose:
            print(f"ppo update {update}/{config.updates} return={row['mean_return']:.3f} elo={row['elo']:.1f}", flush=True)

    _save_actor_checkpoint(out_dir / "ppo_policy.pt", model, config, {"final_update": config.updates, "elo": elo["ppo_current"]})
    curve = pd.DataFrame(rows)
    curve.to_csv(out_dir / "ppo_training_curve.csv", index=False)
    leaderboard = run_policy_leaderboard(model, historical, role_population, out_dir, seed=config.seed + 99, max_steps=config.max_steps)
    summary = {
        "updates": config.updates,
        "steps": int(curve["steps"].sum()) if not curve.empty else 0,
        "matches": int(curve["episodes"].sum()) if not curve.empty else 0,
        "final_mean_return": float(curve["mean_return"].iloc[-1]) if not curve.empty else 0.0,
        "final_elo": float(elo["ppo_current"]),
        "historical_opponents": len(historical),
        "leaderboard": leaderboard["summary"],
        "config": config.__dict__,
    }
    (out_dir / "ppo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_rl_card(out_dir / "RL_TRAINING_CARD.md", summary)
    return {"summary": summary, "model_path": str(out_dir / "ppo_policy.pt"), "curve": rows}


def _collect_ppo_rollout(model, historical, role_population, config: PPOConfig, rng, verbose: bool = False) -> dict[str, object]:
    obs_buf, action_buf, logp_buf, value_buf, reward_buf, done_buf = [], [], [], [], [], []
    episode_returns, episode_lengths = [], []
    for episode in range(config.matches_per_update):
        seed = int(rng.integers(1, 10_000_000))
        env = FightEnv(config=SimConfig(max_steps=config.max_steps, seed=seed), seed=seed)
        obs_red, obs_blue = env.reset(randomize=True)
        profile = None
        if config.domain_randomization:
            profile = sample_domain_randomization(rng, config.domain_intensity)
            apply_domain_randomization(env, profile)
            obs_red = apply_observation_noise(env.observe(0), rng, profile)
            obs_blue = apply_observation_noise(env.observe(1), rng, profile)
        blue_policy = _sample_opponent_policy(historical, role_population, rng)
        total = 0.0
        length = 0
        done = False
        while not done:
            action, logp, value = model.act(obs_red, deterministic=False)
            action_blue = int(blue_policy.select_action(obs_blue, env, 1))
            next_red, next_blue, reward, _blue_reward, done, _info = env.step(action, action_blue)
            reward += _safety_shaping(env)
            if profile is not None and not done:
                apply_external_push(env, rng, profile)
                next_red = apply_observation_noise(env.observe(0), rng, profile)
                next_blue = apply_observation_noise(env.observe(1), rng, profile)
            obs_buf.append(obs_red.copy())
            action_buf.append(action)
            logp_buf.append(logp)
            value_buf.append(value)
            reward_buf.append(float(reward))
            done_buf.append(bool(done))
            total += float(reward)
            length += 1
            obs_red, obs_blue = next_red, next_blue
        episode_returns.append(total)
        episode_lengths.append(length)
    advantages, returns = _gae(np.asarray(reward_buf, dtype=np.float32), np.asarray(value_buf, dtype=np.float32), np.asarray(done_buf, dtype=bool), config)
    return {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "actions": np.asarray(action_buf, dtype=np.int64),
        "old_logp": np.asarray(logp_buf, dtype=np.float32),
        "values": np.asarray(value_buf, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "episodes": config.matches_per_update,
    }


def _ppo_update(model, optimizer, rollout: dict[str, object], config: PPOConfig) -> None:
    obs = torch.as_tensor(rollout["obs"], dtype=torch.float32)
    actions = torch.as_tensor(rollout["actions"], dtype=torch.long)
    old_logp = torch.as_tensor(rollout["old_logp"], dtype=torch.float32)
    advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32)
    returns = torch.as_tensor(rollout["returns"], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    n = int(actions.shape[0])
    policy_losses, value_losses, entropies = [], [], []
    for _epoch in range(config.epochs):
        idx = torch.randperm(n)
        for start in range(0, n, config.batch_size):
            batch = idx[start : start + config.batch_size]
            logits, values = model(obs[batch])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions[batch])
            ratio = torch.exp(logp - old_logp[batch])
            unclipped = ratio * advantages[batch]
            clipped = torch.clamp(ratio, 1.0 - config.clip, 1.0 + config.clip) * advantages[batch]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, returns[batch])
            entropy = dist.entropy().mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
    rollout["policy_loss"] = float(np.mean(policy_losses)) if policy_losses else 0.0
    rollout["value_loss"] = float(np.mean(value_losses)) if value_losses else 0.0
    rollout["entropy"] = float(np.mean(entropies)) if entropies else 0.0


def _gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, config: PPOConfig) -> tuple[np.ndarray, np.ndarray]:
    adv = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + config.gamma * next_value * nonterminal - values[t]
        lastgaelam = delta + config.gamma * config.gae_lambda * nonterminal * lastgaelam
        adv[t] = lastgaelam
        next_value = values[t]
    returns = adv + values
    return adv.astype(np.float32), returns.astype(np.float32)


def _sample_opponent_policy(historical, role_population, rng):
    if historical and rng.random() < 0.55:
        snap = historical[int(rng.integers(0, len(historical)))]
        return NeuralActorPolicy(_snapshot_model(snap), deterministic=False)
    attrs = role_population[int(rng.integers(0, len(role_population)))]
    return AttributePolicy(attrs, lookahead=False)


def _safety_shaping(env: FightEnv) -> float:
    return -0.35 * env.red.falls - 0.025 * max(0.0, 0.25 - env.red.balance) - 0.010 * max(0.0, 0.18 - env.red.stamina)


def _update_training_elo(model, historical, role_population, rng, config: PPOConfig) -> dict[str, float]:
    opponents = [AttributePolicy(attrs, lookahead=False) for attrs in role_population[: min(3, len(role_population))]]
    if historical:
        opponents.append(NeuralActorPolicy(_snapshot_model(historical[-1]), deterministic=True))
    elo = 1000.0
    wins, falls, played = 0.0, 0, 0
    for opponent in opponents:
        env = FightEnv(config=SimConfig(max_steps=max(30, config.max_steps // 2), seed=int(rng.integers(1, 10_000_000))))
        obs_red, obs_blue = env.reset(randomize=True)
        red = NeuralActorPolicy(model, deterministic=True)
        done = False
        while not done:
            obs_red, obs_blue, _rr, _rb, done, _info = env.step(red.select_action(obs_red, env, 0), opponent.select_action(obs_blue, env, 1))
        score = 0.5 if env.winner() == -1 else (1.0 if env.winner() == 0 else 0.0)
        elo, _opp = _update_elo(elo, 1000.0, score, k=20.0)
        wins += score
        falls += int(env.red.falls > 0)
        played += 1
    return {"elo": float(elo), "win_rate": float(wins / max(1, played)), "fall_rate": float(falls / max(1, played))}


def run_policy_leaderboard(model, historical, role_population, out_dir: str | Path, seed: int = 2001, max_steps: int = 80) -> Dict[str, object]:
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    agents = [("ppo_current", NeuralActorPolicy(model, deterministic=True))]
    for snap in historical[-4:]:
        agents.append((snap["name"], NeuralActorPolicy(_snapshot_model(snap), deterministic=True)))
    for attrs in role_population:
        agents.append((attrs.policy_id, AttributePolicy(attrs, lookahead=False)))
    elo = {name: 1000.0 for name, _ in agents}
    rows = []
    for i, (red_name, red_policy) in enumerate(agents):
        for j, (blue_name, blue_policy) in enumerate(agents):
            if i == j:
                continue
            env = FightEnv(config=SimConfig(max_steps=max_steps, seed=int(rng.integers(1, 10_000_000))))
            obs_red, obs_blue = env.reset(randomize=True)
            done = False
            while not done:
                obs_red, obs_blue, _rr, _rb, done, _info = env.step(red_policy.select_action(obs_red, env, 0), blue_policy.select_action(obs_blue, env, 1))
            score = 0.5 if env.winner() == -1 else (1.0 if env.winner() == 0 else 0.0)
            elo[red_name], elo[blue_name] = _update_elo(elo[red_name], elo[blue_name], score)
            rows.append({"red": red_name, "blue": blue_name, "winner": env.winner(), "red_falls": env.red.falls, "blue_falls": env.blue.falls})
    board = []
    for name, value in sorted(elo.items(), key=lambda item: item[1], reverse=True):
        played = [r for r in rows if r["red"] == name or r["blue"] == name]
        score = sum((1.0 if r["winner"] == 0 else 0.5 if r["winner"] == -1 else 0.0) for r in rows if r["red"] == name)
        score += sum((1.0 if r["winner"] == 1 else 0.5 if r["winner"] == -1 else 0.0) for r in rows if r["blue"] == name)
        falls = sum((r["red_falls"] if r["red"] == name else r["blue_falls"]) for r in played)
        board.append({"agent": name, "elo": float(value), "played": len(played), "score_rate": float(score / max(1, len(played))), "falls": int(falls)})
    pd.DataFrame(rows).to_csv(out_dir / "league_matches.csv", index=False)
    pd.DataFrame(board).to_csv(out_dir / "leaderboard.csv", index=False)
    _write_leaderboard_md(out_dir / "LEADERBOARD.md", board)
    summary = {"agents": len(agents), "matches": len(rows), "top_agent": board[0]["agent"] if board else "", "top_elo": board[0]["elo"] if board else 0.0}
    (out_dir / "leaderboard_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "rows": rows, "leaderboard": board}


def _snapshot_model(snap) -> ActorCriticPolicy:
    # The observation dimension is recoverable from the first linear layer.
    obs_dim = int(snap["state"]["encoder.0.weight"].shape[1])
    hidden = int(snap["state"]["encoder.0.weight"].shape[0])
    model = ActorCriticPolicy(obs_dim=obs_dim, hidden=hidden)
    model.load_state_dict(snap["state"])
    model.eval()
    return model


def _save_actor_checkpoint(path: Path, model: ActorCriticPolicy, config: PPOConfig, metrics: dict[str, object]) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "obs_dim": model.obs_dim,
            "hidden": int(model.encoder[0].out_features),
            "action_names": ACTION_NAMES,
            "style_names": STYLE_NAMES,
            "metrics": metrics,
            "ppo_config": config.__dict__,
            "model_type": "actor_critic_ppo",
        },
        path,
    )


def _write_leaderboard_md(path: Path, board: list[dict[str, object]]) -> None:
    lines = ["# GhostFighter League Leaderboard", "", "| Rank | Agent | Elo | Score Rate | Falls |", "|---:|---|---:|---:|---:|"]
    for idx, row in enumerate(board, start=1):
        lines.append(f"| {idx} | `{row['agent']}` | {row['elo']:.1f} | {row['score_rate']:.3f} | {row['falls']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rl_card(path: Path, summary: dict[str, object]) -> None:
    text = f"""# RL Self-Play Training Card

This run trains an actor-critic policy with PPO from match rewards. Generation Zero remains useful for bootstrapping, but this artifact is the first real learning loop: the current policy collects rollouts, updates from advantages, snapshots historical opponents, evaluates against a small league, and writes a leaderboard.

```json
{json.dumps(summary, indent=2)}
```
"""
    path.write_text(text, encoding="utf-8")
