from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .attributes import AttributePolicy
from .config import ACTION_NAMES
from .domain import apply_domain_randomization, apply_external_push, apply_observation_noise, sample_domain_randomization
from .env import FightEnv, SimConfig
from .rl import NeuralActorPolicy, load_actor_checkpoint
from .selfplay import _make_population


def make_replay_viewer(
    policy_path: str | Path,
    out_dir: str | Path,
    seed: int = 2601,
    max_steps: int = 100,
    domain_randomization: bool = True,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    replay = record_policy_replay(policy_path, seed=seed, max_steps=max_steps, domain_randomization=domain_randomization)
    json_path = out_dir / "replay.json"
    html_path = out_dir / "replay_viewer.html"
    json_path.write_text(json.dumps(replay, indent=2), encoding="utf-8")
    html_path.write_text(_html(replay), encoding="utf-8")
    return {"replay": str(json_path), "viewer": str(html_path)}


def record_policy_replay(
    policy_path: str | Path,
    seed: int = 2601,
    max_steps: int = 100,
    domain_randomization: bool = True,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    model, ckpt = load_actor_checkpoint(policy_path)
    env = FightEnv(config=SimConfig(max_steps=max_steps, seed=seed), seed=seed)
    obs_red, obs_blue = env.reset(randomize=True)
    profile = None
    if domain_randomization:
        profile = sample_domain_randomization(rng, 0.50)
        apply_domain_randomization(env, profile)
        obs_red = apply_observation_noise(env.observe(0), rng, profile)
        obs_blue = apply_observation_noise(env.observe(1), rng, profile)
    role_population = _make_population(variants_per_role=1, seed=seed + 13)
    opponent = AttributePolicy(role_population[int(rng.integers(0, len(role_population)))], lookahead=False)
    policy = NeuralActorPolicy(model, deterministic=True)
    frames = []
    done = False
    while not done:
        action_red = policy.select_action(obs_red, env, 0)
        action_blue = opponent.select_action(obs_blue, env, 1)
        obs_red, obs_blue, reward_red, reward_blue, done, info = env.step(action_red, action_blue)
        frames.append(
            {
                "step": env.step_count,
                "red_action": ACTION_NAMES[action_red],
                "blue_action": ACTION_NAMES[action_blue],
                "reward_red": float(reward_red),
                "reward_blue": float(reward_blue),
                "winner": int(env.winner()) if done else None,
                "events": info["events"],
                "red": _fighter(env.red),
                "blue": _fighter(env.blue),
            }
        )
        if profile is not None and not done:
            apply_external_push(env, rng, profile)
            obs_red = apply_observation_noise(env.observe(0), rng, profile)
            obs_blue = apply_observation_noise(env.observe(1), rng, profile)
    return {
        "policy": str(policy_path),
        "policy_metrics": ckpt.get("metrics", {}),
        "seed": seed,
        "max_steps": max_steps,
        "domain_randomization": bool(domain_randomization),
        "domain_profile": profile.__dict__ if profile else None,
        "arena_radius": env.config.arena_radius,
        "frames": frames,
    }


def _fighter(f) -> dict[str, object]:
    return {
        "x": float(f.x),
        "y": float(f.y),
        "theta": float(f.theta),
        "health": float(f.health),
        "stamina": float(f.stamina),
        "balance": float(f.balance),
        "guard": float(f.guard),
        "score": float(f.score),
        "falls": int(f.falls),
        "fallen": bool(f.fallen),
        "damage": [float(x) for x in f.damage_vector()],
    }


def _html(replay: dict[str, object]) -> str:
    payload = json.dumps(replay)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GhostFighter Replay Viewer</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f7f7f3; color: #222; }}
    main {{ max-width: 980px; margin: 0 auto; }}
    canvas {{ background: white; border: 1px solid #bbb; width: 100%; max-width: 760px; aspect-ratio: 1 / 1; display: block; }}
    .row {{ display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }}
    .panel {{ min-width: 260px; flex: 1; }}
    input[type=range] {{ width: 100%; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    td {{ border-bottom: 1px solid #ddd; padding: 5px 2px; }}
    .event {{ font-size: 13px; padding: 6px 0; border-bottom: 1px solid #ddd; }}
  </style>
</head>
<body>
<main>
  <h1>GhostFighter Replay Viewer</h1>
  <input id="scrub" type="range" min="0" max="0" value="0">
  <p id="caption"></p>
  <div class="row">
    <canvas id="arena" width="760" height="760"></canvas>
    <div class="panel">
      <h2>State</h2>
      <table id="state"></table>
      <h2>Events</h2>
      <div id="events"></div>
    </div>
  </div>
</main>
<script>
const replay = {payload};
const frames = replay.frames;
const scrub = document.getElementById('scrub');
const canvas = document.getElementById('arena');
const ctx = canvas.getContext('2d');
scrub.max = Math.max(0, frames.length - 1);
scrub.addEventListener('input', () => draw(Number(scrub.value)));
function sx(x) {{ return canvas.width / 2 + x / replay.arena_radius * canvas.width * 0.42; }}
function sy(y) {{ return canvas.height / 2 - y / replay.arena_radius * canvas.height * 0.42; }}
function fighter(f, color, label) {{
  const x = sx(f.x), y = sy(f.y);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, f.fallen ? 17 : 13, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = '#222';
  ctx.fillText(label, x - 5, y + 29);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + Math.cos(f.theta) * 28, y - Math.sin(f.theta) * 28);
  ctx.stroke();
}}
function draw(i) {{
  const f = frames[i];
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#444';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(canvas.width/2, canvas.height/2, canvas.width * 0.42, 0, Math.PI*2);
  ctx.stroke();
  fighter(f.red, '#c0322e', 'R');
  fighter(f.blue, '#3056aa', 'B');
  document.getElementById('caption').textContent = `step ${{f.step}} | red: ${{f.red_action}} | blue: ${{f.blue_action}}`;
  document.getElementById('state').innerHTML = rows('red', f.red) + rows('blue', f.blue);
  document.getElementById('events').innerHTML = f.events.length ? f.events.map(e => `<div class="event">${{e.text}} (${{e.kind}})</div>`).join('') : '<div class="event">No event</div>';
}}
function rows(name, f) {{
  return `<tr><td><b>${{name}}</b></td><td>hp ${{f.health.toFixed(1)}} bal ${{f.balance.toFixed(2)}} sta ${{f.stamina.toFixed(2)}} score ${{f.score.toFixed(1)}} falls ${{f.falls}}</td></tr>`;
}}
draw(0);
</script>
</body>
</html>
"""
