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
                "distance": float(np.hypot(env.red.x - env.blue.x, env.red.y - env.blue.y)),
                "red_edge_clearance": float(env.config.arena_radius - np.hypot(env.red.x, env.red.y)),
                "blue_edge_clearance": float(env.config.arena_radius - np.hypot(env.blue.x, env.blue.y)),
                "red_max_damage": float(max(env.red.damage_vector())),
                "blue_max_damage": float(max(env.blue.damage_vector())),
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
        "summary": {
            "steps": len(frames),
            "winner": env.winner(),
            "red_score": float(env.red.score),
            "blue_score": float(env.blue.score),
            "red_falls": int(env.red.falls),
            "blue_falls": int(env.blue.falls),
            "red_health": float(env.red.health),
            "blue_health": float(env.blue.health),
        },
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
    payload = json.dumps(replay).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GhostFighter Replay Viewer</title>
  <style>
    :root {{ color-scheme: dark; --bg:#11161d; --panel:#1b232e; --line:#334155; --text:#e8edf5; --muted:#93a4b8; --red:#ef4b48; --blue:#4f8cff; --gold:#ffbf5b; --green:#61dc8b; --danger:#ff7a6b; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; background: radial-gradient(circle at 50% 0%, #273446 0, var(--bg) 42rem); color: var(--text); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    header {{ display:flex; justify-content:space-between; gap:18px; align-items:flex-end; margin-bottom:18px; }}
    h1 {{ margin:0; font-size: clamp(26px, 4vw, 46px); letter-spacing:0; }}
    .sub {{ color:var(--muted); margin:8px 0 0; max-width:720px; }}
    .layout {{ display:grid; grid-template-columns:minmax(420px, 1.4fr) minmax(300px, .8fr); gap:18px; align-items:start; }}
    .surface {{ background:rgba(27,35,46,.86); border:1px solid var(--line); border-radius:8px; box-shadow:0 18px 60px rgba(0,0,0,.28); }}
    canvas {{ width:100%; aspect-ratio:1/1; display:block; border-radius:8px; }}
    .controls {{ display:grid; grid-template-columns:auto auto 1fr auto; gap:12px; align-items:center; padding:14px; margin-top:12px; }}
    button {{ border:1px solid #51637a; background:#263244; color:var(--text); border-radius:8px; padding:9px 13px; cursor:pointer; }}
    button:hover {{ background:#314159; }}
    input[type=range] {{ width:100%; accent-color:var(--gold); }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    .metric {{ padding:14px; }}
    .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
    .value {{ font-size:22px; margin-top:4px; }}
    .red {{ color:var(--red); }} .blue {{ color:var(--blue); }} .gold {{ color:var(--gold); }}
    .bar {{ height:8px; border-radius:999px; background:#101720; overflow:hidden; margin-top:8px; }}
    .fill {{ height:100%; border-radius:999px; }}
    .side {{ padding:14px; margin-top:12px; }}
    .side h2 {{ margin:0 0 12px; font-size:16px; }}
    .event {{ padding:9px 0; border-top:1px solid #2c3848; color:#dbe5f2; font-size:13px; }}
    .event:first-child {{ border-top:0; }}
    .chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }}
    .chip {{ border:1px solid #405168; background:#202b39; border-radius:999px; padding:6px 9px; color:#d9e4f2; font-size:12px; }}
    .chip strong {{ color:#fff; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; color:#b7c6d9; overflow:auto; max-height:150px; white-space:pre-wrap; }}
    @media (max-width: 900px) {{ main {{ padding:14px; }} header,.layout {{ display:block; }} .side {{ margin-top:14px; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>GhostFighter Replay</h1>
      <p class="sub">Step through a trained policy fight: actions, rewards, health, balance, stamina, damage, domain randomization, and event timeline.</p>
    </div>
    <div class="metric surface"><div class="label">Winner</div><div id="winner" class="value gold"></div></div>
  </header>
  <section class="layout">
    <div>
      <div class="surface"><canvas id="arena" width="860" height="860"></canvas></div>
      <div class="controls surface">
        <button id="play">Play</button>
        <button id="snapshot">PNG</button>
        <input id="scrub" type="range" min="0" max="0" value="0">
        <span id="caption" class="mono"></span>
      </div>
      <div class="side surface">
        <h2>Reward Trace</h2>
        <canvas id="reward" width="900" height="170"></canvas>
      </div>
    </div>
    <div>
      <div class="grid">
        <div class="metric surface"><div class="label">Red Score</div><div id="redScore" class="value red"></div></div>
        <div class="metric surface"><div class="label">Blue Score</div><div id="blueScore" class="value blue"></div></div>
        <div class="metric surface"><div class="label">Step</div><div id="stepNo" class="value"></div></div>
        <div class="metric surface"><div class="label">Action</div><div id="actions" class="value"></div></div>
      </div>
      <div class="side surface"><h2>Tactical Telemetry</h2><div id="telemetry"></div></div>
      <div class="side surface"><h2>Fighter State</h2><div id="state"></div></div>
      <div class="side surface"><h2>Events</h2><div id="events"></div></div>
      <div class="side surface"><h2>Domain Profile</h2><div id="domain" class="mono"></div></div>
    </div>
  </section>
</main>
<script>
const replay = {payload};
const frames = replay.frames;
const scrub = document.getElementById('scrub');
const canvas = document.getElementById('arena');
const ctx = canvas.getContext('2d');
const rewardCanvas = document.getElementById('reward');
const rctx = rewardCanvas.getContext('2d');
let playing = false;
let timer = null;
scrub.max = Math.max(0, frames.length - 1);
scrub.addEventListener('input', () => draw(Number(scrub.value)));
document.addEventListener('keydown', (event) => {{
  if (event.key === 'ArrowRight') {{
    scrub.value = Math.min(frames.length - 1, Number(scrub.value) + 1);
    draw(Number(scrub.value));
  }}
  if (event.key === 'ArrowLeft') {{
    scrub.value = Math.max(0, Number(scrub.value) - 1);
    draw(Number(scrub.value));
  }}
  if (event.key === ' ') {{
    event.preventDefault();
    document.getElementById('play').click();
  }}
}});
document.getElementById('play').addEventListener('click', () => {{
  playing = !playing;
  document.getElementById('play').textContent = playing ? 'Pause' : 'Play';
  if (playing) timer = setInterval(() => {{
    let next = Number(scrub.value) + 1;
    if (next >= frames.length) next = 0;
    scrub.value = next;
    draw(next);
  }}, 120);
  else clearInterval(timer);
}});
document.getElementById('snapshot').addEventListener('click', () => {{
  const link = document.createElement('a');
  link.download = `ghostfighter-step-${{frames[Number(scrub.value)].step}}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
}});
function sx(x) {{ return canvas.width / 2 + x / replay.arena_radius * canvas.width * 0.42; }}
function sy(y) {{ return canvas.height / 2 - y / replay.arena_radius * canvas.height * 0.42; }}
function meter(label, value, color) {{
  const pct = Math.max(0, Math.min(1, value));
  return `<div class="label">${{label}}</div><div class="bar"><div class="fill" style="width:${{pct*100}}%;background:${{color}}"></div></div>`;
}}
function fighter(f, color, label) {{
  const x = sx(f.x), y = sy(f.y);
  ctx.shadowColor = color;
  ctx.shadowBlur = 18;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, f.fallen ? 22 : 16, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.fillStyle = '#f8fafc';
  ctx.font = 'bold 14px system-ui';
  ctx.fillText(label, x - 5, y + 34);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + Math.cos(f.theta) * 36, y - Math.sin(f.theta) * 36);
  ctx.stroke();
}}
function drawRangeAndBoundary(f) {{
  const midx = (sx(f.red.x) + sx(f.blue.x)) / 2;
  const midy = (sy(f.red.y) + sy(f.blue.y)) / 2;
  ctx.strokeStyle = f.distance < 0.9 ? 'rgba(255,191,91,.85)' : 'rgba(147,164,184,.45)';
  ctx.setLineDash([8, 8]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(sx(f.red.x), sy(f.red.y));
  ctx.lineTo(sx(f.blue.x), sy(f.blue.y));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#dbe5f2';
  ctx.font = '13px ui-monospace, monospace';
  ctx.fillText(`${{f.distance.toFixed(2)}}m`, midx + 8, midy - 8);
  for (const [side, color] of [[f.red, 'rgba(239,75,72,.28)'], [f.blue, 'rgba(79,140,255,.25)']]) {{
    const edge = replay.arena_radius - Math.hypot(side.x, side.y);
    if (edge < 0.7) {{
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(sx(side.x), sy(side.y), 46, 0, Math.PI * 2);
      ctx.fill();
    }}
  }}
}}
function drawTrails(i) {{
  for (let k = Math.max(0, i - 16); k < i; k++) {{
    const a = (k - Math.max(0, i - 16) + 1) / 16;
    for (const [f, color] of [[frames[k].red, '239,75,72'], [frames[k].blue, '79,140,255']]) {{
      ctx.fillStyle = `rgba(${{color}},${{0.08 + a * 0.20}})`;
      ctx.beginPath(); ctx.arc(sx(f.x), sy(f.y), 3 + a * 5, 0, Math.PI*2); ctx.fill();
    }}
  }}
}}
function drawReward(i) {{
  const vals = frames.map(f => f.reward_red);
  const min = Math.min(...vals, -0.01), max = Math.max(...vals, 0.01);
  rctx.clearRect(0,0,rewardCanvas.width,rewardCanvas.height);
  rctx.strokeStyle = '#334155'; rctx.lineWidth = 1;
  rctx.beginPath(); rctx.moveTo(0, rewardCanvas.height/2); rctx.lineTo(rewardCanvas.width, rewardCanvas.height/2); rctx.stroke();
  rctx.strokeStyle = '#ffbf5b'; rctx.lineWidth = 3; rctx.beginPath();
  vals.forEach((v, idx) => {{
    const x = idx / Math.max(1, vals.length - 1) * rewardCanvas.width;
    const y = rewardCanvas.height - ((v - min) / (max - min)) * rewardCanvas.height;
    if (idx === 0) rctx.moveTo(x, y); else rctx.lineTo(x, y);
  }});
  rctx.stroke();
  frames.forEach((frame, idx) => {{
    if (!frame.events.length) return;
    const x = idx / Math.max(1, vals.length - 1) * rewardCanvas.width;
    rctx.fillStyle = frame.events.some(e => e.kind === 'fall') ? '#ff7a6b' : '#61dc8b';
    rctx.fillRect(x - 2, 8, 4, 18);
  }});
  rctx.fillStyle = '#e8edf5';
  rctx.fillRect(i / Math.max(1, vals.length - 1) * rewardCanvas.width - 2, 0, 4, rewardCanvas.height);
}}
function draw(i) {{
  const f = frames[i];
  ctx.fillStyle = '#121821';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#d5dce7';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(canvas.width/2, canvas.height/2, canvas.width * 0.42, 0, Math.PI*2);
  ctx.stroke();
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
  [0.20,0.32].forEach(r => {{ ctx.beginPath(); ctx.arc(canvas.width/2, canvas.height/2, canvas.width*r, 0, Math.PI*2); ctx.stroke(); }});
  drawTrails(i);
  drawRangeAndBoundary(f);
  fighter(f.red, '#c0322e', 'R');
  fighter(f.blue, '#3056aa', 'B');
  document.getElementById('caption').textContent = `step ${{f.step}} | red: ${{f.red_action}} | blue: ${{f.blue_action}}`;
  document.getElementById('winner').textContent = replay.summary.winner === 0 ? 'RED' : replay.summary.winner === 1 ? 'BLUE' : 'DRAW';
  document.getElementById('redScore').textContent = f.red.score.toFixed(1);
  document.getElementById('blueScore').textContent = f.blue.score.toFixed(1);
  document.getElementById('stepNo').textContent = `${{i+1}}/${{frames.length}}`;
  document.getElementById('actions').textContent = `${{f.red_action}} / ${{f.blue_action}}`;
  document.getElementById('telemetry').innerHTML = telemetryPanel(f);
  document.getElementById('state').innerHTML = fighterPanel('Red', f.red, 'var(--red)') + fighterPanel('Blue', f.blue, 'var(--blue)');
  document.getElementById('events').innerHTML = f.events.length ? f.events.map(e => `<div class="event">${{e.text}} (${{e.kind}})</div>`).join('') : '<div class="event">No event</div>';
  drawReward(i);
}}
function fighterPanel(name, f, color) {{
  const damage = Math.max(...f.damage);
  return `<div class="event"><b style="color:${{color}}">${{name}}</b> score ${{f.score.toFixed(1)}} falls ${{f.falls}}${{f.fallen ? ' <span class="chip"><strong>fallen</strong></span>' : ''}}<br>${{meter('health', f.health/100, color)}}${{meter('balance', f.balance, 'var(--green)')}}${{meter('stamina', f.stamina, 'var(--gold)')}}${{meter('damage', damage, 'var(--danger)')}}</div>`;
}}
function telemetryPanel(f) {{
  const redEdge = f.red_edge_clearance;
  const blueEdge = f.blue_edge_clearance;
  return `<div class="chips">
    <span class="chip"><strong>range</strong> ${{f.distance.toFixed(2)}}m</span>
    <span class="chip"><strong>red edge</strong> ${{redEdge.toFixed(2)}}m</span>
    <span class="chip"><strong>blue edge</strong> ${{blueEdge.toFixed(2)}}m</span>
    <span class="chip"><strong>red dmg</strong> ${{f.red_max_damage.toFixed(2)}}</span>
    <span class="chip"><strong>blue dmg</strong> ${{f.blue_max_damage.toFixed(2)}}</span>
    <span class="chip"><strong>reward</strong> ${{f.reward_red.toFixed(3)}}</span>
  </div>`;
}}
document.getElementById('domain').textContent = replay.domain_profile ? JSON.stringify(replay.domain_profile, null, 2) : 'disabled';
draw(0);
</script>
</body>
</html>
"""
