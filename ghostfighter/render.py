from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from .config import ACTION_NAMES, STYLE_NAMES
from .env import FightEnv, SimConfig
from .evaluate import run_match, _setup_boundary_trap
from .models import load_policy_checkpoint
from .policies import EnsembleOpponent, NeuralGhostPolicy, ScriptedPilot
from .safety import CombatSafetyFirewall


BG = (245, 245, 242)
INK = (35, 35, 35)
RED = (192, 50, 46)
BLUE = (48, 86, 170)
GREEN = (50, 130, 70)
ORANGE = (218, 125, 36)
GRAY = (170, 170, 170)


def _font(size: int = 13):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def render_env_frame(
    env: FightEnv,
    title: str,
    subtitle: str = "",
    width: int = 960,
    height: int = 640,
    history: list[FightEnv] | None = None,
) -> Image.Image:
    scale_factor = 2
    width2, height2 = width * scale_factor, height * scale_factor
    img = Image.new("RGB", (width2, height2), (18, 22, 28))
    d = ImageDraw.Draw(img)
    font_title = _font(24 * scale_factor)
    font = _font(14 * scale_factor)
    small = _font(11 * scale_factor)
    tiny = _font(9 * scale_factor)
    margin = 28 * scale_factor
    panel_top = 72 * scale_factor
    panel_h = height2 - panel_top - 124 * scale_factor
    panel_w = width2 - 2 * margin
    cx = margin + panel_w / 2
    cy = panel_top + panel_h / 2
    scale = min(panel_w, panel_h) / (2 * env.config.arena_radius * 1.08)

    # Header.
    d.rectangle((0, 0, width2, 68 * scale_factor), fill=(28, 34, 43))
    d.text((margin, 15 * scale_factor), title, fill=(245, 247, 250), font=font_title)
    if subtitle:
        d.text((margin, 45 * scale_factor), subtitle, fill=(168, 178, 190), font=font)
    d.rounded_rectangle((width2 - 190 * scale_factor, 17 * scale_factor, width2 - margin, 51 * scale_factor), radius=10 * scale_factor, fill=(38, 46, 58), outline=(70, 84, 104), width=1 * scale_factor)
    d.text((width2 - 178 * scale_factor, 25 * scale_factor), f"STEP {env.step_count}/{env.config.max_steps}", fill=(225, 231, 238), font=font)

    # Arena.
    r = env.config.arena_radius * scale
    d.ellipse((cx - r - 18 * scale_factor, cy - r - 18 * scale_factor, cx + r + 18 * scale_factor, cy + r + 18 * scale_factor), fill=(22, 29, 37), outline=(53, 67, 84), width=2 * scale_factor)
    for frac, color, w in [(1.0, (215, 220, 226), 3), (0.78, (63, 79, 99), 1), (0.48, (47, 59, 75), 1)]:
        rr = r * frac
        d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=color, width=w * scale_factor)
    for a in range(0, 360, 30):
        rad = math.radians(a)
        x1, y1 = cx + math.cos(rad) * r * 0.48, cy + math.sin(rad) * r * 0.48
        x2, y2 = cx + math.cos(rad) * r, cy + math.sin(rad) * r
        d.line((x1, y1, x2, y2), fill=(37, 48, 62), width=1 * scale_factor)
    d.line((cx - r, cy, cx + r, cy), fill=(48, 60, 76), width=1 * scale_factor)
    d.line((cx, cy - r, cx, cy + r), fill=(48, 60, 76), width=1 * scale_factor)

    def xy(f):
        return cx + f.x * scale, cy - f.y * scale

    def draw_trails():
        if not history:
            return
        recent = history[-14:]
        for idx, old in enumerate(recent):
            alpha = (idx + 1) / max(1, len(recent))
            for f, color in [(old.red, (230, 68, 64)), (old.blue, (74, 126, 255))]:
                x, y = xy(f)
                radius = (3 + 4 * alpha) * scale_factor
                fill = tuple(int(20 + c * alpha * 0.55) for c in color)
                d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)

    def draw_fighter(f, color, label):
        x, y = xy(f)
        body_r = (16 if not f.fallen else 20) * scale_factor
        glow_r = body_r + 9 * scale_factor
        fill = tuple(int(c * (0.50 if f.fallen else 1.0)) for c in color)
        d.ellipse((x - glow_r, y - glow_r, x + glow_r, y + glow_r), fill=tuple(max(0, int(c * 0.25)) for c in color))
        d.ellipse((x - body_r, y - body_r, x + body_r, y + body_r), fill=fill, outline=(245, 247, 250), width=2 * scale_factor)
        hx = x + math.cos(f.theta) * 31 * scale_factor
        hy = y - math.sin(f.theta) * 31 * scale_factor
        d.line((x, y, hx, hy), fill=(245, 247, 250), width=4 * scale_factor)
        if f.guard > 0.35:
            guard_r = 29 * scale_factor
            d.arc((x - guard_r, y - guard_r, x + guard_r, y + guard_r), start=205, end=335, fill=(87, 221, 132), width=4 * scale_factor)
        if f.last_action in [ACTION_NAMES.index("jab"), ACTION_NAMES.index("cross"), ACTION_NAMES.index("hook"), ACTION_NAMES.index("push")]:
            d.line((x, y, hx + math.cos(f.theta) * 17 * scale_factor, hy - math.sin(f.theta) * 17 * scale_factor), fill=(255, 180, 72), width=6 * scale_factor)
        if f.last_action == ACTION_NAMES.index("low_kick"):
            d.line((x, y, hx + math.cos(f.theta) * 11 * scale_factor, hy - math.sin(f.theta) * 11 * scale_factor), fill=(255, 180, 72), width=8 * scale_factor)
        d.text((x - 8 * scale_factor, y - 8 * scale_factor), label, fill=(20, 22, 26), font=small)
        if f.last_override:
            d.rounded_rectangle((x - 36 * scale_factor, y - 46 * scale_factor, x + 36 * scale_factor, y - 24 * scale_factor), radius=7 * scale_factor, fill=(255, 220, 120), outline=(255, 172, 48), width=2 * scale_factor)
            d.text((x - 25 * scale_factor, y - 44 * scale_factor), "SAFE", fill=(35, 27, 10), font=tiny)

    draw_trails()
    draw_fighter(env.red, RED, "R")
    draw_fighter(env.blue, BLUE, "B")

    def meter(x, y, w, label, val, maxv, color):
        d.text((x, y - 18 * scale_factor), label, fill=(210, 218, 228), font=tiny)
        d.rounded_rectangle((x, y, x + w, y + 12 * scale_factor), radius=5 * scale_factor, fill=(35, 42, 52), outline=(66, 78, 94), width=1 * scale_factor)
        ww = max(0, min(w, val / maxv * w))
        d.rounded_rectangle((x, y, x + ww, y + 12 * scale_factor), radius=5 * scale_factor, fill=color)

    def card(x, y, f, side_name, color):
        card_w = 285 * scale_factor
        card_h = 98 * scale_factor
        d.rounded_rectangle((x, y, x + card_w, y + card_h), radius=14 * scale_factor, fill=(28, 34, 43), outline=(58, 72, 90), width=1 * scale_factor)
        d.ellipse((x + 14 * scale_factor, y + 15 * scale_factor, x + 42 * scale_factor, y + 43 * scale_factor), fill=color, outline=(235, 239, 244), width=1 * scale_factor)
        d.text((x + 52 * scale_factor, y + 13 * scale_factor), side_name, fill=(244, 247, 250), font=font)
        d.text((x + 52 * scale_factor, y + 38 * scale_factor), f"score {f.score:0.1f} | falls {f.falls}", fill=(159, 171, 187), font=tiny)
        for i, (name, val, maxv, bar_color) in enumerate(
            [
                ("HP", f.health, 100.0, color),
                ("BAL", f.balance, 1.0, (84, 224, 135)),
                ("STA", f.stamina, 1.0, (255, 183, 72)),
            ]
        ):
            meter(x + 15 * scale_factor + i * 88 * scale_factor, y + 74 * scale_factor, 72 * scale_factor, name, val, maxv, bar_color)
        act = ACTION_NAMES[int(f.last_action)]
        if f.last_override:
            prop = ACTION_NAMES[int(f.last_proposed_action)]
            act = f"{prop}>{act}"
        d.rounded_rectangle((x + 172 * scale_factor, y + 14 * scale_factor, x + 270 * scale_factor, y + 42 * scale_factor), radius=8 * scale_factor, fill=(39, 48, 61))
        d.text((x + 181 * scale_factor, y + 21 * scale_factor), act[:13], fill=(234, 239, 246), font=tiny)
        if f.last_risk > 0:
            d.text((x + 174 * scale_factor, y + 48 * scale_factor), f"risk {f.last_risk:0.2f}", fill=(255, 201, 98), font=tiny)
        damage = max(f.damage_vector())
        if damage > 0.02:
            d.text((x + 224 * scale_factor, y + 48 * scale_factor), f"dmg {damage:0.2f}", fill=(255, 126, 112), font=tiny)

    card(28 * scale_factor, height2 - 108 * scale_factor, env.red, "RED ghost", (230, 68, 64))
    card(width2 - 313 * scale_factor, height2 - 108 * scale_factor, env.blue, "BLUE opponent", (74, 126, 255))
    event_text = " | ".join(e.text for e in env.last_events[-2:])
    if event_text:
        d.rounded_rectangle((width2 / 2 - 220 * scale_factor, height2 - 91 * scale_factor, width2 / 2 + 220 * scale_factor, height2 - 58 * scale_factor), radius=11 * scale_factor, fill=(46, 36, 27), outline=(255, 172, 64), width=1 * scale_factor)
        d.text((width2 / 2 - 205 * scale_factor, height2 - 83 * scale_factor), event_text[:76], fill=(255, 216, 152), font=tiny)
    return img.resize((width, height), Image.Resampling.LANCZOS)


def make_demo_gif(
    model_path: str | Path,
    out_path: str | Path,
    style_id: int = 0,
    seed: int = 909,
    fps: int = 10,
    max_steps: int = 110,
) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model, _ = load_policy_checkpoint(str(model_path))
    style_name = STYLE_NAMES[int(style_id)]
    frames: List[Image.Image] = []

    segments = []
    # Segment 1: Generation Zero policy reference.
    scripted = ScriptedPilot(style_name, seed=seed + 1)
    opponent = EnsembleOpponent(seed=seed + 2)
    res, trace = run_match(scripted, opponent, seed=seed + 3, style_name=style_name, mode="scripted_reference", max_steps=max_steps, collect_trace=True)
    segments.append(("1. Generation Zero policy", "attribute-driven policy behavior used for rollouts", trace))
    # Segment 2: raw ghost.
    raw = NeuralGhostPolicy(model, style_id=style_id, deterministic=True)
    opponent = EnsembleOpponent(seed=seed + 20)
    res, trace = run_match(raw, opponent, seed=seed + 21, style_name=style_name, mode="ghost_raw", max_steps=max_steps, collect_trace=True)
    segments.append(("2. Autonomous ghost", "policy learned from Generation Zero rollouts, no safety gate", trace))
    # Segment 3: safe ghost.
    safe = NeuralGhostPolicy(model, style_id=style_id, deterministic=True)
    opponent = EnsembleOpponent(seed=seed + 20)
    firewall = CombatSafetyFirewall(threshold=0.62)
    res, trace = run_match(safe, opponent, seed=seed + 21, style_name=style_name, mode="ghost_firewall", firewall=firewall, max_steps=max_steps, collect_trace=True)
    segments.append(("3. Ghost + combat safety firewall", "unsafe actions are replaced before execution", trace))
    # Segment 4: adversarial safety case.
    safe = NeuralGhostPolicy(model, style_id=style_id, deterministic=True)
    opponent = EnsembleOpponent(seed=seed + 40)
    firewall = CombatSafetyFirewall(threshold=0.62)
    res, trace = run_match(
        safe,
        opponent,
        seed=seed + 41,
        style_name=style_name,
        mode="ghost_firewall_adversarial",
        firewall=firewall,
        max_steps=min(max_steps, 95),
        collect_trace=True,
        stress_level=0.45,
        scenario_setup=_setup_boundary_trap,
    )
    segments.append(("4. Adversarial boundary trap", "firewall under low-balance ring pressure", trace))

    for title, subtitle, trace in segments:
        if not trace:
            continue
        sample_every = max(1, len(trace) // 35)
        for i, item in enumerate(trace):
            if i % sample_every != 0 and i != len(trace) - 1:
                continue
            env = item["env"]
            hist = [h["env"] for h in trace[max(0, i - 14) : i + 1]]
            frames.append(render_env_frame(env, title, subtitle, history=hist))
        # Hold title moment.
        if frames:
            frames.extend([frames[-1].copy() for _ in range(4)])

    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    return str(out_path)


def make_dashboard(report_dir: str | Path, out_path: str | Path | None = None) -> str:
    report_dir = Path(report_dir)
    out_path = Path(out_path) if out_path else report_dir / "dashboard.png"
    df = pd.read_csv(report_dir / "match_results.csv")
    preferred = ["raw", "firewall", "raw_stress", "firewall_stress"]
    modes_present = [m for m in preferred if m in set(df["mode"])] + [m for m in df["mode"].unique() if m not in preferred]
    mode = df.groupby("mode").agg(
        win_rate=("winner", lambda x: float((x == 0).mean())),
        fall_rate=("red_falls", lambda x: float((x > 0).mean())),
        unsafe_rate=("unsafe_rate", "mean"),
        avg_risk=("avg_risk", "mean"),
    ).reindex(modes_present).reset_index()
    health = df.groupby("mode").apply(lambda g: float((g["red_health"] - g["blue_health"]).mean()), include_groups=False)
    mode["health_margin"] = mode["mode"].map(health).fillna(0.0)

    style = df.groupby(["mode", "style"]).agg(
        win_rate=("winner", lambda x: float((x == 0).mean())),
        fall_rate=("red_falls", lambda x: float((x > 0).mean())),
        attack_rate=("red_attack_rate", "mean"),
        guard_rate=("red_guard_rate", "mean"),
        unsafe_rate=("unsafe_rate", "mean"),
    ).reset_index()

    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    x = np.arange(len(mode))
    ax1.bar(x, mode["win_rate"])
    ax1.set_xticks(x, mode["mode"], rotation=18)
    ax1.set_ylim(0, 1)
    ax1.set_title("Ghost win rate by mode")
    ax1.set_ylabel("rate")

    ax2.bar(x, mode["fall_rate"])
    ax2.set_xticks(x, mode["mode"], rotation=18)
    ax2.set_ylim(0, max(0.1, min(1.0, mode["fall_rate"].max() * 1.25 + 0.02)))
    ax2.set_title("Red knockdown rate by mode")

    styles = STYLE_NAMES
    compare_a, compare_b = ("raw_stress", "firewall_stress") if "raw_stress" in set(df["mode"]) else ("raw", "firewall")
    width = 0.36
    a = style[style["mode"] == compare_a].set_index("style").reindex(styles)
    b = style[style["mode"] == compare_b].set_index("style").reindex(styles)
    sx = np.arange(len(styles))
    ax3.bar(sx - width / 2, a["fall_rate"].fillna(0), width, label=compare_a)
    ax3.bar(sx + width / 2, b["fall_rate"].fillna(0), width, label=compare_b)
    ax3.set_xticks(sx, styles, rotation=20)
    ax3.set_ylim(0, max(0.1, min(1.0, max(a["fall_rate"].fillna(0).max(), b["fall_rate"].fillna(0).max()) * 1.25 + 0.02)))
    ax3.set_title("Safety ablation by style")
    ax3.legend()

    ax4.bar(x, mode["unsafe_rate"], label="unsafe rejection rate")
    ax4.plot(x, mode["avg_risk"], marker="o", label="average estimated risk")
    ax4.set_xticks(x, mode["mode"], rotation=18)
    ax4.set_ylim(0, max(0.1, min(1.0, max(mode["unsafe_rate"].max(), mode["avg_risk"].max()) * 1.35 + 0.02)))
    ax4.set_title("Safety gate activity")
    ax4.legend()

    fig.suptitle("GhostFighter evaluation dashboard", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def make_safety_dashboard(report_dir: str | Path, out_path: str | Path | None = None) -> str:
    report_dir = Path(report_dir)
    out_path = Path(out_path) if out_path else report_dir / "safety_dashboard.png"
    scenario_path = report_dir / "scenario_results.csv"
    summary_path = report_dir / "scenario_summary.json"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario results: {scenario_path}")
    df = pd.read_csv(scenario_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    by_scenario = df.groupby(["scenario", "mode"]).agg(
        win_rate=("winner", lambda x: float((x == 0).mean())),
        fall_rate=("red_falls", lambda x: float((x > 0).mean())),
        unsafe_rate=("unsafe_rate", "mean"),
    ).reset_index()
    scenarios = list(dict.fromkeys(df["scenario"].tolist()))
    raw = by_scenario[by_scenario["mode"] == "raw"].set_index("scenario").reindex(scenarios).fillna(0)
    firewall = by_scenario[by_scenario["mode"] == "firewall"].set_index("scenario").reindex(scenarios).fillna(0)
    reasons = summary.get("firewall_reason_counts", {})
    counter = summary.get("counterfactuals", {})

    fig = plt.figure(figsize=(13, 9))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    x = np.arange(len(scenarios))
    width = 0.36
    ax1.bar(x - width / 2, raw["fall_rate"], width, label="raw")
    ax1.bar(x + width / 2, firewall["fall_rate"], width, label="firewall")
    ax1.set_xticks(x, scenarios, rotation=24, ha="right")
    ax1.set_ylim(0, 1)
    ax1.set_title("Adversarial red fall rate")
    ax1.legend()

    ax2.bar(x - width / 2, raw["win_rate"], width, label="raw")
    ax2.bar(x + width / 2, firewall["win_rate"], width, label="firewall")
    ax2.set_xticks(x, scenarios, rotation=24, ha="right")
    ax2.set_ylim(0, 1)
    ax2.set_title("Scenario win rate")
    ax2.legend()

    top_reasons = list(reasons.items())[:8]
    labels = [r for r, _ in top_reasons] or ["none"]
    vals = [v for _, v in top_reasons] or [0]
    ax3.barh(np.arange(len(labels)), vals)
    ax3.set_yticks(np.arange(len(labels)), labels)
    ax3.invert_yaxis()
    ax3.set_title("Firewall override reasons")

    counter_labels = ["avoided fall", "avoided boundary", "damage saved", "balance saved"]
    counter_vals = [
        counter.get("avoided_fall_rate", 0.0),
        counter.get("avoided_boundary_loss_rate", 0.0),
        counter.get("avg_avoided_damage", 0.0),
        counter.get("avg_balance_saved", 0.0),
    ]
    ax4.bar(counter_labels, counter_vals)
    ax4.set_title("One-step counterfactual safety deltas")
    ax4.tick_params(axis="x", rotation=18)

    fig.suptitle("GhostFighter safety benchmark dashboard", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def write_model_card(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    card_path = run_dir / "MODEL_CARD.md"
    data_summary = json.loads((run_dir / "data" / "traces.summary.json").read_text()) if (run_dir / "data" / "traces.summary.json").exists() else {}
    gen0_summary = json.loads((run_dir / "gen0" / "attribute_dataset_summary.json").read_text()) if (run_dir / "gen0" / "attribute_dataset_summary.json").exists() else {}
    train_metrics = json.loads((run_dir / "models" / "training_metrics.json").read_text()) if (run_dir / "models" / "training_metrics.json").exists() else {}
    eval_summary = json.loads((run_dir / "reports" / "eval_summary.json").read_text()) if (run_dir / "reports" / "eval_summary.json").exists() else {}
    scenario_summary = json.loads((run_dir / "reports" / "scenario_summary.json").read_text()) if (run_dir / "reports" / "scenario_summary.json").exists() else {}
    safety_tuning = json.loads((run_dir / "reports" / "safety_tuning.json").read_text()) if (run_dir / "reports" / "safety_tuning.json").exists() else {}
    scaling = json.loads((run_dir / "scaling" / "scaling_study.json").read_text()) if (run_dir / "scaling" / "scaling_study.json").exists() else {}
    selfplay = json.loads((run_dir / "selfplay" / "selfplay_summary.json").read_text()) if (run_dir / "selfplay" / "selfplay_summary.json").exists() else {}
    rl = json.loads((run_dir / "rl" / "ppo_summary.json").read_text()) if (run_dir / "rl" / "ppo_summary.json").exists() else {}
    robustness = json.loads((run_dir / "robustness" / "robustness_summary.json").read_text()) if (run_dir / "robustness" / "robustness_summary.json").exists() else {}
    text = f"""# GhostFighter Model Card

## Model

GhostFighter uses a conditional behavior-cloning policy. A single PyTorch network receives the combat observation plus a policy-condition embedding and predicts one high-level humanoid skill token.

## Training Data

- Source: {data_summary.get('source', 'n/a')}
- Episodes: {data_summary.get('episodes', 'n/a')}
- Trace samples: {data_summary.get('samples', 'n/a')}
- Observation dimension: {data_summary.get('obs_dim', 'n/a')}
- Policy archetypes: {', '.join(STYLE_NAMES)}
- Generation Zero variants: {data_summary.get('policy_variants', gen0_summary.get('policy_variants', 'n/a'))}
- Action entropy: {data_summary.get('action_entropy', gen0_summary.get('action_entropy', 'n/a'))}

## Generation Zero

Generation Zero is created from user-configurable policy attributes rather than fixed scripted pilots. The user-facing archetypes keep fighting-genre language, but each row is a robotics-style behavior prior: engagement drive, guard discipline, counter timing, lateral mobility, stamina discipline, boundary awareness, damage targeting, risk tolerance, and close-range pressure.

```json
{json.dumps({k: v for k, v in gen0_summary.items() if k in ['source', 'policy_variants', 'episodes', 'samples', 'variant_episode_counts']}, indent=2)}
```

## Training Metrics

- Validation action accuracy: {train_metrics.get('val_acc', 'n/a')}
- Best validation action accuracy: {train_metrics.get('best_val_acc', 'n/a')}
- Dataset samples: {train_metrics.get('dataset_samples', 'n/a')}

## Evaluation

```json
{json.dumps(eval_summary.get('by_mode', []), indent=2)}
```

## Safety Benchmark

```json
{json.dumps(scenario_summary.get('by_mode', []), indent=2)}
```

## Self-Improvement

```json
{json.dumps(safety_tuning, indent=2)}
```

## Population Self-Play

```json
{json.dumps(selfplay, indent=2)}
```

## PPO Self-Play Training

```json
{json.dumps(rl, indent=2)}
```

## Robustness Ablations

```json
{json.dumps(robustness, indent=2)}
```

## Scaling Study

```json
{json.dumps({k: v for k, v in scaling.items() if k != 'rows'}, indent=2)}
```

## Intended Use And Limits

This is a self-contained autonomy and safety architecture prototype for robot-combat policy development. It is not a hardware dynamics certificate. The simulator uses high-level skill tokens so reviewers can inspect the policy-data flywheel, conditional policy learning, stress evaluation, and safety-firewall design without external robotics stacks.
"""
    card_path.write_text(text, encoding="utf-8")
    return str(card_path)


def write_run_card(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    card_path = run_dir / "RUN_CARD.md"
    data_summary = json.loads((run_dir / "data" / "traces.summary.json").read_text()) if (run_dir / "data" / "traces.summary.json").exists() else {}
    gen0_summary = json.loads((run_dir / "gen0" / "attribute_dataset_summary.json").read_text()) if (run_dir / "gen0" / "attribute_dataset_summary.json").exists() else {}
    train_metrics = json.loads((run_dir / "models" / "training_metrics.json").read_text()) if (run_dir / "models" / "training_metrics.json").exists() else {}
    eval_summary = json.loads((run_dir / "reports" / "eval_summary.json").read_text()) if (run_dir / "reports" / "eval_summary.json").exists() else {}
    scenario_summary = json.loads((run_dir / "reports" / "scenario_summary.json").read_text()) if (run_dir / "reports" / "scenario_summary.json").exists() else {}
    extra_files = []
    for rel, desc in [
        ("reports/scenario_results.csv", "scenario benchmark results"),
        ("reports/scenario_summary.json", "aggregated scenario benchmark"),
        ("reports/safety_tuning.json", "firewall threshold sweep recommendation"),
        ("reports/replays/scenario_replays.json", "serialized replay bundle"),
        ("reports/safety_dashboard.png", "safety benchmark dashboard"),
        ("reports/safety_case.md", "explainable safety case"),
        ("gen0/policy_specs.resolved.json", "resolved Generation Zero policy-attribute ranges"),
        ("gen0/policy_variants.csv", "sampled Generation Zero policy variants"),
        ("gen0/policy_variants.json", "sampled Generation Zero policy variants as JSON"),
        ("gen0/attribute_dataset_summary.json", "Generation Zero dataset summary"),
        ("gen0/attribute_dashboard.png", "Generation Zero attribute dashboard"),
        ("gen0/GENERATION_ZERO_CARD.md", "Generation Zero provenance card"),
        ("gen0/DOMAIN_RANDOMIZATION_CARD.md", "Generation Zero domain-randomization card"),
        ("selfplay/selfplay_matches.csv", "population self-play match table"),
        ("selfplay/population.csv", "self-play population ratings and failure modes"),
        ("selfplay/selfplay_summary.json", "self-play Elo/diversity/exploitability summary"),
        ("selfplay/selfplay_dashboard.png", "self-play dashboard"),
        ("selfplay/SELF_PLAY_CARD.md", "population self-play card"),
        ("selfplay/DOMAIN_RANDOMIZATION_CARD.md", "self-play domain-randomization card"),
        ("rl/ppo_policy.pt", "PPO-trained actor-critic policy"),
        ("rl/ppo_incumbent.pt", "best retained PPO deployment incumbent"),
        ("rl/incumbent_summary.json", "PPO deployment incumbent summary"),
        ("rl/INCUMBENT.md", "PPO deployment incumbent report"),
        ("rl/ppo_training_curve.csv", "PPO self-play learning curve"),
        ("rl/ppo_reward_terms.csv", "PPO decomposed reward term table"),
        ("rl/ppo_summary.json", "PPO self-play training summary"),
        ("rl/leaderboard.csv", "PPO league leaderboard"),
        ("rl/LEADERBOARD.md", "PPO league leaderboard report"),
        ("rl/league_matches.csv", "PPO league match table"),
        ("rl/payoff_matrix.csv", "PPO league empirical payoff matrix"),
        ("rl/meta_strategy.csv", "PPO league replicator meta-strategy"),
        ("rl/league_analysis.json", "PPO league exploitability analysis"),
        ("rl/LEAGUE_ANALYSIS.md", "PPO league exploitability report"),
        ("rl/RL_TRAINING_CARD.md", "PPO self-play training card"),
        ("robustness/robustness_results.csv", "PPO robustness ablation table"),
        ("robustness/robustness_summary.json", "PPO robustness ablation summary"),
        ("robustness/robustness_dashboard.png", "PPO robustness ablation dashboard"),
        ("robustness/ROBUSTNESS_REPORT.md", "PPO robustness ablation report"),
        ("replay/replay.json", "serialized PPO replay"),
        ("replay/replay_viewer.html", "offline PPO replay viewer"),
        ("backends/backend_scale_plan.json", "backend rollout-scale target mapping"),
        ("backends/BACKEND_SCALE_PLAN.md", "Isaac Lab and MuJoCo scale plan"),
        ("scaling/scaling_study.json", "self-improvement scaling summary"),
        ("scaling/scaling_dashboard.png", "self-improvement scaling dashboard"),
        ("scaling/LEARNING_CASE.md", "learning-over-time case"),
        ("MODEL_CARD.md", "model card"),
    ]:
        if (run_dir / rel).exists():
            extra_files.append(f"- `{rel}`: {desc}")
    extra_text = "\n".join(extra_files)
    text = f"""# GhostFighter Run Card

This run demonstrates the complete pipeline: attribute-driven Generation Zero data, conditional policy learning, autonomous match evaluation, safety-firewall ablation, dashboard generation, and demo rendering.

## Data

- Source: {data_summary.get('source', 'n/a')}
- Trace samples: {data_summary.get('samples', 'n/a')}
- Episodes: {data_summary.get('episodes', 'n/a')}
- Observation dimension: {data_summary.get('obs_dim', 'n/a')}
- Generation Zero variants: {data_summary.get('policy_variants', gen0_summary.get('policy_variants', 'n/a'))}
- Action entropy: {data_summary.get('action_entropy', gen0_summary.get('action_entropy', 'n/a'))}

## Training

- Validation action accuracy: {train_metrics.get('val_acc', 'n/a')}
- Best validation action accuracy: {train_metrics.get('best_val_acc', 'n/a')}
- Dataset samples: {train_metrics.get('dataset_samples', 'n/a')}

## Evaluation

The central ablation is `raw` versus `firewall`. The raw ghost executes its chosen skill token directly. The firewall ghost runs the same policy but blocks actions with high predicted fall, boundary, stamina, actuator, cooldown, or whiff risk.

```json
{json.dumps(eval_summary.get('by_mode', []), indent=2)}
```

## Safety Benchmark

```json
{json.dumps(scenario_summary.get('by_mode', []), indent=2)}
```

## Files

- `data/traces.npz`: logged Generation Zero policy rollouts
- `models/ghost_policy.pt`: trained conditional ghost policy
- `models/training_curve.csv`: epoch metrics
- `reports/match_results.csv`: per-match evaluation
- `reports/eval_summary.json`: aggregated evaluation
- `reports/dashboard.png`: visual summary
- `videos/ghostfighter_demo.gif`: rendered demonstration
{extra_text}
"""
    card_path.write_text(text, encoding="utf-8")
    return str(card_path)
