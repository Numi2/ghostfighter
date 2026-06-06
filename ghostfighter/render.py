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
from .evaluate import run_match
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
    width: int = 720,
    height: int = 520,
) -> Image.Image:
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    font_title = _font(18)
    font = _font(12)
    small = _font(10)
    margin = 34
    panel_top = 58
    panel_h = height - panel_top - 44
    panel_w = width - 2 * margin
    cx = margin + panel_w / 2
    cy = panel_top + panel_h / 2
    scale = min(panel_w, panel_h) / (2 * env.config.arena_radius * 1.08)

    d.text((margin, 16), title, fill=INK, font=font_title)
    if subtitle:
        d.text((margin, 39), subtitle, fill=(80, 80, 80), font=font)

    r = env.config.arena_radius * scale
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(80, 80, 80), width=3)
    d.ellipse((cx - r * 0.72, cy - r * 0.72, cx + r * 0.72, cy + r * 0.72), outline=(220, 220, 220), width=1)
    d.line((cx - r, cy, cx + r, cy), fill=(225, 225, 225), width=1)
    d.line((cx, cy - r, cx, cy + r), fill=(225, 225, 225), width=1)

    def xy(f):
        return cx + f.x * scale, cy - f.y * scale

    def draw_fighter(f, color, label):
        x, y = xy(f)
        body_r = 14 if not f.fallen else 17
        fill = tuple(int(c * (0.55 if f.fallen else 1.0)) for c in color)
        d.ellipse((x - body_r, y - body_r, x + body_r, y + body_r), fill=fill, outline=INK, width=2)
        hx = x + math.cos(f.theta) * 24
        hy = y - math.sin(f.theta) * 24
        d.line((x, y, hx, hy), fill=INK, width=3)
        # Guard arc/arms.
        if f.guard > 0.35:
            d.arc((x - 24, y - 24, x + 24, y + 24), start=205, end=335, fill=GREEN, width=3)
        if f.last_action in [ACTION_NAMES.index("jab"), ACTION_NAMES.index("cross"), ACTION_NAMES.index("hook"), ACTION_NAMES.index("push")]:
            d.line((x, y, hx + math.cos(f.theta) * 13, hy - math.sin(f.theta) * 13), fill=ORANGE, width=4)
        if f.last_action == ACTION_NAMES.index("low_kick"):
            d.line((x, y, hx + math.cos(f.theta) * 8, hy - math.sin(f.theta) * 8), fill=ORANGE, width=6)
        d.text((x - 12, y + 18), label, fill=INK, font=small)
        if f.last_override:
            d.rectangle((x - 28, y - 34, x + 28, y - 22), fill=(255, 234, 188), outline=ORANGE)
            d.text((x - 25, y - 35), "SAFE", fill=INK, font=small)

    draw_fighter(env.red, RED, "R")
    draw_fighter(env.blue, BLUE, "B")

    def bars(x, y, f, side_name, color):
        d.text((x, y - 15), side_name, fill=INK, font=font)
        for i, (name, val, maxv, bar_color) in enumerate(
            [
                ("HP", f.health, 100.0, color),
                ("BAL", f.balance, 1.0, GREEN),
                ("STA", f.stamina, 1.0, ORANGE),
            ]
        ):
            yy = y + i * 13
            d.text((x, yy), name, fill=INK, font=small)
            d.rectangle((x + 30, yy + 2, x + 130, yy + 9), outline=GRAY)
            w = max(0, min(100, val / maxv * 100))
            d.rectangle((x + 30, yy + 2, x + 30 + w, yy + 9), fill=bar_color)
        d.text((x, y + 42), f"score {f.score:0.1f}  falls {f.falls}", fill=INK, font=small)
        act = ACTION_NAMES[int(f.last_action)]
        if f.last_override:
            prop = ACTION_NAMES[int(f.last_proposed_action)]
            act = f"{prop} → {act}"
        d.text((x, y + 56), act[:22], fill=INK, font=small)
        if f.last_risk > 0:
            d.text((x, y + 70), f"risk {f.last_risk:0.2f}", fill=INK, font=small)

    bars(34, height - 102, env.red, "RED ghost", RED)
    bars(width - 180, height - 102, env.blue, "BLUE opponent", BLUE)
    event_text = " | ".join(e.text for e in env.last_events[-2:])
    if event_text:
        d.rectangle((margin, panel_top + panel_h + 8, width - margin, panel_top + panel_h + 30), fill=(255, 255, 255), outline=(220, 220, 220))
        d.text((margin + 7, panel_top + panel_h + 12), event_text[:95], fill=INK, font=small)
    d.text((width - 130, 18), f"step {env.step_count}/{env.config.max_steps}", fill=(80, 80, 80), font=font)
    return img


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
    # Segment 1: scripted pilot reference.
    scripted = ScriptedPilot(style_name, seed=seed + 1)
    opponent = EnsembleOpponent(seed=seed + 2)
    res, trace = run_match(scripted, opponent, seed=seed + 3, style_name=style_name, mode="scripted_reference", max_steps=max_steps, collect_trace=True)
    segments.append(("1. Pilot trace style", "scripted fighter behavior to clone", trace))
    # Segment 2: raw ghost.
    raw = NeuralGhostPolicy(model, style_id=style_id, deterministic=True)
    opponent = EnsembleOpponent(seed=seed + 20)
    res, trace = run_match(raw, opponent, seed=seed + 21, style_name=style_name, mode="ghost_raw", max_steps=max_steps, collect_trace=True)
    segments.append(("2. Autonomous ghost", "policy cloned from pilot traces, no safety gate", trace))
    # Segment 3: safe ghost.
    safe = NeuralGhostPolicy(model, style_id=style_id, deterministic=True)
    opponent = EnsembleOpponent(seed=seed + 20)
    firewall = CombatSafetyFirewall(threshold=0.62)
    res, trace = run_match(safe, opponent, seed=seed + 21, style_name=style_name, mode="ghost_firewall", firewall=firewall, max_steps=max_steps, collect_trace=True)
    segments.append(("3. Ghost + combat safety firewall", "unsafe actions are replaced before execution", trace))

    for title, subtitle, trace in segments:
        if not trace:
            continue
        sample_every = max(1, len(trace) // 35)
        for i, item in enumerate(trace):
            if i % sample_every != 0 and i != len(trace) - 1:
                continue
            env = item["env"]
            frames.append(render_env_frame(env, title, subtitle))
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


def write_run_card(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    card_path = run_dir / "RUN_CARD.md"
    data_summary = json.loads((run_dir / "data" / "traces.summary.json").read_text()) if (run_dir / "data" / "traces.summary.json").exists() else {}
    train_metrics = json.loads((run_dir / "models" / "training_metrics.json").read_text()) if (run_dir / "models" / "training_metrics.json").exists() else {}
    eval_summary = json.loads((run_dir / "reports" / "eval_summary.json").read_text()) if (run_dir / "reports" / "eval_summary.json").exists() else {}
    text = f"""# GhostFighter Run Card

This run demonstrates the complete pipeline: pilot trace generation, conditional behavior cloning, autonomous match evaluation, safety-firewall ablation, dashboard generation, and demo rendering.

## Data

- Trace samples: {data_summary.get('samples', 'n/a')}
- Episodes: {data_summary.get('episodes', 'n/a')}
- Observation dimension: {data_summary.get('obs_dim', 'n/a')}

## Training

- Validation action accuracy: {train_metrics.get('val_acc', 'n/a')}
- Best validation action accuracy: {train_metrics.get('best_val_acc', 'n/a')}
- Dataset samples: {train_metrics.get('dataset_samples', 'n/a')}

## Evaluation

The central ablation is `raw` versus `firewall`. The raw ghost executes its chosen skill token directly. The firewall ghost runs the same policy but blocks actions with high predicted fall, boundary, stamina, actuator, cooldown, or whiff risk.

```json
{json.dumps(eval_summary.get('by_mode', []), indent=2)}
```

## Files

- `data/traces.npz`: logged pilot traces
- `models/ghost_policy.pt`: trained conditional ghost policy
- `models/training_curve.csv`: epoch metrics
- `reports/match_results.csv`: per-match evaluation
- `reports/eval_summary.json`: aggregated evaluation
- `reports/dashboard.png`: visual summary
- `videos/ghostfighter_demo.gif`: rendered demonstration
"""
    card_path.write_text(text, encoding="utf-8")
    return str(card_path)
