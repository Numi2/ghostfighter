from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .config import TrainConfig
from .dataset import generate_trace_dataset
from .evaluate import evaluate_policy, run_safety_threshold_sweep
from .render import make_dashboard
from .train import train_behavior_cloning


def run_scale_study(
    out_dir: str | Path,
    episode_schedule: Iterable[int] = (8, 16, 32),
    epochs: int = 3,
    eval_episodes: int = 24,
    seed: int = 1201,
    max_steps: int = 90,
    batch_size: int = 1024,
    hidden: int = 128,
    verbose: bool = False,
) -> dict[str, object]:
    """Train/evaluate a data-scaling ladder that demonstrates improvement over time."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    schedule = [int(x) for x in episode_schedule]
    if not schedule:
        raise ValueError("episode_schedule must contain at least one positive integer")
    for generation, episodes_per_style in enumerate(schedule, start=1):
        if episodes_per_style <= 0:
            raise ValueError("episode schedule values must be positive")
        gen_dir = out_dir / f"generation_{generation:02d}_eps_{episodes_per_style}"
        data_path = gen_dir / "data" / "traces.npz"
        model_dir = gen_dir / "models"
        report_dir = gen_dir / "reports"
        if verbose:
            print(f"[scale {generation}/{len(schedule)}] data episodes_per_style={episodes_per_style}", flush=True)
        data_summary = generate_trace_dataset(
            data_path,
            episodes_per_style=episodes_per_style,
            seed=seed + generation * 101,
            max_steps=max_steps,
        )
        if verbose:
            print(f"[scale {generation}/{len(schedule)}] train samples={data_summary['samples']}", flush=True)
        train_result = train_behavior_cloning(
            data_path,
            model_dir,
            config=TrainConfig(epochs=epochs, batch_size=batch_size, seed=seed + generation * 101 + 1),
            hidden=hidden,
            verbose=verbose,
        )
        if verbose:
            print(f"[scale {generation}/{len(schedule)}] evaluate", flush=True)
        eval_result = evaluate_policy(
            train_result["model_path"],
            report_dir,
            episodes=eval_episodes,
            seed=seed + generation * 101 + 2,
            max_steps=max_steps,
            verbose=False,
            include_stress=True,
        )
        tuning = run_safety_threshold_sweep(
            train_result["model_path"],
            report_dir,
            episodes=max(8, eval_episodes // 2),
            seed=seed + generation * 101 + 3,
            max_steps=max_steps,
            suite="regression",
            verbose=False,
        )
        dashboard_path = make_dashboard(report_dir)
        mode_summary = {item["mode"]: item for item in eval_result["summary"]["by_mode"]}
        raw_stress = mode_summary.get("raw_stress", {})
        firewall_stress = mode_summary.get("firewall_stress", {})
        raw_fall = float(raw_stress.get("red_fall_rate", 0.0))
        firewall_fall = float(firewall_stress.get("red_fall_rate", 0.0))
        firewall_win = float(firewall_stress.get("win_rate", 0.0))
        val_acc = float(train_result["metrics"].get("val_acc", 0.0))
        # One scalar for trend plots. It rewards imitation quality and stress win rate,
        # while heavily penalizing falls in stress modes.
        research_score = float(val_acc + firewall_win - 0.75 * firewall_fall)
        rows.append(
            {
                "generation": generation,
                "episodes_per_style": int(episodes_per_style),
                "samples": int(data_summary["samples"]),
                "model_path": str(train_result["model_path"]),
                "dashboard": str(dashboard_path),
                "val_acc": val_acc,
                "best_val_acc": float(train_result["metrics"].get("best_val_acc", 0.0)),
                "raw_stress_win_rate": float(raw_stress.get("win_rate", 0.0)),
                "raw_stress_fall_rate": raw_fall,
                "firewall_stress_win_rate": firewall_win,
                "firewall_stress_fall_rate": firewall_fall,
                "stress_fall_reduction": raw_fall - firewall_fall,
                "recommended_threshold": tuning["summary"].get("recommended_threshold"),
                "research_score": research_score,
            }
        )
        _annotate_incumbents(rows)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "scaling_study.csv", index=False)
    summary = summarize_scale_study(rows)
    with open(out_dir / "scaling_study.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plot_path = make_scaling_dashboard(out_dir, df)
    summary["scaling_dashboard"] = plot_path
    with open(out_dir / "scaling_study.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_learning_case(out_dir / "LEARNING_CASE.md", summary)
    return {"summary": summary, "rows": rows}


def summarize_scale_study(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"generations": 0, "trend": "no data", "rows": []}
    first = rows[0]
    last = rows[-1]
    incumbent = max(rows, key=lambda item: float(item["research_score"]))
    score_delta = float(last["research_score"]) - float(first["research_score"])
    incumbent_delta = float(incumbent["research_score"]) - float(first["research_score"])
    sample_scale = float(last["samples"]) / max(1.0, float(first["samples"]))
    val_delta = float(last["val_acc"]) - float(first["val_acc"])
    fall_delta = float(first["firewall_stress_fall_rate"]) - float(last["firewall_stress_fall_rate"])
    return {
        "generations": len(rows),
        "sample_scale": sample_scale,
        "start": first,
        "final": last,
        "deployment_incumbent": incumbent,
        "deltas": {
            "research_score": score_delta,
            "incumbent_research_score": incumbent_delta,
            "val_acc": val_delta,
            "firewall_stress_fall_rate_reduction": fall_delta,
        },
        "trend": "improved" if incumbent_delta > 0 else "mixed",
        "rows": rows,
    }


def _annotate_incumbents(rows: list[dict[str, object]]) -> None:
    best_score = float("-inf")
    best_generation = 0
    for row in rows:
        score = float(row["research_score"])
        if score > best_score:
            best_score = score
            best_generation = int(row["generation"])
        row["incumbent_generation"] = best_generation
        row["incumbent_research_score"] = best_score


def make_scaling_dashboard(out_dir: str | Path, df: pd.DataFrame) -> str:
    out_dir = Path(out_dir)
    out_path = out_dir / "scaling_dashboard.png"
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    x = df["samples"]
    ax1.plot(x, df["val_acc"], marker="o")
    ax1.set_title("Imitation improves with data")
    ax1.set_xlabel("trace samples")
    ax1.set_ylabel("validation accuracy")

    ax2.plot(x, df["research_score"], marker="o")
    ax2.set_title("Research score over generations")
    ax2.set_xlabel("trace samples")

    ax3.plot(x, df["raw_stress_fall_rate"], marker="o", label="raw stress")
    ax3.plot(x, df["firewall_stress_fall_rate"], marker="o", label="firewall stress")
    ax3.set_title("Stress fall rate")
    ax3.set_xlabel("trace samples")
    ax3.set_ylim(0, 1)
    ax3.legend()

    ax4.bar(df["generation"].astype(str), df["stress_fall_reduction"])
    ax4.set_title("Safety layer fall reduction")
    ax4.set_xlabel("generation")
    fig.suptitle("GhostFighter self-improvement and scaling study", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def write_learning_case(path: str | Path, summary: dict[str, object]) -> str:
    path = Path(path)
    text = f"""# GhostFighter Learning Case

This artifact shows whether the robot-learning loop improves as the trace budget grows. Each generation regenerates pilot data at a larger scale, trains a fresh conditional ghost policy, evaluates raw/firewall behavior under stress, and tunes the safety threshold on deterministic regression scenarios.

## Summary

```json
{json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2)}
```

## Interpretation

- `val_acc` measures imitation quality from pilot traces.
- `firewall_stress_fall_rate` measures whether the learned policy remains usable under damaged/low-balance conditions.
- `research_score` combines imitation quality, stress win rate, and a strong fall penalty.
- `sample_scale` shows how much larger the final generation is than the first generation.
- `deployment_incumbent` is the best retained policy generation. This is the self-improvement guardrail: the platform can explore larger data budgets without regressing the policy selected for deployment.
"""
    path.write_text(text, encoding="utf-8")
    return str(path)
