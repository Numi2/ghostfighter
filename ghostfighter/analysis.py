from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def _boundary_clearance(env, fighter) -> float:
    radius = float(np.linalg.norm(fighter.pos()))
    return max(0.0, env.config.arena_radius - radius) / env.config.arena_radius


def analyze_counterfactual_overrides(
    trace: Iterable[dict],
    scenario: str,
    style: str,
    match_seed: int,
) -> list[dict]:
    """Replay overridden steps with the rejected action for a cheap safety delta."""
    rows: list[dict] = []
    for item in trace:
        decision = item.get("decision") or {}
        if not decision.get("overridden"):
            continue
        safe_env = item.get("env")
        pre_env = item.get("pre_env")
        if safe_env is None or pre_env is None:
            continue
        proposed_action = int(item["proposed_action"])
        recorded_blue_action = int(item["action_blue"])
        try:
            raw_next, _, _, _, _, _ = pre_env.step(proposed_action, recorded_blue_action)
        except RuntimeError:
            continue
        raw_red = pre_env.red
        safe_red = safe_env.red
        raw_clearance = _boundary_clearance(pre_env, raw_red)
        safe_clearance = _boundary_clearance(safe_env, safe_red)
        rows.append(
            {
                "scenario": scenario,
                "style": style,
                "match_seed": int(match_seed),
                "step": int(item["step"]),
                "proposed_action": decision.get("proposed_action", ""),
                "replacement_action": decision.get("action", ""),
                "reason": decision.get("reason", ""),
                "risk": float(decision.get("risk", 0.0)),
                "avoided_fall": int(raw_red.falls > safe_red.falls),
                "avoided_boundary_loss": int(raw_clearance + 1e-6 < safe_clearance),
                "avoided_damage": float(max(0.0, safe_red.health - raw_red.health)),
                "balance_saved": float(max(0.0, safe_red.balance - raw_red.balance)),
                "score_saved": float(max(0.0, safe_red.score - raw_red.score)),
                "raw_balance": float(raw_red.balance),
                "safe_balance": float(safe_red.balance),
                "raw_boundary_clearance": float(raw_clearance),
                "safe_boundary_clearance": float(safe_clearance),
            }
        )
    return rows


def summarize_counterfactuals(rows: list[dict]) -> Dict[str, object]:
    if not rows:
        return {
            "overrides_analyzed": 0,
            "avoided_fall_rate": 0.0,
            "avoided_boundary_loss_rate": 0.0,
            "avg_avoided_damage": 0.0,
            "avg_balance_saved": 0.0,
            "top_reasons": {},
            "examples": [],
        }
    reason_counts: Dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "overrides_analyzed": int(len(rows)),
        "avoided_fall_rate": float(np.mean([r["avoided_fall"] for r in rows])),
        "avoided_boundary_loss_rate": float(np.mean([r["avoided_boundary_loss"] for r in rows])),
        "avg_avoided_damage": float(np.mean([r["avoided_damage"] for r in rows])),
        "avg_balance_saved": float(np.mean([r["balance_saved"] for r in rows])),
        "top_reasons": dict(sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)[:8]),
        "examples": sorted(rows, key=lambda r: (r["avoided_fall"], r["balance_saved"], r["risk"]), reverse=True)[:5],
    }


def serialize_replay_trace(
    trace: Iterable[dict],
    scenario: str,
    style: str,
    mode: str,
    match_seed: int,
    max_steps: int = 160,
) -> Dict[str, object]:
    steps = []
    for item in list(trace)[:max_steps]:
        decision = item.get("decision") or {}
        steps.append(
            {
                "step": int(item.get("step", 0)),
                "proposed_action": str(decision.get("proposed_action", "")),
                "executed_action": str(decision.get("action", "")),
                "blue_action": _action_name(item.get("action_blue")),
                "risk": float(decision.get("risk", 0.0)),
                "overridden": bool(decision.get("overridden", False)),
                "reason": str(decision.get("reason", "")),
                "events": item.get("events", []),
                "red": item.get("red", {}),
                "blue": item.get("blue", {}),
            }
        )
    return {
        "scenario": scenario,
        "style": style,
        "mode": mode,
        "match_seed": int(match_seed),
        "steps": steps,
    }


def write_replay_bundle(path: str | Path, replays: list[Dict[str, object]]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"replays": replays}, indent=2), encoding="utf-8")
    return str(path)


def _action_name(action) -> str:
    if action is None:
        return ""
    from .config import ACTION_NAMES

    return ACTION_NAMES[int(action)]


def write_safety_case(path: str | Path, summary: Dict[str, object], counterfactual_rows: list[dict]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_mode = summary.get("by_mode", [])
    reasons = summary.get("firewall_reason_counts", {})
    counterfactuals = summary.get("counterfactuals", {})
    tuning = summary.get("safety_tuning", {})
    text = f"""# GhostFighter Safety Case

This report summarizes the benchmark evidence for the pre-controller combat safety firewall. The firewall does not replace the learned policy; it blocks high-risk skill tokens before execution and substitutes recovery, guard, or escape actions.

## Benchmark Summary

```json
{json.dumps(by_mode, indent=2)}
```

## What The Firewall Blocked

```json
{json.dumps(reasons, indent=2)}
```

## Counterfactual Replay

For each analyzed override, GhostFighter replays the same pre-step simulator state with the rejected action and compares the immediate outcome against the firewall replacement.

```json
{json.dumps(counterfactuals, indent=2)}
```

## Self-Improvement Loop

GhostFighter can sweep firewall thresholds on deterministic benchmark scenarios and recommend the best setting for the current policy.

```json
{json.dumps(tuning, indent=2)}
```

## Limits

- The replay is a one-step local counterfactual, not a full proof of future match outcome.
- The simulator is intentionally high level and should be treated as a safety architecture testbed, not a dynamics certificate for physical hardware.
- Firewall thresholds are fixed heuristics in this prototype; production use would calibrate them against robot logs and hardware limits.
"""
    path.write_text(text, encoding="utf-8")
    return str(path)
