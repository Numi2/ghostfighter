from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


VECTOR_BACKEND_PLAN: Dict[str, object] = {
    "local_backend": {
        "purpose": "CI, smoke tests, reproducible review artifacts, algorithm iteration",
        "expected_steps": "10^3 to 10^6",
        "status": "implemented",
    },
    "isaac_lab_vectorized_backend": {
        "purpose": "GPU-parallel adversarial curriculum rollouts and imitation/RL data generation",
        "expected_steps": "10^6 to 10^8+",
        "status": "interface target",
        "maps": [
            "policy roles",
            "domain randomization profile",
            "self-play match scheduler",
            "Elo/exploitability/diversity metrics",
        ],
    },
    "mujoco_validation_backend": {
        "purpose": "higher-fidelity validation of selected policies before hardware transfer",
        "expected_steps": "10^4 to 10^7",
        "status": "interface target",
        "maps": [
            "selected policy checkpoints",
            "domain randomization sweeps",
            "safety-firewall decisions",
            "failure-mode replay cases",
        ],
    },
}


def write_backend_scale_plan(out_dir: str | Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "backend_scale_plan.json").write_text(json.dumps(VECTOR_BACKEND_PLAN, indent=2), encoding="utf-8")
    text = f"""# Backend Scale Plan

GhostFighter now has the ingredients required for serious rollout scale: population self-play, domain-randomization profiles, replayable failure modes, and policy-condition metadata. The default backend remains self-contained for CI and review, but serious robot-learning claims require a vectorized simulator.

## Implemented Local Backend

- Use for smoke tests, reproducible demos, benchmark development, and fast policy iteration.
- Practical range: thousands to low millions of high-level simulation steps.
- Produces the same self-play, domain-randomization, safety, and replay artifacts expected from larger backends.

## Isaac Lab Target

- Use Isaac Lab for GPU-parallel rollout generation and adversarial curriculum training.
- Target range: millions to hundreds of millions of steps.
- Map GhostFighter policy roles, self-play scheduling, Elo/exploitability metrics, and domain-randomization profiles into vectorized environments.

## MuJoCo Target

- Use MuJoCo as a higher-fidelity validation backend for selected policies and safety-firewall decisions.
- Validate policies against contact-rich dynamics, actuator limits, latency, noisy state estimation, and replayed failure modes before hardware tests.

```json
{json.dumps(VECTOR_BACKEND_PLAN, indent=2)}
```
"""
    path = out_dir / "BACKEND_SCALE_PLAN.md"
    path.write_text(text, encoding="utf-8")
    return str(path)
