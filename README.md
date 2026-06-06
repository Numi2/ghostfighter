# GhostFighter

GhostFighter is a complete, self-contained prototype for autonomous humanoid robot combat development.

teh pipeline:

1. Generate fight logs from distinct pilot styles.
2. Train a conditional autonomous ghost fighter from those traces.
3. Evaluate the same policy with and without a pre-controller safety firewall.
4. Render a demo fight and produce an evaluation dashboard.
5. Package all logs, metrics, models, and reports so the work can be reviewed like an internal engineering artifact.

goal is to make the sim fights become a data flywheel: pilot traces become autonomous styles, autonomous styles are stress-tested in batch, and risky actions are blocked before a real robot ever receives them.

## What is implemented

- High-level humanoid combat simulator with ring boundary, stamina, guard, balance, actuator damage, cooldowns, knockdowns, and scoring.
- Four pilot styles: `pressure`, `counter`, `evasive`, and `bully`.
- Dataset generator that logs observations, actions, rewards, style ids, episode ids, and fighter ids.
- Conditional PyTorch behavior-cloning policy that can act as different ghost styles.
- Combat safety firewall that estimates risk from balance, stamina, boundary pressure, actuator damage, cooldown state, momentum, incoming contact, and likely whiffs.
- Raw-vs-firewall evaluation harness, including optional hardware-stress matches with actuator damage, low balance, perturbations, and boundary pressure.
- Scripted baseline evaluation.
- Dashboard and GIF renderer.
- CLI and test suite.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project does not require MuJoCo, Isaac, ROS, or a GPU. It is intentionally self-contained so the full pipeline can run without simulator setup delays.

## Run the complete pipeline

```bash
python -m ghostfighter.cli all --out runs/default --episodes-per-style 80 --epochs 8 --eval-episodes 160 --stress --benchmark --scale-study
```

This creates:

```text
runs/default/
  data/traces.npz
  data/traces.summary.json
  models/ghost_policy.pt
  models/training_curve.csv
  models/training_metrics.json
  reports/match_results.csv
  reports/eval_summary.json
  reports/scripted_baseline.csv
  reports/scripted_baseline_summary.json
  reports/dashboard.png
  reports/scenario_results.csv
  reports/scenario_summary.json
  reports/safety_tuning.json
  reports/replays/scenario_replays.json
  reports/safety_dashboard.png
  reports/safety_case.md
  videos/ghostfighter_demo.gif
  MODEL_CARD.md
  scaling/scaling_study.csv
  scaling/scaling_study.json
  scaling/scaling_dashboard.png
  scaling/LEARNING_CASE.md
  RUN_CARD.md
```

For a fast end-to-end smoke run:

```bash
make smoke
```

## Individual commands

Generate traces:

```bash
python -m ghostfighter.cli generate-data --out runs/default/data/traces.npz --episodes-per-style 80
```

Train the ghost policy:

```bash
python -m ghostfighter.cli train --data runs/default/data/traces.npz --out runs/default/models --epochs 8
```

Evaluate raw policy versus safety-firewall policy:

```bash
python -m ghostfighter.cli evaluate --model runs/default/models/ghost_policy.pt --out runs/default/reports --episodes 160 --scripted-baseline --stress
```

Create the dashboard:

```bash
python -m ghostfighter.cli dashboard --reports runs/default/reports
```

Run deterministic benchmark scenarios:

```bash
python -m ghostfighter.cli benchmark --model runs/default/models/ghost_policy.pt --out runs/default/reports --suite all --episodes 80
```

Tune the safety firewall threshold on deterministic scenarios:

```bash
python -m ghostfighter.cli tune-safety --model runs/default/models/ghost_policy.pt --out runs/default/reports --suite regression --episodes 20
```

Run the self-improvement scaling ladder:

```bash
python -m ghostfighter.cli scale-study --out runs/default/scaling --episodes-schedule 8,16,32 --epochs 3 --eval-episodes 24
```

Render a demo GIF:

```bash
python -m ghostfighter.cli demo --model runs/default/models/ghost_policy.pt --out runs/default/videos/ghostfighter_demo.gif --style pressure
```

Run tests:

```bash
pytest -q
```

## Architecture

The simulator uses high-level humanoid skill tokens rather than raw joint torques:

```text
guard, step_forward, step_back, sidestep_left, sidestep_right,
circle_left, circle_right, jab, cross, hook, low_kick, push, recover
```

This is deliberate. A one-week project should not pretend to solve full humanoid motor control. It should prove the data, autonomy, evaluation, and safety architecture. The same structure could later sit above a lower-level MuJoCo, Isaac, Unitree, or real-robot controller.

The policy is conditional on style. A single network receives the current observation and a style id, then predicts the next high-level combat action. This creates a practical path from human/sim pilot traces to autonomous ghost fighters.

The firewall is a pre-controller gate. It does not replace the policy. It filters the policy’s proposed action and replaces unsafe commands with recover, guard, step, or escape actions when risk is high.

## Why this would matter to a robot-combat company

A robot-combat company will not only need better walking or better punching. It will need a way to convert simulator activity into autonomous fighters, evaluate policies over thousands of matches, protect hardware from unstable learned behavior, and explain why a policy is safe enough to test.

GhostFighter demonstrates that operating model end to end.

## What makes this credible

- The benchmark suite includes normal matches, hardware-stress matches, and adversarial setups such as boundary traps, low-stamina rushes, damaged-leg pursuit, unstable recovery, and close-range brawls.
- The safety firewall is evaluated as an ablation, so reviewers can compare raw policy behavior against the same policy with pre-controller shielding.
- Counterfactual replay analyzes overridden actions from the same simulator state and reports avoided falls, boundary losses, damage, and balance loss.
- Serialized replay bundles capture representative benchmark fights step by step for inspection without rerunning the simulator.
- The safety tuning loop sweeps firewall thresholds and recommends the best setting for the current policy under a fall-averse benchmark objective.
- The scaling ladder trains multiple generations with growing trace budgets and reports whether imitation accuracy, stress behavior, and the combined research score improve as data increases.
- Each full run writes a model card, run card, dashboard, safety dashboard, and safety case so results are inspectable without reading code first.

## Limits

This is not a physically exact humanoid dynamics simulator. It is a high-level combat autonomy testbed. It models the operational constraints that matter for the portfolio signal: style data, policy cloning, batch evaluation, action safety, knockdown risk, boundary pressure, damage-aware behavior, and reproducible reporting.
