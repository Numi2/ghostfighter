# GhostFighter

GhostFighter is a small robot-combat autonomy lab: a fast humanoid fight simulator, configurable policy generation, PPO self-play, safety shielding, domain randomization, league analysis, robustness ablations, and replayable evidence.

It is deliberately not a full rigid-body humanoid simulator. It studies the autonomy layer above motor control: how policies are generated, trained, stress-tested, selected, and explained.

## Why It Stands Out

- **Generation Zero without scripts:** users define policy attributes; GhostFighter samples randomized policy variants and logs rollouts.
- **Real self-play loop:** PPO actor-critic training against role-based and historical opponents.
- **Vectorized local rollouts:** `train-rl --envs N` collects interleaved multi-env PPO data with trajectory-safe GAE.
- **Inspectable rewards:** PPO logs reward terms, KL, clip fraction, entropy, and explained variance.
- **League theory:** empirical payoff matrix, replicator-dynamics meta-strategy, best response, and exploitability.
- **Deployment guardrail:** `ppo_incumbent.pt` keeps the best retained checkpoint instead of blindly shipping the latest update.
- **Robotics realism hooks:** domain randomization, safety firewall, robustness ablations, counterfactual safety replay, and offline replay viewer.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make test
```

No MuJoCo, Isaac, ROS, or GPU is required for the local backend.

## Run The Lab

Full review run:

```bash
python -m ghostfighter.cli all --out runs/default \
  --episodes-per-style 80 --epochs 8 --eval-episodes 160 \
  --stress --benchmark --scale-study --self-play --rl \
  --robustness --replay-viewer --domain-randomization
```

Fast smoke:

```bash
make smoke
```

PPO self-play with vectorized local rollouts:

```bash
python -m ghostfighter.cli train-rl \
  --out runs/default/rl \
  --envs 8 \
  --updates 8 \
  --matches-per-update 64 \
  --max-steps 90
```

Inspect a trained PPO policy:

```bash
python -m ghostfighter.cli robustness \
  --policy runs/default/rl/ppo_policy.pt \
  --out runs/default/robustness

python -m ghostfighter.cli replay-viewer \
  --policy runs/default/rl/ppo_policy.pt \
  --out runs/default/replay
```

## Artifacts Worth Opening

```text
runs/default/
  rl/ppo_training_curve.csv      # PPO loss, KL, entropy, explained variance
  rl/ppo_reward_terms.csv        # decomposed reward terms
  rl/payoff_matrix.csv           # empirical league payoffs
  rl/meta_strategy.csv           # replicator-dynamics strategy
  rl/LEAGUE_ANALYSIS.md          # exploitability and best response
  rl/ppo_incumbent.pt            # best retained deployment checkpoint
  rl/INCUMBENT.md                # why that checkpoint was selected
  robustness/ROBUSTNESS_REPORT.md
  replay/replay_viewer.html
  reports/safety_case.md
  MODEL_CARD.md
  RUN_CARD.md
```

## Algorithmic Core

The simulator exposes high-level humanoid skill tokens:

```text
guard, step_forward, step_back, sidestep_left, sidestep_right,
circle_left, circle_right, jab, cross, hook, low_kick, push, recover
```

Generation Zero creates behavior priors from attributes like engagement drive, guard discipline, counter timing, lateral mobility, stamina discipline, boundary awareness, damage targeting, risk tolerance, and close-range pressure.

PPO self-play then learns from match rewards. Rollouts can be collected across multiple local environments; advantages are computed per trajectory, not across interleaved env steps. Each update is evaluated against a population, added to the historical opponent league, and compared against the deployment incumbent.

The league report is intentionally more than Elo: it estimates a meta-strategy from the payoff matrix and reports exploitability, which is the closest this compact project gets to “is this policy strategically robust?”

## Scale Path

The local backend is for iteration, CI, and review. The interfaces are shaped so the same policy roles, domain-randomization profile, PPO loop, and failure/replay artifacts can move to Isaac Lab for massive vectorized rollout generation and MuJoCo for higher-fidelity validation.
