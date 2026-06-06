# Portfolio Pitch

## Project title

GhostFighter: generating policy-conditioned robot-fight data and turning it into autonomous combat policies with a pre-controller safety firewall.

## One-sentence pitch

I built a complete simulator-to-autonomy pipeline that creates randomized Generation Zero rollouts from user-specified policy attributes, runs adversarial population self-play, trains autonomous ghost fighters from those traces, stress-tests them under domain randomization, and blocks high-risk actions before execution.

## Why it is relevant

A robot-combat company needs more than walking or punching demos. It needs a data engine. Simulator users and autonomy engineers should be able to define behavior attributes, generate randomized fight traces, run self-play curricula, and turn those traces into autonomous sparring partners, opponent policies, tournament bots, safety test cases, and policy-evaluation datasets.

GhostFighter demonstrates that full loop.

## What to show first

Open `reference_run/videos/ghostfighter_demo.gif`.

Then open `reference_run/reports/dashboard.png`. The reference stress ablation shows the raw ghost falling in 83% of hardware-stress matches, versus 21% for the same policy behind the firewall.

Then show the CLI:

```bash
python -m ghostfighter.cli all --out runs/review --episodes-per-style 20 --epochs 3 --eval-episodes 24 --max-steps 80 --benchmark
```

## What to say in an interview

The technical bet is that the industry’s simulator can become the training substrate for autonomous robot fighters. GhostFighter builds the autonomy layer above low-level controllers: configurable policy-data generation, population self-play, domain randomization, conditional policy learning, batch evaluation, replayable safety evidence, self-improvement studies, and pre-controller safety filtering.

The safety firewall is deliberately separate from the learned policy. The policy proposes intent. The firewall decides whether that intent is safe enough to execute in the current physical state.

## What makes it non-generic

Most portfolio projects show a robot walking, punching, or following an RL tutorial. This project shows how to turn a combat simulator into an autonomy product: policy archetype generation, self-play population ratings, domain-randomized stress, ghost fighters, safety gates, metrics, and renderable review artifacts. The archetypes are inspired by fighting-game language, but the actual controls are robotics-style policy conditions.
