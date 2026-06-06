# Portfolio Pitch

## Project title

GhostFighter: converting robot-fight pilot traces into autonomous combat styles with a pre-controller safety firewall.

## One-sentence pitch

I built a complete simulator-to-autonomy pipeline that logs pilot fights, trains autonomous ghost fighters from those traces, stress-tests them in batch, and blocks high-risk actions before execution.

## Why it is relevant

A robot-combat company needs more than walking or punching demos. It needs a data engine. Simulator pilots generate fight traces. Those traces can become autonomous sparring partners, opponent styles, tournament bots, safety test cases, and policy-evaluation datasets.

GhostFighter demonstrates that full loop.

## What to show first

Open `reference_run/videos/ghostfighter_demo.gif`.

Then open `reference_run/reports/dashboard.png`. The reference stress ablation shows the raw ghost falling in 83% of hardware-stress matches, versus 21% for the same policy behind the firewall.

Then show the CLI:

```bash
python -m ghostfighter.cli all --out runs/review --episodes-per-style 20 --epochs 3 --eval-episodes 24 --max-steps 80
```

## What to say in an interview

The technical bet is that the industry’s simulator can become the training substrate for autonomous robot fighters. GhostFighter builds the autonomy layer above low-level controllers: logging, conditional style cloning, batch evaluation, replayable safety evidence, self-improvement studies, and pre-controller safety filtering.

The safety firewall is deliberately separate from the learned policy. The policy proposes intent. The firewall decides whether that intent is safe enough to execute in the current physical state.

## What makes it non-generic

Most portfolio projects show a robot walking, punching, or following an RL tutorial. This project shows how to turn a combat simulator into an autonomy product: style capture, ghost fighters, safety gates, metrics, and renderable review artifacts.
