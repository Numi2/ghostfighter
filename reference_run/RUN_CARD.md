# GhostFighter Run Card

This run demonstrates the complete pipeline: pilot trace generation, conditional behavior cloning, autonomous match evaluation, safety-firewall ablation, dashboard generation, and demo rendering.

## Data

- Trace samples: 72000
- Episodes: 240
- Observation dimension: 80

## Training

- Validation action accuracy: 0.3611111111111111
- Best validation action accuracy: 0.3611111111111111
- Dataset samples: 72000

## Evaluation

The central ablation is `raw` versus `firewall`. The raw ghost executes its chosen skill token directly. The firewall ghost runs the same policy but blocks actions with high predicted fall, boundary, stamina, actuator, cooldown, or whiff risk.

```json
[
  {
    "mode": "firewall",
    "matches": 24,
    "win_rate": 0.3333333333333333,
    "red_fall_rate": 0.0,
    "avg_red_falls": 0.0,
    "avg_health_margin": -1.8357730282253488,
    "avg_score_margin": -1.5336896948920133,
    "avg_unsafe_rate": 0.0067708333333333336,
    "avg_risk": 0.11124735135561807
  },
  {
    "mode": "firewall_stress",
    "matches": 24,
    "win_rate": 0.16666666666666666,
    "red_fall_rate": 0.20833333333333334,
    "avg_red_falls": 0.20833333333333334,
    "avg_health_margin": -1.5549895093204398,
    "avg_score_margin": -2.838322842653772,
    "avg_unsafe_rate": 0.1046875,
    "avg_risk": 0.3480689668787334
  },
  {
    "mode": "raw",
    "matches": 24,
    "win_rate": 0.625,
    "red_fall_rate": 0.041666666666666664,
    "avg_red_falls": 0.041666666666666664,
    "avg_health_margin": 7.778759299545662,
    "avg_score_margin": 9.220425966212327,
    "avg_unsafe_rate": 0.0,
    "avg_risk": 0.0
  },
  {
    "mode": "raw_stress",
    "matches": 24,
    "win_rate": 0.0,
    "red_fall_rate": 0.8333333333333334,
    "avg_red_falls": 2.2083333333333335,
    "avg_health_margin": -10.471341794627941,
    "avg_score_margin": -25.60884179462794,
    "avg_unsafe_rate": 0.0,
    "avg_risk": 0.0
  }
]
```

## Files

- `data/traces.npz`: logged pilot traces
- `models/ghost_policy.pt`: trained conditional ghost policy
- `models/training_curve.csv`: epoch metrics
- `reports/match_results.csv`: per-match evaluation
- `reports/eval_summary.json`: aggregated evaluation
- `reports/dashboard.png`: visual summary
- `videos/ghostfighter_demo.gif`: rendered demonstration
