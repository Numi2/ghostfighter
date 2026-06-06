# Technical Design

## Objective

GhostFighter demonstrates a deployable autonomy pipeline for robot combat:

- generate randomized Generation Zero policy rollouts from user-specified behavior attributes;
- evaluate policies over many matches;
- block unsafe high-level actions before low-level controllers execute them;
- produce reviewable metrics and visual artifacts.

The project is intentionally self-contained. It avoids external robotics simulator setup so that a reviewer can run the full system immediately.

## Simulator abstraction

The simulator is a high-level humanoid combat environment, not a raw torque simulator. Each fighter has position, velocity, facing direction, stamina, balance, guard, cooldown, health, component damage, knockdown state, and score.

The action space is a set of skill tokens:

```text
guard, step_forward, step_back, sidestep_left, sidestep_right,
circle_left, circle_right, jab, cross, hook, low_kick, push, recover
```

Each attack has range, cone, damage, stamina cost, cooldown, self-balance cost, impact, guard-break value, and lunge. This gives the policy enough tactical structure to produce meaningful combat behavior while keeping training fast.

## Observation design

The observation is egocentric. It includes:

- own normalized position, local velocity, heading, health, stamina, balance, guard, cooldown, knockdown status, recent contact flags;
- own actuator/core damage;
- own previous action as one-hot;
- opponent relative position and relative velocity in the fighter’s local frame;
- opponent health, stamina, balance, guard, cooldown, knockdown status, contact flags, and damage;
- opponent previous action;
- local boundary clearance features;
- normalized match time.

This is enough for conditional policy learning, tactical response, boundary awareness, and safety filtering.

## Policy

The policy is a conditional behavior-cloning network. It receives the observation and a policy-condition id. That condition id is embedded and concatenated to the observation before a multilayer MLP predicts the next action.

This makes one model behave as multiple autonomous fighters: pressure, counter, evasive, and bully. Those labels are fighting-genre shorthand, but the implementation treats them as policy archetypes backed by explicit behavior attributes such as engagement drive, guard discipline, counter timing, lateral mobility, stamina discipline, boundary awareness, damage targeting, risk tolerance, and close-range pressure.

## Generation Zero

Generation Zero is not hardcoded to one set of scripts. A user defines policy-attribute ranges in `configs/gen0_policies.json`; the generator samples concrete variants from those ranges, pairs them against varied opponents, and logs both fighters as training examples.

The dataset preserves provenance fields alongside observations and actions: policy-condition ids, policy variant ids, source ids, fighter ids, episode ids, rewards, and the sampled attribute vectors. The resulting artifacts under `gen0/` make the starting corpus inspectable before training begins.

## Safety firewall

The firewall sits between the learned policy and the controller. It estimates risk for the proposed action using:

- low balance;
- fragile balance;
- low stamina;
- boundary pressure;
- leg damage and mobility risk;
- arm damage and strike risk;
- high momentum strike risk;
- angular instability;
- cooldown-forced whiffs;
- incoming contact without guard;
- likely whiff due to range or aim;
- stamina exhaustion;
- strike breaking balance.

When risk exceeds the threshold, it replaces the action with recover, guard, step-forward, or sidestep depending on the reason.

## Evaluation

The evaluation harness compares:

- raw ghost policy;
- the same ghost policy with the safety firewall;
- raw ghost under hardware-stress conditions;
- firewall ghost under the same hardware-stress conditions;
- scripted baseline fighters for comparison.

Metrics include win rate, draw rate, fall rate, health margin, score margin, unsafe rejection rate, average risk, attack rate, guard rate, and action entropy. The stress mode injects partial actuator damage, low balance, boundary pressure, and random perturbations to test whether the same learned policy becomes safer when placed behind the firewall.

## Review signal

The project shows practical judgment by separating autonomy research from low-level actuation. It builds the layer a robot-combat company needs to scale learning from simulator users and behavior specifications: configurable policy-data generation, conditional policy learning, safety gating, batch policy evaluation, replayable evidence, self-improvement studies, and reproducible reporting.
