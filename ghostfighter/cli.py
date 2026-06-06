from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import STYLE_NAMES, TrainConfig


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return ivalue


def positive_int_list(value: str) -> list[int]:
    try:
        values = [positive_int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a comma-separated list of positive integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("must include at least one positive integer")
    return values


def _optional_all_steps(args) -> list[str]:
    steps = ["data", "train", "evaluate", "scripted_baseline"]
    if args.benchmark:
        steps.append("benchmark")
    if args.self_play:
        steps.append("self_play")
    if args.rl:
        steps.append("rl")
    if args.robustness:
        steps.append("robustness")
    if args.replay_viewer:
        steps.append("replay_viewer")
    if args.scale_study:
        steps.append("scale_study")
    steps.extend(["render", "model_card", "run_card"])
    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ghostfighter",
        description="Autonomous robot-combat policy learning with safety-firewall evaluation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-data", help="Generate Generation Zero rollout traces.")
    p.add_argument("--out", default="runs/default/data/traces.npz")
    p.add_argument("--episodes-per-style", type=positive_int, default=80)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--max-steps", type=positive_int, default=180)
    p.add_argument("--source", choices=["attributes", "scripted"], default="attributes")
    p.add_argument("--policy-spec", default=None)
    p.add_argument("--variants-per-archetype", type=positive_int, default=8)
    p.add_argument("--domain-randomization", action="store_true")
    p.add_argument("--domain-intensity", type=float, default=0.65)

    p = sub.add_parser("forge-zero", help="Generate attribute-driven Generation Zero policy data and artifacts.")
    p.add_argument("--out", default="runs/default/data/traces.npz")
    p.add_argument("--episodes-per-style", type=positive_int, default=80)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--max-steps", type=positive_int, default=180)
    p.add_argument("--policy-spec", default=None)
    p.add_argument("--variants-per-archetype", type=positive_int, default=8)
    p.add_argument("--domain-randomization", action="store_true")
    p.add_argument("--domain-intensity", type=float, default=0.65)

    p = sub.add_parser("train", help="Train behavior-cloned ghost policy.")
    p.add_argument("--data", default="runs/default/data/traces.npz")
    p.add_argument("--out", default="runs/default/models")
    p.add_argument("--epochs", type=positive_int, default=8)
    p.add_argument("--batch-size", type=positive_int, default=2048)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--hidden", type=positive_int, default=192)

    p = sub.add_parser("evaluate", help="Evaluate raw vs firewall ghost policies.")
    p.add_argument("--model", default="runs/default/models/ghost_policy.pt")
    p.add_argument("--out", default="runs/default/reports")
    p.add_argument("--episodes", type=positive_int, default=160)
    p.add_argument("--seed", type=int, default=222)
    p.add_argument("--max-steps", type=positive_int, default=180)
    p.add_argument("--scripted-baseline", action="store_true")
    p.add_argument("--stress", action="store_true", help="Also run damaged/low-balance hardware-stress matches.")

    p = sub.add_parser("demo", help="Render an animated GIF demo.")
    p.add_argument("--model", default="runs/default/models/ghost_policy.pt")
    p.add_argument("--out", default="runs/default/videos/ghostfighter_demo.gif")
    p.add_argument("--style", choices=STYLE_NAMES, default="pressure")
    p.add_argument("--seed", type=int, default=909)
    p.add_argument("--max-steps", type=positive_int, default=110)

    p = sub.add_parser("dashboard", help="Create dashboard.png from report CSVs.")
    p.add_argument("--reports", default="runs/default/reports")
    p.add_argument("--out", default=None)

    p = sub.add_parser("benchmark", help="Run deterministic scenario benchmark suites.")
    p.add_argument("--model", default="runs/default/models/ghost_policy.pt")
    p.add_argument("--out", default="runs/default/reports")
    p.add_argument("--episodes", type=positive_int, default=40)
    p.add_argument("--seed", type=int, default=444)
    p.add_argument("--max-steps", type=positive_int, default=120)
    p.add_argument("--suite", choices=["standard", "stress", "adversarial", "regression", "all"], default="adversarial")

    p = sub.add_parser("tune-safety", help="Sweep firewall thresholds and recommend a safety setting.")
    p.add_argument("--model", default="runs/default/models/ghost_policy.pt")
    p.add_argument("--out", default="runs/default/reports")
    p.add_argument("--episodes", type=positive_int, default=20)
    p.add_argument("--seed", type=int, default=707)
    p.add_argument("--max-steps", type=positive_int, default=80)
    p.add_argument("--suite", choices=["standard", "stress", "adversarial", "regression", "all"], default="regression")

    p = sub.add_parser("scale-study", help="Train/evaluate growing data budgets to show self-improvement and scaling.")
    p.add_argument("--out", default="runs/default/scaling")
    p.add_argument("--episodes-schedule", type=positive_int_list, default=[8, 16, 32])
    p.add_argument("--epochs", type=positive_int, default=3)
    p.add_argument("--eval-episodes", type=positive_int, default=24)
    p.add_argument("--seed", type=int, default=1201)
    p.add_argument("--max-steps", type=positive_int, default=90)
    p.add_argument("--batch-size", type=positive_int, default=1024)
    p.add_argument("--hidden", type=positive_int, default=128)

    p = sub.add_parser("self-play", help="Run population-based adversarial self-play.")
    p.add_argument("--out", default="runs/default/selfplay")
    p.add_argument("--generations", type=positive_int, default=3)
    p.add_argument("--matches-per-pair", type=positive_int, default=2)
    p.add_argument("--variants-per-role", type=positive_int, default=2)
    p.add_argument("--seed", type=int, default=1601)
    p.add_argument("--max-steps", type=positive_int, default=90)
    p.add_argument("--domain-randomization", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--domain-intensity", type=float, default=0.55)

    p = sub.add_parser("train-rl", help="Train an actor-critic policy with PPO self-play.")
    p.add_argument("--out", default="runs/default/rl")
    p.add_argument("--updates", type=positive_int, default=4)
    p.add_argument("--matches-per-update", type=positive_int, default=8)
    p.add_argument("--max-steps", type=positive_int, default=80)
    p.add_argument("--epochs", type=positive_int, default=3)
    p.add_argument("--batch-size", type=positive_int, default=512)
    p.add_argument("--hidden", type=positive_int, default=128)
    p.add_argument("--seed", type=int, default=1801)
    p.add_argument("--domain-randomization", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--domain-intensity", type=float, default=0.45)

    p = sub.add_parser("robustness", help="Run robustness ablations for a PPO actor policy.")
    p.add_argument("--policy", default="runs/default/rl/ppo_policy.pt")
    p.add_argument("--out", default="runs/default/robustness")
    p.add_argument("--episodes", type=positive_int, default=12)
    p.add_argument("--seed", type=int, default=2401)
    p.add_argument("--max-steps", type=positive_int, default=80)

    p = sub.add_parser("replay-viewer", help="Write an offline HTML replay viewer for a PPO actor policy.")
    p.add_argument("--policy", default="runs/default/rl/ppo_policy.pt")
    p.add_argument("--out", default="runs/default/replay")
    p.add_argument("--seed", type=int, default=2601)
    p.add_argument("--max-steps", type=positive_int, default=100)
    p.add_argument("--domain-randomization", action=argparse.BooleanOptionalAction, default=True)

    p = sub.add_parser("scale-plan", help="Write the Isaac Lab/MuJoCo rollout-scale backend plan.")
    p.add_argument("--out", default="runs/default/backends")

    p = sub.add_parser("all", help="Run the complete pipeline.")
    p.add_argument("--out", default="runs/default")
    p.add_argument("--episodes-per-style", type=positive_int, default=80)
    p.add_argument("--epochs", type=positive_int, default=8)
    p.add_argument("--batch-size", type=positive_int, default=2048)
    p.add_argument("--eval-episodes", type=positive_int, default=160)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--max-steps", type=positive_int, default=180)
    p.add_argument("--demo-style", choices=STYLE_NAMES, default="pressure")
    p.add_argument("--gen0-source", choices=["attributes", "scripted"], default="attributes")
    p.add_argument("--policy-spec", default=None)
    p.add_argument("--variants-per-archetype", type=positive_int, default=8)
    p.add_argument("--domain-randomization", action="store_true")
    p.add_argument("--domain-intensity", type=float, default=0.65)
    p.add_argument("--stress", action="store_true", help="Include hardware-stress evaluation modes.")
    p.add_argument("--benchmark", action="store_true", help="Run scenario benchmark, safety dashboard, safety case, and model card.")
    p.add_argument("--scale-study", action="store_true", help="Run a data-scaling self-improvement study.")
    p.add_argument("--self-play", action="store_true", help="Run population self-play and write Elo/diversity/failure-mode artifacts.")
    p.add_argument("--rl", action="store_true", help="Run PPO self-play training and league leaderboard artifacts.")
    p.add_argument("--robustness", action="store_true", help="Run PPO robustness ablations after RL training.")
    p.add_argument("--replay-viewer", action="store_true", help="Write an offline replay viewer after RL training.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate-data":
        from .dataset import generate_trace_dataset

        summary = generate_trace_dataset(
            args.out,
            args.episodes_per_style,
            args.seed,
            max_steps=args.max_steps,
            source=args.source,
            policy_spec=args.policy_spec,
            variants_per_archetype=args.variants_per_archetype,
            domain_randomization=args.domain_randomization,
            domain_intensity=args.domain_intensity,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "forge-zero":
        from .dataset import generate_trace_dataset

        summary = generate_trace_dataset(
            args.out,
            args.episodes_per_style,
            args.seed,
            max_steps=args.max_steps,
            source="attributes",
            policy_spec=args.policy_spec,
            variants_per_archetype=args.variants_per_archetype,
            domain_randomization=args.domain_randomization,
            domain_intensity=args.domain_intensity,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "train":
        from .train import train_behavior_cloning

        result = train_behavior_cloning(
            args.data,
            args.out,
            config=TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed),
            hidden=args.hidden,
            verbose=True,
        )
        print(json.dumps({"model_path": result["model_path"], "metrics": result["metrics"]}, indent=2))
        return 0

    if args.command == "evaluate":
        from .evaluate import evaluate_policy, evaluate_scripted_baseline

        result = evaluate_policy(args.model, args.out, episodes=args.episodes, seed=args.seed, max_steps=args.max_steps, verbose=True, include_stress=args.stress)
        if args.scripted_baseline:
            result["scripted_baseline"] = evaluate_scripted_baseline(args.out, episodes=max(40, args.episodes // 2), seed=args.seed + 9, verbose=True)
        print(json.dumps(result["summary"], indent=2))
        return 0

    if args.command == "demo":
        from .render import make_demo_gif

        style_id = STYLE_NAMES.index(args.style)
        path = make_demo_gif(args.model, args.out, style_id=style_id, seed=args.seed, max_steps=args.max_steps)
        print(path)
        return 0

    if args.command == "dashboard":
        from .render import make_dashboard

        path = make_dashboard(args.reports, args.out)
        print(path)
        return 0

    if args.command == "benchmark":
        from .evaluate import run_scenario_suite
        from .render import make_safety_dashboard

        result = run_scenario_suite(args.model, args.out, episodes=args.episodes, seed=args.seed, max_steps=args.max_steps, suite=args.suite, verbose=True)
        dashboard = make_safety_dashboard(args.out)
        print(json.dumps({"summary": result["summary"], "safety_dashboard": dashboard}, indent=2))
        return 0

    if args.command == "tune-safety":
        from .evaluate import run_safety_threshold_sweep

        result = run_safety_threshold_sweep(args.model, args.out, episodes=args.episodes, seed=args.seed, max_steps=args.max_steps, suite=args.suite, verbose=True)
        print(json.dumps(result["summary"], indent=2))
        return 0

    if args.command == "scale-study":
        from .improve import run_scale_study

        result = run_scale_study(
            args.out,
            episode_schedule=args.episodes_schedule,
            epochs=args.epochs,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            hidden=args.hidden,
            verbose=True,
        )
        print(json.dumps(result["summary"], indent=2))
        return 0

    if args.command == "self-play":
        from .selfplay import run_population_self_play

        result = run_population_self_play(
            args.out,
            generations=args.generations,
            matches_per_pair=args.matches_per_pair,
            seed=args.seed,
            max_steps=args.max_steps,
            variants_per_role=args.variants_per_role,
            domain_randomization=args.domain_randomization,
            domain_intensity=args.domain_intensity,
            verbose=True,
        )
        print(json.dumps(result["summary"], indent=2))
        return 0

    if args.command == "train-rl":
        from .rl import PPOConfig, train_ppo_self_play

        result = train_ppo_self_play(
            args.out,
            config=PPOConfig(
                updates=args.updates,
                matches_per_update=args.matches_per_update,
                max_steps=args.max_steps,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden=args.hidden,
                seed=args.seed,
                domain_randomization=args.domain_randomization,
                domain_intensity=args.domain_intensity,
            ),
            verbose=True,
        )
        print(json.dumps(result["summary"], indent=2))
        return 0

    if args.command == "robustness":
        from .robustness import run_robustness_ablations

        result = run_robustness_ablations(args.policy, args.out, episodes=args.episodes, seed=args.seed, max_steps=args.max_steps)
        print(json.dumps(result["summary"], indent=2))
        return 0

    if args.command == "replay-viewer":
        from .replay import make_replay_viewer

        print(json.dumps(make_replay_viewer(args.policy, args.out, seed=args.seed, max_steps=args.max_steps, domain_randomization=args.domain_randomization), indent=2))
        return 0

    if args.command == "scale-plan":
        from .backends import write_backend_scale_plan

        print(write_backend_scale_plan(args.out))
        return 0

    if args.command == "all":
        from .dataset import generate_trace_dataset
        from .evaluate import evaluate_policy, evaluate_scripted_baseline
        from .backends import write_backend_scale_plan
        from .render import make_dashboard, make_demo_gif, make_safety_dashboard, write_model_card, write_run_card
        from .train import train_behavior_cloning

        if (args.robustness or args.replay_viewer) and not args.rl:
            parser.error("--robustness and --replay-viewer require --rl in the all command")
        run_dir = Path(args.out)
        all_steps = _optional_all_steps(args)
        step_idx = 1

        def announce(label: str) -> None:
            nonlocal step_idx
            print(f"[{step_idx}/{len(all_steps)}] {label}", flush=True)
            step_idx += 1

        data_path = run_dir / "data" / "traces.npz"
        model_dir = run_dir / "models"
        report_dir = run_dir / "reports"
        video_path = run_dir / "videos" / "ghostfighter_demo.gif"
        announce(f"generating Generation Zero traces ({args.gen0_source})")
        data_summary = generate_trace_dataset(
            data_path,
            args.episodes_per_style,
            args.seed,
            max_steps=args.max_steps,
            source=args.gen0_source,
            policy_spec=args.policy_spec,
            variants_per_archetype=args.variants_per_archetype,
            domain_randomization=args.domain_randomization,
            domain_intensity=args.domain_intensity,
        )
        print(json.dumps(data_summary, indent=2), flush=True)
        announce("training ghost policy")
        train_result = train_behavior_cloning(data_path, model_dir, config=TrainConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed + 1), verbose=True)
        print(json.dumps({"model_path": train_result["model_path"], "metrics": train_result["metrics"]}, indent=2), flush=True)
        announce("evaluating raw vs firewall")
        eval_result = evaluate_policy(train_result["model_path"], report_dir, episodes=args.eval_episodes, seed=args.seed + 2, max_steps=args.max_steps, verbose=True, include_stress=args.stress)
        print(json.dumps(eval_result["summary"], indent=2), flush=True)
        announce("evaluating scripted baseline")
        evaluate_scripted_baseline(report_dir, episodes=max(40, args.eval_episodes // 2), seed=args.seed + 3, verbose=True)
        if args.benchmark:
            from .evaluate import run_scenario_suite

            announce("running scenario benchmark")
            benchmark_result = run_scenario_suite(
                train_result["model_path"],
                report_dir,
                episodes=max(40, args.eval_episodes // 2),
                seed=args.seed + 5,
                max_steps=args.max_steps,
                suite="all",
                verbose=True,
            )
            print(json.dumps(benchmark_result["summary"], indent=2), flush=True)
        if args.self_play:
            from .selfplay import run_population_self_play

            announce("running population self-play")
            selfplay_result = run_population_self_play(
                run_dir / "selfplay",
                generations=3,
                matches_per_pair=2,
                seed=args.seed + 7,
                max_steps=max(60, args.max_steps // 2),
                variants_per_role=2,
                domain_randomization=True,
                domain_intensity=0.55,
                verbose=True,
            )
            print(json.dumps(selfplay_result["summary"], indent=2), flush=True)
        else:
            selfplay_result = None
        if args.rl:
            from .rl import PPOConfig, train_ppo_self_play

            announce("running PPO self-play training")
            rl_result = train_ppo_self_play(
                run_dir / "rl",
                config=PPOConfig(
                    updates=3,
                    matches_per_update=6,
                    max_steps=max(50, args.max_steps // 2),
                    epochs=max(2, args.epochs // 3),
                    batch_size=min(args.batch_size, 1024),
                    hidden=128,
                    seed=args.seed + 8,
                    domain_randomization=True,
                    domain_intensity=0.45,
                ),
                verbose=True,
            )
            print(json.dumps(rl_result["summary"], indent=2), flush=True)
            if args.robustness:
                from .robustness import run_robustness_ablations

                announce("running robustness ablations")
                robustness_result = run_robustness_ablations(
                    rl_result["model_path"],
                    run_dir / "robustness",
                    episodes=max(4, args.eval_episodes // 20),
                    seed=args.seed + 10,
                    max_steps=max(50, args.max_steps // 2),
                )
                print(json.dumps(robustness_result["summary"], indent=2), flush=True)
            if args.replay_viewer:
                from .replay import make_replay_viewer

                announce("writing replay viewer")
                replay_result = make_replay_viewer(
                    rl_result["model_path"],
                    run_dir / "replay",
                    seed=args.seed + 11,
                    max_steps=min(120, args.max_steps),
                    domain_randomization=True,
                )
                print(json.dumps(replay_result, indent=2), flush=True)
        else:
            rl_result = None
        if args.scale_study:
            from .improve import run_scale_study

            announce("running self-improvement scale study")
            scale_result = run_scale_study(
                run_dir / "scaling",
                episode_schedule=[
                    max(2, args.episodes_per_style // 10),
                    max(4, args.episodes_per_style // 5),
                    max(8, args.episodes_per_style // 2),
                ],
                epochs=max(2, args.epochs // 2),
                eval_episodes=max(16, args.eval_episodes // 4),
                seed=args.seed + 6,
                max_steps=max(60, args.max_steps // 2),
                batch_size=args.batch_size,
                hidden=128,
                verbose=True,
            )
            print(json.dumps(scale_result["summary"], indent=2), flush=True)
        else:
            scale_result = None
        announce("rendering dashboards and demo")
        dashboard_path = make_dashboard(report_dir)
        backend_plan_path = write_backend_scale_plan(run_dir / "backends")
        safety_dashboard_path = make_safety_dashboard(report_dir) if args.benchmark else None
        style_id = STYLE_NAMES.index(args.demo_style)
        gif_path = make_demo_gif(train_result["model_path"], video_path, style_id=style_id, seed=args.seed + 4, max_steps=min(120, args.max_steps))
        announce("writing model card")
        model_card_path = write_model_card(run_dir)
        announce("writing run card")
        card_path = write_run_card(run_dir)
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "dashboard": dashboard_path,
                    "safety_dashboard": safety_dashboard_path,
                    "backend_scale_plan": backend_plan_path,
                    "demo_gif": gif_path,
                    "model_card": model_card_path,
                    "run_card": card_path,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
