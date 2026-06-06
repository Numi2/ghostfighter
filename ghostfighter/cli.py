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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ghostfighter",
        description="Autonomous robot-combat style cloning with safety-firewall evaluation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-data", help="Generate scripted pilot traces.")
    p.add_argument("--out", default="runs/default/data/traces.npz")
    p.add_argument("--episodes-per-style", type=positive_int, default=80)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--max-steps", type=positive_int, default=180)

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

    p = sub.add_parser("all", help="Run the complete pipeline.")
    p.add_argument("--out", default="runs/default")
    p.add_argument("--episodes-per-style", type=positive_int, default=80)
    p.add_argument("--epochs", type=positive_int, default=8)
    p.add_argument("--batch-size", type=positive_int, default=2048)
    p.add_argument("--eval-episodes", type=positive_int, default=160)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--max-steps", type=positive_int, default=180)
    p.add_argument("--demo-style", choices=STYLE_NAMES, default="pressure")
    p.add_argument("--stress", action="store_true", help="Include hardware-stress evaluation modes.")
    p.add_argument("--benchmark", action="store_true", help="Run scenario benchmark, safety dashboard, safety case, and model card.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate-data":
        from .dataset import generate_trace_dataset

        summary = generate_trace_dataset(args.out, args.episodes_per_style, args.seed, max_steps=args.max_steps)
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

    if args.command == "all":
        from .dataset import generate_trace_dataset
        from .evaluate import evaluate_policy, evaluate_scripted_baseline
        from .render import make_dashboard, make_demo_gif, make_safety_dashboard, write_model_card, write_run_card
        from .train import train_behavior_cloning

        run_dir = Path(args.out)
        data_path = run_dir / "data" / "traces.npz"
        model_dir = run_dir / "models"
        report_dir = run_dir / "reports"
        video_path = run_dir / "videos" / "ghostfighter_demo.gif"
        print("[1/6] generating pilot traces", flush=True)
        data_summary = generate_trace_dataset(data_path, args.episodes_per_style, args.seed, max_steps=args.max_steps)
        print(json.dumps(data_summary, indent=2), flush=True)
        print("[2/6] training ghost policy", flush=True)
        train_result = train_behavior_cloning(data_path, model_dir, config=TrainConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed + 1), verbose=True)
        print(json.dumps({"model_path": train_result["model_path"], "metrics": train_result["metrics"]}, indent=2), flush=True)
        print("[3/6] evaluating raw vs firewall", flush=True)
        eval_result = evaluate_policy(train_result["model_path"], report_dir, episodes=args.eval_episodes, seed=args.seed + 2, max_steps=args.max_steps, verbose=True, include_stress=args.stress)
        print(json.dumps(eval_result["summary"], indent=2), flush=True)
        print("[4/6] evaluating scripted baseline", flush=True)
        evaluate_scripted_baseline(report_dir, episodes=max(40, args.eval_episodes // 2), seed=args.seed + 3, verbose=True)
        if args.benchmark:
            from .evaluate import run_scenario_suite

            print("[5/8] running scenario benchmark", flush=True)
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
            print("[6/8] rendering dashboards and demo", flush=True)
        else:
            print("[5/6] rendering dashboard and demo", flush=True)
        dashboard_path = make_dashboard(report_dir)
        safety_dashboard_path = make_safety_dashboard(report_dir) if args.benchmark else None
        style_id = STYLE_NAMES.index(args.demo_style)
        gif_path = make_demo_gif(train_result["model_path"], video_path, style_id=style_id, seed=args.seed + 4, max_steps=min(120, args.max_steps))
        if args.benchmark:
            print("[7/8] writing model card", flush=True)
            model_card_path = write_model_card(run_dir)
            print("[8/8] writing run card", flush=True)
        else:
            model_card_path = None
            print("[6/6] writing run card", flush=True)
        card_path = write_run_card(run_dir)
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "dashboard": dashboard_path,
                    "safety_dashboard": safety_dashboard_path,
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
