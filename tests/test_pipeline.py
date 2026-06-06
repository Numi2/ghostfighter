from pathlib import Path

from ghostfighter.dataset import generate_trace_dataset
from ghostfighter.train import train_behavior_cloning
from ghostfighter.config import TrainConfig
from ghostfighter.evaluate import evaluate_policy, run_safety_threshold_sweep, run_scenario_suite
from ghostfighter.improve import run_scale_study
from ghostfighter.render import make_safety_dashboard
from ghostfighter.cli import build_parser


def test_small_pipeline_runs(tmp_path: Path):
    data = tmp_path / "traces.npz"
    summary = generate_trace_dataset(data, episodes_per_style=1, seed=77, max_steps=30)
    assert summary["samples"] > 20
    train = train_behavior_cloning(data, tmp_path / "models", config=TrainConfig(epochs=1, batch_size=64, seed=78), hidden=48)
    assert Path(train["model_path"]).exists()
    result = evaluate_policy(train["model_path"], tmp_path / "reports", episodes=4, seed=79, max_steps=30)
    assert "summary" in result
    assert (tmp_path / "reports" / "match_results.csv").exists()


def test_tiny_benchmark_outputs(tmp_path: Path):
    data = tmp_path / "traces.npz"
    generate_trace_dataset(data, episodes_per_style=1, seed=177, max_steps=25)
    train = train_behavior_cloning(data, tmp_path / "models", config=TrainConfig(epochs=1, batch_size=64, seed=178), hidden=48)
    result = run_scenario_suite(train["model_path"], tmp_path / "reports", episodes=4, seed=179, max_steps=30, suite="regression")
    assert result["summary"]["by_mode"]
    assert (tmp_path / "reports" / "scenario_results.csv").exists()
    assert (tmp_path / "reports" / "scenario_summary.json").exists()
    assert (tmp_path / "reports" / "safety_case.md").exists()
    assert (tmp_path / "reports" / "safety_tuning.json").exists()
    assert (tmp_path / "reports" / "replays" / "scenario_replays.json").exists()
    path = make_safety_dashboard(tmp_path / "reports")
    assert Path(path).exists()


def test_safety_threshold_sweep_recommends_candidate(tmp_path: Path):
    data = tmp_path / "traces.npz"
    generate_trace_dataset(data, episodes_per_style=1, seed=277, max_steps=25)
    train = train_behavior_cloning(data, tmp_path / "models", config=TrainConfig(epochs=1, batch_size=64, seed=278), hidden=48)
    result = run_safety_threshold_sweep(train["model_path"], tmp_path / "reports", episodes=4, seed=279, max_steps=30, suite="regression", thresholds=(0.50, 0.70))
    assert result["summary"]["recommended_threshold"] in {0.5, 0.7}
    assert (tmp_path / "reports" / "safety_tuning.csv").exists()


def test_tiny_scale_study_outputs(tmp_path: Path):
    result = run_scale_study(
        tmp_path / "scaling",
        episode_schedule=(1, 2),
        epochs=1,
        eval_episodes=4,
        seed=379,
        max_steps=25,
        batch_size=64,
        hidden=48,
    )
    assert result["summary"]["generations"] == 2
    assert result["summary"]["sample_scale"] > 1.0
    assert (tmp_path / "scaling" / "scaling_study.csv").exists()
    assert (tmp_path / "scaling" / "scaling_study.json").exists()
    assert (tmp_path / "scaling" / "scaling_dashboard.png").exists()
    assert (tmp_path / "scaling" / "LEARNING_CASE.md").exists()


def test_cli_accepts_benchmark_options():
    parser = build_parser()
    args = parser.parse_args(["benchmark", "--suite", "regression", "--episodes", "4"])
    assert args.command == "benchmark"
    assert args.suite == "regression"
    args = parser.parse_args(["tune-safety", "--suite", "regression", "--episodes", "4"])
    assert args.command == "tune-safety"
    args = parser.parse_args(["scale-study", "--episodes-schedule", "1,2", "--epochs", "1"])
    assert args.command == "scale-study"
    assert args.episodes_schedule == [1, 2]
    args = parser.parse_args(["all", "--benchmark"])
    assert args.command == "all"
    assert args.benchmark is True
    args = parser.parse_args(["all", "--scale-study"])
    assert args.scale_study is True
