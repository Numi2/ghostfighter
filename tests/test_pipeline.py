from pathlib import Path

from ghostfighter.dataset import generate_trace_dataset
from ghostfighter.attributes import AttributePolicy, load_policy_spec, sample_attribute_policies
from ghostfighter.train import train_behavior_cloning
from ghostfighter.config import TrainConfig
from ghostfighter.env import FightEnv
from ghostfighter.evaluate import evaluate_policy, run_safety_threshold_sweep, run_scenario_suite
from ghostfighter.improve import run_scale_study
from ghostfighter.render import make_safety_dashboard
from ghostfighter.selfplay import run_population_self_play
from ghostfighter.rl import PPOConfig, train_ppo_self_play
from ghostfighter.robustness import run_robustness_ablations
from ghostfighter.replay import make_replay_viewer
from ghostfighter.vector_env import SyncVectorFightEnv
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


def test_attribute_policy_spec_and_sampling_are_deterministic():
    spec = load_policy_spec()
    a = sample_attribute_policies(spec, variants_per_archetype=2, seed=11)
    b = sample_attribute_policies(spec, variants_per_archetype=2, seed=11)
    assert [p.vector().tolist() for p in a] == [p.vector().tolist() for p in b]
    assert len(a) == 8
    env = FightEnv(seed=12)
    obs0, _ = env.reset(randomize=False)
    action = AttributePolicy(a[0]).select_action(obs0, env, 0)
    assert isinstance(action, int)


def test_attribute_dataset_outputs_gen0_artifacts(tmp_path: Path):
    data = tmp_path / "run" / "data" / "traces.npz"
    summary = generate_trace_dataset(data, episodes_per_style=1, seed=55, max_steps=20, source="attributes", variants_per_archetype=2)
    assert summary["source"] == "attributes"
    assert summary["policy_variants"] == 8
    assert (tmp_path / "run" / "gen0" / "policy_specs.resolved.json").exists()
    assert (tmp_path / "run" / "gen0" / "policy_variants.csv").exists()
    assert (tmp_path / "run" / "gen0" / "GENERATION_ZERO_CARD.md").exists()
    train = train_behavior_cloning(data, tmp_path / "models", config=TrainConfig(epochs=1, batch_size=64, seed=56), hidden=48)
    assert Path(train["model_path"]).exists()


def test_domain_randomized_dataset_outputs_card(tmp_path: Path):
    data = tmp_path / "run" / "data" / "traces.npz"
    summary = generate_trace_dataset(
        data,
        episodes_per_style=1,
        seed=65,
        max_steps=18,
        source="attributes",
        variants_per_archetype=1,
        domain_randomization=True,
        domain_intensity=0.4,
    )
    assert summary["domain_randomization"]["enabled"] is True
    assert (tmp_path / "run" / "gen0" / "DOMAIN_RANDOMIZATION_CARD.md").exists()


def test_tiny_selfplay_outputs_population_metrics(tmp_path: Path):
    result = run_population_self_play(
        tmp_path / "selfplay",
        generations=1,
        matches_per_pair=1,
        variants_per_role=1,
        seed=91,
        max_steps=20,
        domain_randomization=True,
        domain_intensity=0.25,
    )
    summary = result["summary"]
    assert summary["population_size"] == 5
    assert summary["matches"] == 20
    assert "exploitability_elo_gap" in summary
    assert "policy_diversity_jsd" in summary
    assert (tmp_path / "selfplay" / "selfplay_matches.csv").exists()
    assert (tmp_path / "selfplay" / "population.csv").exists()
    assert (tmp_path / "selfplay" / "SELF_PLAY_CARD.md").exists()
    assert (tmp_path / "selfplay" / "DOMAIN_RANDOMIZATION_CARD.md").exists()


def test_sync_vector_env_steps_batch():
    env = SyncVectorFightEnv(num_envs=3, seed=123)
    red, blue = env.reset(randomize=False)
    assert red.shape[0] == 3
    step = env.step([0, 1, 2], [0, 1, 2])
    assert step.obs_red.shape[0] == 3
    assert step.reward_red.shape == (3,)


def test_tiny_ppo_selfplay_outputs_leaderboard(tmp_path: Path):
    result = train_ppo_self_play(
        tmp_path / "rl",
        config=PPOConfig(
            updates=1,
            matches_per_update=2,
            max_steps=20,
            epochs=1,
            batch_size=64,
            hidden=48,
            seed=191,
            domain_randomization=False,
        ),
    )
    assert result["summary"]["updates"] == 1
    assert Path(result["model_path"]).exists()
    assert (tmp_path / "rl" / "ppo_training_curve.csv").exists()
    assert (tmp_path / "rl" / "leaderboard.csv").exists()
    assert (tmp_path / "rl" / "LEADERBOARD.md").exists()
    assert (tmp_path / "rl" / "RL_TRAINING_CARD.md").exists()


def test_tiny_robustness_and_replay_outputs(tmp_path: Path):
    result = train_ppo_self_play(
        tmp_path / "rl",
        config=PPOConfig(
            updates=1,
            matches_per_update=1,
            max_steps=15,
            epochs=1,
            batch_size=32,
            hidden=32,
            seed=211,
            domain_randomization=False,
        ),
    )
    robustness = run_robustness_ablations(result["model_path"], tmp_path / "robustness", episodes=1, seed=212, max_steps=15)
    assert robustness["summary"]["ablations"]
    assert (tmp_path / "robustness" / "ROBUSTNESS_REPORT.md").exists()
    assert (tmp_path / "robustness" / "robustness_dashboard.png").exists()
    replay = make_replay_viewer(result["model_path"], tmp_path / "replay", seed=213, max_steps=15, domain_randomization=False)
    assert Path(replay["replay"]).exists()
    assert Path(replay["viewer"]).exists()


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
    args = parser.parse_args(["generate-data", "--source", "attributes", "--variants-per-archetype", "2"])
    assert args.source == "attributes"
    args = parser.parse_args(["generate-data", "--domain-randomization", "--domain-intensity", "0.5"])
    assert args.domain_randomization is True
    args = parser.parse_args(["forge-zero", "--variants-per-archetype", "2"])
    assert args.command == "forge-zero"
    args = parser.parse_args(["self-play", "--generations", "1", "--matches-per-pair", "1"])
    assert args.command == "self-play"
    args = parser.parse_args(["train-rl", "--updates", "1", "--matches-per-update", "2"])
    assert args.command == "train-rl"
    args = parser.parse_args(["robustness", "--episodes", "1"])
    assert args.command == "robustness"
    args = parser.parse_args(["replay-viewer", "--max-steps", "10"])
    assert args.command == "replay-viewer"
    args = parser.parse_args(["scale-plan"])
    assert args.command == "scale-plan"
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
    args = parser.parse_args(["all", "--self-play", "--domain-randomization"])
    assert args.self_play is True
    assert args.domain_randomization is True
    args = parser.parse_args(["all", "--rl"])
    assert args.rl is True
    args = parser.parse_args(["all", "--rl", "--robustness", "--replay-viewer"])
    assert args.robustness is True
    assert args.replay_viewer is True
    args = parser.parse_args(["all", "--gen0-source", "attributes", "--variants-per-archetype", "2"])
    assert args.gen0_source == "attributes"
