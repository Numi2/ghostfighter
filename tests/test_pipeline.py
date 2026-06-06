from pathlib import Path

from ghostfighter.dataset import generate_trace_dataset
from ghostfighter.train import train_behavior_cloning
from ghostfighter.config import TrainConfig
from ghostfighter.evaluate import evaluate_policy


def test_small_pipeline_runs(tmp_path: Path):
    data = tmp_path / "traces.npz"
    summary = generate_trace_dataset(data, episodes_per_style=1, seed=77, max_steps=30)
    assert summary["samples"] > 20
    train = train_behavior_cloning(data, tmp_path / "models", config=TrainConfig(epochs=1, batch_size=64, seed=78), hidden=48)
    assert Path(train["model_path"]).exists()
    result = evaluate_policy(train["model_path"], tmp_path / "reports", episodes=4, seed=79, max_steps=30)
    assert "summary" in result
    assert (tmp_path / "reports" / "match_results.csv").exists()
