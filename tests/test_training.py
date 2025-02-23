import json
from pathlib import Path

from grpo_math.train import train_grpo


def test_tiny_training_tracks_kl_entropy_and_length():
    examples = json.loads(Path("data/examples.json").read_text())[:2]
    _, history = train_grpo(examples, steps=2, group_size=3, max_length=5, seed=4)
    assert len(history) == 2
    for metrics in history:
        assert {"reward", "kl", "entropy", "length", "loss"} <= metrics.keys()
        assert metrics["kl"] >= 0
        assert metrics["entropy"] > 0
        assert 1 <= metrics["length"] <= 6
