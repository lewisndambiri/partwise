from pathlib import Path

import numpy as np

from partwise.baseline import cosine_nearest_scores, normal_threshold, score_manifest
from partwise.data import build_manifest
from partwise.evaluation import evaluate_result, summarize_binary
from test_data import make_category


def test_cosine_scores_rank_unfamiliar_parts_higher() -> None:
    reference = np.array([[1.0, 0.0], [0.0, 1.0]])
    query = np.array([[2.0, 0.0], [-1.0, 0.0]])
    scores, nearest = cosine_nearest_scores(reference, query)
    assert np.allclose(scores, [0.0, 1.0])
    assert nearest.tolist() == [0, 1]


def test_baseline_uses_normal_validation_for_threshold(tmp_path: Path) -> None:
    make_category(tmp_path)
    manifest = build_manifest(tmp_path)

    def fake_embed(paths: list[Path], batch_size: int) -> np.ndarray:
        del batch_size
        return np.array([
            [0.0, 1.0] if "scratch" in str(path) else [1.0, 0.0]
            for path in paths
        ], dtype=np.float32)

    result = score_manifest(manifest, tmp_path, embed=fake_embed)
    metrics = evaluate_result(result)
    assert result["threshold"] == 0.0
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 0
    assert metrics["image_auroc"] == 1.0
    assert result["scores"]["test_defective"][0]["nearest_train_image"] in manifest["train_normal"]


def test_threshold_and_metrics_handle_ties() -> None:
    assert normal_threshold(np.array([0.1, 0.2, 0.2]), 0.1) == 0.2
    metrics = summarize_binary(
        np.array([0.2, 0.2, 0.8, 0.8]),
        np.array([0, 1, 0, 1]),
        0.2,
    )
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["image_auroc"] == 0.5


def test_wilson_interval_reflects_small_sample_uncertainty() -> None:
    from partwise.evaluation import wilson_interval

    low, high = wilson_interval(4, 22)
    assert low < 4 / 22 < high
    assert high - low > 0.2
