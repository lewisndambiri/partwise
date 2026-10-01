"""Inspection metrics with explicit defective = positive semantics."""

from __future__ import annotations

import numpy as np


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    """Approximate 95% binomial interval with the usual z=1.96."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial count")
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [float(max(0, center - margin)), float(min(1, center + margin))]


def summarize_binary(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels):
        raise ValueError("Scores and labels must be equal-length vectors")
    if not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("Scores must be nonempty and finite")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must be 0 (good) or 1 (defective)")
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("Evaluation needs both good and defective images")

    predicted = scores > threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0

    # Pairwise form treats tied scores as half-correct and is small for one
    # MVTec category. It avoids a hidden classifier threshold in AUROC.
    good = scores[~positive]
    defective = scores[positive]
    comparisons = defective[:, None] - good[None, :]
    auroc = float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))

    # Average precision groups tied scores before updating recall.
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_positive = positive[order]
    cumulative_tp = 0
    cumulative_seen = 0
    ap = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_tp = int(np.sum(sorted_positive[start:end]))
        cumulative_tp += group_tp
        cumulative_seen += end - start
        ap += (group_tp / len(defective)) * (cumulative_tp / cumulative_seen)
        start = end

    return {
        "n_good": len(good),
        "n_defective": len(defective),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "defect_recall": recall,
        "defect_recall_wilson_95": wilson_interval(tp, tp + fn),
        "false_reject_rate": fp / (fp + tn),
        "false_reject_rate_wilson_95": wilson_interval(fp, fp + tn),
        "precision_on_dataset": precision,
        "image_auroc": auroc,
        "image_average_precision": ap,
    }


def evaluate_result(result: dict) -> dict:
    test = result["scores"]["test_normal"] + result["scores"]["test_defective"]
    return summarize_binary(
        np.array([item["score"] for item in test]),
        np.array([item["label"] for item in test]),
        result["threshold"],
    )
