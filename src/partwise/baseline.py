"""Image-level anomaly baseline using pretrained ResNet18 features."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np


EmbeddingFunction = Callable[[list[Path], int], np.ndarray]


def cosine_nearest_scores(
    reference: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return anomaly score (1 - best cosine match) and reference index.

    Features are normalized here so tests and alternate embedders obey the same
    scoring rule. A larger score means a less familiar part.
    """
    reference = np.asarray(reference, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    if reference.ndim != 2 or query.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional arrays")
    if not len(reference) or reference.shape[1] != query.shape[1]:
        raise ValueError("Reference embeddings are empty or dimensions differ")
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    if np.any(reference_norm == 0) or np.any(query_norm == 0):
        raise ValueError("Zero-length embedding cannot be cosine-normalized")
    similarity = (query / query_norm) @ (reference / reference_norm).T
    nearest = np.argmax(similarity, axis=1)
    scores = 1.0 - similarity[np.arange(len(query)), nearest]
    return scores, nearest


def normal_threshold(scores: np.ndarray, false_reject_target: float = 0.05) -> float:
    """Calibrate solely on normal validation images using an upper quantile."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("Validation scores must be a nonempty finite vector")
    if not 0 < false_reject_target < 1:
        raise ValueError("false_reject_target must be between 0 and 1")
    return float(np.quantile(scores, 1 - false_reject_target, method="higher"))


@lru_cache(maxsize=1)
def _resnet18_components():
    try:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError(
            "Install the baseline dependencies in the local Python environment"
        ) from exc
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, weights.transforms(), device


def resnet18_embeddings(paths: list[Path], batch_size: int = 16) -> np.ndarray:
    """Extract fixed ImageNet features; never fall back to random weights."""
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Install the baseline dependencies in the local Python environment"
        ) from exc
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not paths:
        return np.empty((0, 512), dtype=np.float32)
    model, transform, device = _resnet18_components()
    chunks = []
    with torch.inference_mode():
        for offset in range(0, len(paths), batch_size):
            batch_paths = paths[offset : offset + batch_size]
            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            batch = torch.stack(tensors).to(device)
            chunks.append(model(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def score_manifest(
    manifest: dict,
    data_root: Path,
    embed: EmbeddingFunction = resnet18_embeddings,
    batch_size: int = 16,
    false_reject_target: float = 0.05,
) -> dict:
    """Fit on normal training images, calibrate on normal validation images."""
    root = data_root.expanduser().resolve()
    train_paths = manifest["train_normal"]
    if not train_paths or not manifest["validation_normal"]:
        raise ValueError("Manifest must have training and validation normal images")

    def paths(items: list[str]) -> list[Path]:
        result = [root / item for item in items]
        missing = [path for path in result if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Image missing: {missing[0]}")
        return result

    reference = embed(paths(train_paths), batch_size)
    splits = {
        "validation_normal": [(item, 0) for item in manifest["validation_normal"]],
        "test_normal": [(item, 0) for item in manifest["test_normal"]],
        "test_defective": [
            (item["image"], 1) for item in manifest["test_defective"]
        ],
    }
    result = {}
    for split, items in splits.items():
        if not items:
            raise ValueError(f"Manifest split is empty: {split}")
        image_paths = [item[0] for item in items]
        query = embed(paths(image_paths), batch_size)
        scores, nearest = cosine_nearest_scores(reference, query)
        result[split] = [
            {
                "image": image,
                "label": label,
                "score": float(score),
                "nearest_train_image": train_paths[int(match)],
            }
            for (image, label), score, match in zip(items, scores, nearest)
        ]

    threshold = normal_threshold(
        np.array([item["score"] for item in result["validation_normal"]]),
        false_reject_target,
    )
    return {
        "method": "resnet18_global_cosine_nearest",
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "threshold": threshold,
        "false_reject_target": false_reject_target,
        "scores": result,
    }
