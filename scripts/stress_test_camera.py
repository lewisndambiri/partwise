"""Probe fixed PatchCore decisions under simple simulated camera changes.

These are descriptive tests on the official held-out test images. No threshold,
model weight, or transformation strength is selected from their outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import average_precision_score, roc_auc_score

from partwise.inference import load_patchcore_model, predict_image


def condition_image(image: Image.Image, condition: str) -> Image.Image:
    if condition == "original":
        return image
    if condition == "dim_20_percent":
        return ImageEnhance.Brightness(image).enhance(0.8)
    if condition == "bright_20_percent":
        return ImageEnhance.Brightness(image).enhance(1.2)
    if condition == "blur_radius_1_5":
        return image.filter(ImageFilter.GaussianBlur(radius=1.5))
    raise ValueError(f"Unknown condition: {condition}")


def summarize(records: list[dict], threshold: float) -> dict:
    good = [item for item in records if item["label"] == 0]
    defective = [item for item in records if item["label"] == 1]
    labels = np.array([item["label"] for item in records])
    scores = np.array([item["score"] for item in records])
    return {
        "defects_found": sum(item["score"] > threshold for item in defective),
        "defects_total": len(defective),
        "good_rejected": sum(item["score"] > threshold for item in good),
        "good_total": len(good),
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_average_precision": float(average_precision_score(labels, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/metal_nut_manifest.json"))
    parser.add_argument("--result", type=Path, default=Path("artifacts/metal_nut_patchcore.json"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/metal_nut_patchcore_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/camera_stress_test.json"))
    args = parser.parse_args()

    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    result = json.loads(args.result.read_text())
    threshold = float(result["threshold"])
    image_size = int(result["configuration"]["image_size"])
    model = load_patchcore_model(args.model)
    cases = [(path, 0) for path in manifest["test_normal"]]
    cases.extend((item["image"], 1) for item in manifest["test_defective"])
    conditions = ("original", "dim_20_percent", "bright_20_percent", "blur_radius_1_5")
    observations = {condition: [] for condition in conditions}

    for index, (path, label) in enumerate(cases, start=1):
        with Image.open(args.data_root / path) as source:
            image = source.convert("RGB")
            for condition in conditions:
                changed = condition_image(image, condition)
                score, _ = predict_image(model, changed, image_size)
                observations[condition].append({"image": path, "label": label, "score": score})
        if index % 25 == 0 or index == len(cases):
            print(f"Scored {index}/{len(cases)} images in four conditions", flush=True)

    original = {item["image"]: item for item in observations["original"]}
    summary = {}
    for condition in conditions:
        records = observations[condition]
        stats = summarize(records, threshold)
        stats["new_misses"] = [
            item["image"] for item in records
            if item["label"] == 1
            and original[item["image"]]["score"] > threshold
            and item["score"] <= threshold
        ]
        stats["new_false_rejects"] = [
            item["image"] for item in records
            if item["label"] == 0
            and original[item["image"]]["score"] <= threshold
            and item["score"] > threshold
        ]
        summary[condition] = stats

    output = {
        "description": "Held-out test images transformed at inference only; descriptive camera-condition stress test",
        "threshold": threshold,
        "conditions": {
            "original": "Unchanged official test image",
            "dim_20_percent": "Pillow brightness factor 0.8",
            "bright_20_percent": "Pillow brightness factor 1.2",
            "blur_radius_1_5": "Pillow Gaussian blur radius 1.5 source pixels",
        },
        "summary": summary,
        "scores": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for name, item in summary.items():
        print(
            f"{name}: found {item['defects_found']}/{item['defects_total']}, "
            f"good rejected {item['good_rejected']}/{item['good_total']}, "
            f"AUROC {item['image_auroc']:.3f}"
        )


if __name__ == "__main__":
    main()
