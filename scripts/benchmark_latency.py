"""Measure single-image model-and-scoring latency with inputs preprocessed."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from partwise.baseline import _resnet18_components, resnet18_embeddings
from partwise.inference import load_patchcore_model


def summarize(samples: list[float]) -> dict:
    values = np.asarray(samples) * 1000
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "samples": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/metal_nut_manifest.json"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/metal_nut_patchcore_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/batch1_latency.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.data_root.resolve()
    test_paths = manifest["test_normal"][:10] + [
        item["image"] for item in manifest["test_defective"][:10]
    ]
    reference_paths = [root / item for item in manifest["train_normal"]]
    reference = resnet18_embeddings(reference_paths, batch_size=16)
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    baseline_model, baseline_transform, device = _resnet18_components()
    reference_tensor = torch.from_numpy(reference).to(device)
    patch_model = load_patchcore_model(args.model)
    patch_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    inputs = []
    for path in test_paths:
        with Image.open(root / path) as image:
            rgb = image.convert("RGB")
            inputs.append((
                baseline_transform(rgb).unsqueeze(0).to(device),
                patch_transform(rgb).unsqueeze(0),
            ))

    def baseline_predict(tensor):
        feature = baseline_model(tensor)
        feature = torch.nn.functional.normalize(feature, dim=1)
        return 1 - (feature @ reference_tensor.T).max()

    def patch_predict(tensor):
        return patch_model(tensor).pred_score

    with torch.inference_mode():
        for tensor_base, tensor_patch in inputs[:3]:
            baseline_predict(tensor_base)
            patch_predict(tensor_patch)
        baseline_samples = []
        patch_samples = []
        for repeat in range(3):
            for tensor_base, tensor_patch in inputs:
                # Alternate order to limit systematic thermal/order bias.
                operations = (
                    ((baseline_predict, tensor_base, baseline_samples), (patch_predict, tensor_patch, patch_samples))
                    if repeat % 2 == 0 else
                    ((patch_predict, tensor_patch, patch_samples), (baseline_predict, tensor_base, baseline_samples))
                )
                for predict, tensor, samples in operations:
                    start = time.perf_counter()
                    predict(tensor)
                    samples.append(time.perf_counter() - start)

    result = {
        "boundary": "preprocessed tensor to anomaly score, batch size 1; excludes image I/O and preprocessing",
        "image_count": len(test_paths),
        "repeats": 3,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "processor": platform.processor(),
        "baseline": summarize(baseline_samples),
        "patchcore": summarize(patch_samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
