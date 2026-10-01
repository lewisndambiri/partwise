"""CPU-conscious PatchCore experiment on the locked Partwise manifest."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from partwise.baseline import normal_threshold
from partwise.evaluation import evaluate_result


def run_patchcore(
    manifest: dict,
    data_root: Path,
    *,
    batch_size: int = 8,
    coreset_ratio: float = 0.005,
    image_size: int = 224,
    seed: int = 42,
    false_reject_target: float = 0.05,
    model_output: Path | None = None,
    maps_output: Path | None = None,
) -> dict:
    """Fit normal patches, lock a normal-only threshold, then evaluate test data."""
    if batch_size < 1 or image_size < 32 or not 0 < coreset_ratio <= 1:
        raise ValueError("Invalid batch size, image size, or coreset ratio")
    try:
        import torch
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel
        from PIL import Image
        from sklearn.metrics import average_precision_score, roc_auc_score
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor
        from torchvision.transforms import InterpolationMode
    except ImportError as exc:
        raise RuntimeError("Install Anomalib and the PatchCore dependencies first") from exc

    root = data_root.expanduser().resolve()
    torch.manual_seed(seed)
    np.random.seed(seed)
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    mask_resize = Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST)
    model = PatchcoreModel(
        layers=("layer2", "layer3"),
        backbone="resnet18",
        pre_trained=True,
        num_neighbors=9,
    )
    model.train()
    model.feature_extractor.eval()  # Keep pretrained BatchNorm statistics fixed.

    def batches(items: list[str]):
        for start in range(0, len(items), batch_size):
            selected = items[start : start + batch_size]
            tensors = []
            for item in selected:
                path = root / item
                if not path.is_file():
                    raise FileNotFoundError(f"Image missing: {path}")
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            yield selected, torch.stack(tensors)

    train_paths = manifest["train_normal"]
    if not train_paths:
        raise ValueError("Manifest has no normal training images")
    print(f"PatchCore: extracting patches from {len(train_paths)} normal images", flush=True)
    fit_start = time.perf_counter()
    with torch.inference_mode():
        for _, tensor in batches(train_paths):
            model(tensor)
    print(f"PatchCore: selecting {coreset_ratio:.3%} of training patches", flush=True)
    model.subsample_embedding(coreset_ratio)
    fit_seconds = time.perf_counter() - fit_start
    memory_size = int(model.memory_bank.shape[0])
    model.eval()

    if model_output is not None:
        model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_output)

    splits = {
        "validation_normal": [(item, 0) for item in manifest["validation_normal"]],
        "test_normal": [(item, 0) for item in manifest["test_normal"]],
        "test_defective": [
            (item["image"], 1) for item in manifest["test_defective"]
        ],
    }
    scored = {}
    test_maps = []
    map_paths = []
    inference_seconds = 0.0
    for split, items in splits.items():
        if not items:
            raise ValueError(f"Manifest split is empty: {split}")
        print(f"PatchCore: scoring {split} ({len(items)} images)", flush=True)
        records = []
        split_paths = [item[0] for item in items]
        for selected, tensor in batches(split_paths):
            start = time.perf_counter()
            with torch.inference_mode():
                output = model(tensor)
            inference_seconds += time.perf_counter() - start
            scores = output.pred_score.detach().cpu().numpy().reshape(-1)
            maps = output.anomaly_map.detach().cpu().numpy()
            for offset, image_path in enumerate(selected):
                label = items[len(records)][1]
                records.append({
                    "image": image_path,
                    "label": label,
                    "score": float(scores[offset]),
                })
                if split != "validation_normal":
                    test_maps.append(np.asarray(maps[offset]).squeeze().astype(np.float16))
                    map_paths.append(image_path)
        scored[split] = records

    threshold = normal_threshold(
        np.array([item["score"] for item in scored["validation_normal"]]),
        false_reject_target,
    )
    result = {
        "method": "anomalib_patchcore_resnet18",
        "configuration": {
            "anomalib_version": "2.6.2",
            "backbone": "resnet18",
            "layers": ["layer2", "layer3"],
            "image_size": image_size,
            "coreset_ratio": coreset_ratio,
            "coreset_patches": memory_size,
            "num_neighbors": 9,
            "seed": seed,
        },
        "threshold": threshold,
        "false_reject_target": false_reject_target,
        "scores": scored,
        "runtime": {
            "fit_seconds": fit_seconds,
            "inference_seconds_all_splits": inference_seconds,
            "inference_ms_per_image": inference_seconds * 1000 / sum(len(items) for items in splits.values()),
        },
    }
    result["metrics"] = evaluate_result(result)

    # Evaluate localization with exactly the official test masks. These pixel
    # metrics are descriptive and never feed model or threshold selection.
    masks_by_image = {
        item["image"]: item["mask"] for item in manifest["test_defective"]
    }
    pixel_masks = []
    for image_path in map_paths:
        if image_path in masks_by_image:
            with Image.open(root / masks_by_image[image_path]) as mask:
                pixel_masks.append(np.asarray(mask_resize(mask.convert("L"))) > 0)
        else:
            pixel_masks.append(np.zeros((image_size, image_size), dtype=bool))
    flat_maps = np.stack(test_maps).astype(np.float32).ravel()
    flat_masks = np.stack(pixel_masks).ravel()
    result["metrics"]["pixel_auroc"] = float(roc_auc_score(flat_masks, flat_maps))
    result["metrics"]["pixel_average_precision"] = float(average_precision_score(flat_masks, flat_maps))
    result["metrics"]["pixel_defect_fraction"] = float(np.mean(flat_masks))

    if maps_output is not None:
        maps_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            maps_output,
            image_paths=np.array(map_paths),
            anomaly_maps=np.stack(test_maps),
        )
    return result
