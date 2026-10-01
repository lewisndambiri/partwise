"""Render three held-out inspection cases with their anomaly maps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from partwise.visualization import heat_overlay


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA_ROOT = ROOT / "data/mvtec_ad"
OUTPUT = ROOT / "reports/figures/inspection_cases.png"


def main() -> None:
    manifest = json.loads((ARTIFACTS / "metal_nut_manifest.json").read_text())
    result = json.loads((ARTIFACTS / "metal_nut_patchcore.json").read_text())
    score_lookup = {
        item["image"]: item["score"]
        for split in ("test_normal", "test_defective")
        for item in result["scores"][split]
    }
    mask_lookup = {item["image"]: item["mask"] for item in manifest["test_defective"]}
    with np.load(ARTIFACTS / "metal_nut_patchcore_maps.npz") as bundle:
        maps = {
            str(path): anomaly_map.astype(np.float32)
            for path, anomaly_map in zip(bundle["image_paths"], bundle["anomaly_maps"])
        }
    low, high = np.percentile(np.stack(list(maps.values())), [75, 99])
    examples = [
        ("Detected bent defect", "metal_nut/test/bent/000.png"),
        ("Missed bent defect", "metal_nut/test/bent/006.png"),
        ("False reject: good part", "metal_nut/test/good/016.png"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10.3), facecolor="#111b29")
    headers = ("Original", "PatchCore anomaly overlay", "Ground truth mask")
    for col, title in enumerate(headers):
        axes[0, col].set_title(title, color="#e8f1f0", fontsize=13, pad=13)
    for row, (label, path) in enumerate(examples):
        with Image.open(DATA_ROOT / path) as source:
            original = source.convert("RGB")
            overlay = heat_overlay(original, maps[path], float(low), float(high))
        axes[row, 0].imshow(original)
        axes[row, 1].imshow(overlay)
        if path in mask_lookup:
            with Image.open(DATA_ROOT / mask_lookup[path]) as source:
                axes[row, 2].imshow(source.convert("L"), cmap="gray", vmin=0, vmax=255)
        else:
            axes[row, 2].set_facecolor("#111b29")
            axes[row, 2].text(
                0.5, 0.5, "No defect mask\n(good part)",
                ha="center", va="center", transform=axes[row, 2].transAxes,
                color="#b7c8d2", fontsize=13,
            )
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
        verdict = "FLAGGED" if score_lookup[path] > result["threshold"] else "ACCEPTED"
        axes[row, 0].set_ylabel(
            f"{label}\n{verdict} · score {score_lookup[path]:.2f}",
            color="#e8f1f0", fontsize=11, labelpad=15,
        )
    fig.text(
        0.5, 0.012,
        f"Fixed threshold {result['threshold']:.2f} from normal validation. "
        "Overlay scale: 75th–99th percentile of benchmark patch scores.",
        ha="center", color="#a8bdc9", fontsize=10,
    )
    fig.subplots_adjust(left=0.20, right=0.98, top=0.95, bottom=0.055, hspace=0.15, wspace=0.035)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
