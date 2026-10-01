"""Audit MVTec AD files and create a deterministic, leakage-free manifest."""

from __future__ import annotations

import random
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


class DatasetError(ValueError):
    """The expected MVTec AD category layout is missing or incomplete."""


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise DatasetError(f"Missing directory: {folder}")
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_manifest(
    data_root: Path,
    category: str = "metal_nut",
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> dict:
    """Index one category without moving images or using the official test set to fit.

    ``data_root`` is the directory containing MVTec AD category directories.
    Validation contains only held-out defect-free training images. This lets a
    later model calibrate its false-reject threshold without seeing test labels.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not category or Path(category).name != category or category in {".", ".."}:
        raise ValueError("category must be a single directory name")

    root = data_root.expanduser().resolve()
    category_root = root / category
    normal_train = _images(category_root / "train" / "good")
    normal_test = _images(category_root / "test" / "good")
    if len(normal_train) < 2:
        raise DatasetError("At least two defect-free training images are required")
    if not normal_test:
        raise DatasetError("No defect-free test images found")

    test_dir = category_root / "test"
    defect_dirs = sorted(
        path for path in test_dir.iterdir()
        if path.is_dir() and path.name != "good"
    )
    if not defect_dirs:
        raise DatasetError("No defect categories found in the test directory")

    defective = []
    for defect_dir in defect_dirs:
        defect_images = _images(defect_dir)
        if not defect_images:
            raise DatasetError(f"No defect images found in {defect_dir}")
        mask_dir = category_root / "ground_truth" / defect_dir.name
        for image in defect_images:
            mask = mask_dir / f"{image.stem}_mask.png"
            if not mask.is_file():
                raise DatasetError(f"Missing defect mask: {mask}")
            defective.append({
                "image": _relative(image, root),
                "mask": _relative(mask, root),
                "defect_type": defect_dir.name,
            })

    shuffled = normal_train.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = min(
        len(shuffled) - 1,
        max(1, round(len(shuffled) * validation_fraction)),
    )
    validation = sorted(shuffled[:validation_count])
    training = sorted(shuffled[validation_count:])

    return {
        "schema_version": 1,
        "dataset": "MVTec AD",
        "category": category,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "train_normal": [_relative(path, root) for path in training],
        "validation_normal": [_relative(path, root) for path in validation],
        "test_normal": [_relative(path, root) for path in normal_test],
        "test_defective": defective,
    }
