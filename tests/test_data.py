from pathlib import Path

import pytest

from partwise.data import DatasetError, build_manifest


def make_category(root: Path) -> Path:
    category = root / "metal_nut"
    for name in ("001", "002", "003", "004", "005"):
        image = category / "train" / "good" / f"{name}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.touch()
    good = category / "test" / "good" / "006.png"
    good.parent.mkdir(parents=True, exist_ok=True)
    good.touch()
    defect = category / "test" / "scratch" / "007.png"
    defect.parent.mkdir(parents=True, exist_ok=True)
    defect.touch()
    mask = category / "ground_truth" / "scratch" / "007_mask.png"
    mask.parent.mkdir(parents=True, exist_ok=True)
    mask.touch()
    return category


def test_manifest_is_deterministic_and_keeps_test_separate(tmp_path: Path) -> None:
    make_category(tmp_path)
    first = build_manifest(tmp_path, seed=11)
    second = build_manifest(tmp_path, seed=11)

    assert first == second
    assert len(first["train_normal"]) == 4
    assert len(first["validation_normal"]) == 1
    train = set(first["train_normal"])
    validation = set(first["validation_normal"])
    test = set(first["test_normal"]) | {
        item["image"] for item in first["test_defective"]
    }
    assert not train & validation
    assert not (train | validation) & test
    assert first["test_defective"] == [{
        "image": "metal_nut/test/scratch/007.png",
        "mask": "metal_nut/ground_truth/scratch/007_mask.png",
        "defect_type": "scratch",
    }]


def test_audit_rejects_missing_mask(tmp_path: Path) -> None:
    category = make_category(tmp_path)
    (category / "ground_truth" / "scratch" / "007_mask.png").unlink()

    with pytest.raises(DatasetError, match="Missing defect mask"):
        build_manifest(tmp_path)
