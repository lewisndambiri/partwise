"""Reload a fitted PatchCore model and inspect a new local image."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_patchcore_model(model_path: Path):
    import torch
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    model = PatchcoreModel(
        layers=("layer2", "layer3"),
        backbone="resnet18",
        pre_trained=False,
        num_neighbors=9,
    )
    weights = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    return model


def predict_image(model, image, image_size: int = 224) -> tuple[float, np.ndarray]:
    import torch
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        prediction = model(tensor)
    return (
        float(prediction.pred_score.item()),
        prediction.anomaly_map.detach().cpu().numpy().squeeze().astype(np.float32),
    )
