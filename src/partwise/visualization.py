"""Shared visualization for anomaly-map inspection."""

from __future__ import annotations

import matplotlib
import numpy as np
from PIL import Image


def heat_overlay(
    image: Image.Image, anomaly_map: np.ndarray, low: float, high: float
) -> Image.Image:
    """Overlay elevated patch scores while leaving low-score pixels legible."""
    image = image.convert("RGB")
    intensity = np.clip((anomaly_map - low) / max(high - low, 1e-6), 0, 1)
    colors = matplotlib.colormaps["inferno"](intensity)[..., :3]
    heat = Image.fromarray(np.uint8(colors * 255), mode="RGB")
    heat = heat.resize(image.size, Image.Resampling.BILINEAR)
    alpha = Image.fromarray(np.uint8(175 * intensity**0.75), mode="L")
    alpha = alpha.resize(image.size, Image.Resampling.BILINEAR)
    return Image.composite(heat, image, alpha)
