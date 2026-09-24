"""M2 第一版低照度增强：LAB-L CLAHE 与受限 Gamma。"""
from __future__ import annotations

import cv2
import numpy as np


def enhance(image: np.ndarray, clip_limit: float = 2.0,
            tile_grid: tuple[int, int] = (8, 8), gamma: float = 0.9) -> np.ndarray:
    """Return uint8 RGB with unchanged height/width and pixel coordinates."""
    if not 1.0 <= clip_limit <= 4.0 or any(size < 1 for size in tile_grid):
        raise ValueError("Invalid CLAHE parameters")
    if not 0.75 <= gamma <= 1.0:
        raise ValueError("Gamma must be in [0.75, 1.0]")
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(l)
    if gamma < 1.0:
        lut = np.rint((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255).astype(np.uint8)
        l = cv2.LUT(l, lut)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
