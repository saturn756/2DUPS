"""Conservative color bilateral filter; no geometric coordinate change."""
from __future__ import annotations

import cv2
import numpy as np


def denoise(image: np.ndarray, diameter: int = 5,
            sigma_color: float = 20, sigma_space: float = 5) -> np.ndarray:
    if diameter < 3 or diameter % 2 == 0 or sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("Invalid bilateral filter parameters")
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
