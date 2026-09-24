"""Deterministic, single-frame RGB quality measurements for the first M2 gate."""
from __future__ import annotations

import cv2
import numpy as np


def measure(image_rgb: np.ndarray, under_lt: int = 40, over_gt: int = 245) -> dict[str, float]:
    """Measure source-frame quality; thresholds are uint8 grayscale levels.

    Noise is a robust median-filter residual measured on low-gradient pixels.
    It is only a gate heuristic, not a calibrated sensor-noise estimate.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32)
    residual = cv2.absdiff(gray, cv2.medianBlur(gray, 3)).astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    smooth = residual[gradient < 20.0]
    if smooth.size < 100:
        smooth = residual.ravel()
    sigma = float(np.median(smooth) / 0.6745)
    channel_means = image_rgb.mean(axis=(0, 1))
    return {
        "brightness_mean": float(gray_f.mean()),
        "brightness_p10": float(np.percentile(gray, 10)),
        "underexposed_fraction": float(np.mean(gray < under_lt)),
        "overexposed_fraction": float(np.mean(gray > over_gt)),
        "contrast_std": float(gray_f.std()),
        "blur_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "noise_sigma": sigma,
        "color_cast_score": float((channel_means.max() - channel_means.min()) / 255.0),
    }
