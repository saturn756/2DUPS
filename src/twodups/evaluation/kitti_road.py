"""Local image-plane KITTI Road binary metrics; not the official BEV benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np


def gt_key(stem: str) -> str:
    match = re.fullmatch(r"(um|umm|uu)_(?:(?:road|lane)_)?(\d{6})", stem)
    if match is None:
        raise ValueError(f"Invalid KITTI ground-truth name: {stem}")
    return f"{match[1]}_{match[2]}"


def pred_key(stem: str) -> str:
    match = re.fullmatch(r"(um|umm|uu)_(\d{6})(?:_class_[0-9a-f]+)?", stem)
    if match is None:
        raise ValueError(f"Invalid KITTI prediction name: {stem}")
    return f"{match[1]}_{match[2]}"


def pair(predictions: str | Path, targets: str | Path, *, tag: str) -> list[tuple[Path, Path]]:
    if tag not in {"road", "lane"}:
        raise ValueError("KITTI target tag must be road or lane")
    truth = {gt_key(path.stem): path for path in Path(targets).glob(f"*_{tag}_*.png")}
    result = []
    seen = set()
    for prediction in sorted(Path(predictions).rglob("*.png")):
        key = pred_key(prediction.stem)
        if key in seen:
            raise ValueError(f"Duplicate prediction for {key}")
        seen.add(key)
        if key not in truth:
            raise FileNotFoundError(f"No {tag} ground truth for {key}")
        result.append((prediction, truth[key]))
    if not result:
        raise ValueError("No prediction masks found")
    return result


@dataclass(frozen=True)
class BinaryScore:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def evaluated_pixels(self) -> int:
        return self.true_positive + self.false_positive + self.false_negative + self.true_negative

    @property
    def iou(self) -> float:
        denominator = self.true_positive + self.false_positive + self.false_negative
        return self.true_positive / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return 2 * self.true_positive / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 1.0

    @property
    def pixel_accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.evaluated_pixels


def evaluate_binary(pairs: list[tuple[Path, Path]], *, target_id: int) -> BinaryScore:
    if isinstance(target_id, bool) or not isinstance(target_id, int) or target_id < 0:
        raise ValueError("target_id must be a non-negative class ID")
    counts = np.zeros(4, dtype=np.int64)
    for prediction_path, truth_path in pairs:
        prediction = cv2.imread(str(prediction_path), cv2.IMREAD_UNCHANGED)
        truth = cv2.imread(str(truth_path), cv2.IMREAD_UNCHANGED)
        if prediction is None or truth is None:
            raise ValueError("Cannot decode prediction or ground-truth mask")
        if prediction.ndim != 2 or truth.ndim != 3 or truth.shape[2] != 3:
            raise ValueError("Expected a class-ID prediction and a three-channel KITTI mask")
        if prediction.shape != truth.shape[:2]:
            raise ValueError("Prediction/ground-truth size mismatch")
        # OpenCV BGR: red=valid background, magenta=road, black=ignore.
        colors = truth.reshape(-1, 3)
        allowed = ((colors == (0, 0, 0)).all(axis=1) |
                   (colors == (0, 0, 255)).all(axis=1) |
                   (colors == (255, 0, 255)).all(axis=1))
        if not allowed.all():
            raise ValueError("Unknown KITTI ground-truth color")
        valid = truth[:, :, 2] == 255
        positive = truth[:, :, 0] == 255
        predicted = prediction == target_id
        counts += [np.count_nonzero(valid & predicted & positive),
                   np.count_nonzero(valid & predicted & ~positive),
                   np.count_nonzero(valid & ~predicted & positive),
                   np.count_nonzero(valid & ~predicted & ~positive)]
    result = BinaryScore(*(int(count) for count in counts))
    if result.evaluated_pixels == 0:
        raise ValueError("No valid pixels to evaluate")
    return result
