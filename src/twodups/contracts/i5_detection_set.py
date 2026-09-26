"""I5 DetectionSet: current-frame detection output."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from .base import BBox, Status
from .i2_calibrated_frame import ImageSize


@dataclass
class Detection:
    detection_id: str
    class_ref: str                             # 必须能在类别模式中解析
    bbox: BBox
    score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Detection score must be within [0, 1]")


@dataclass
class DetectionSet:
    """检测为空也要给出空集合与状态，区分“确实没有目标”与“检测失败”。"""

    INTERFACE = "I5"

    sequence_id: str
    frame_id: str
    variant_id: str
    coordinate_space: str
    image_size: ImageSize
    status: Status
    class_schema_ref: str
    producer_ref: str
    config_ref: str
    objects: list[Detection] = field(default_factory=list)
    detection_missing: bool = False
    reason: str | None = None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.status is Status.OK and self.detection_missing:
            raise ValueError("OK detection cannot be marked missing")
        if self.status is not Status.OK and self.objects:
            raise ValueError("Failed detection cannot masquerade as current objects")
