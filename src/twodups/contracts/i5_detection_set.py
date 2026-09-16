"""I5 DetectionSet（边 E07，模块内部必需，对外可读）。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import Producer


@dataclass
class Detection:
    object_id: str
    class_ref: str                             # 必须能在类别模式中解析
    bbox: tuple[float, float, float, float]    # x1, y1, x2, y2
    score: float


@dataclass
class DetectionSet:
    """检测为空也要给出空集合与状态，区分“确实没有目标”与“检测失败”。"""

    INTERFACE = "I5"

    frame_id: str
    variant_id: str
    coordinate_space: str
    producer: Producer
    objects: list[Detection] = field(default_factory=list)
    detection_missing: bool = False
