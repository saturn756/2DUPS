"""I7 SegmentationMap（边 E09 必需）。"""
from __future__ import annotations

from dataclasses import dataclass

from .base import Producer, Ref


@dataclass
class SegmentationMap:
    """类别语义必须经类别模式映射，禁止下游按编号猜语义。"""

    INTERFACE = "I7"

    frame_id: str
    variant_id: str
    class_map_ref: Ref
    class_schema_ref: str
    producer: Producer
    coordinate_space: str | None = None
    segmentation_missing: bool = False
