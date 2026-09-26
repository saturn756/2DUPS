"""I7 SegmentationMap: semantic map with source-frame alignment."""
from __future__ import annotations

from dataclasses import dataclass

from .base import Ref, Status
from .i2_calibrated_frame import ImageSize


@dataclass
class SegmentationMap:
    """类别语义必须经类别模式映射，禁止下游按编号猜语义。"""

    INTERFACE = "I7"

    sequence_id: str
    frame_id: str
    variant_id: str
    coordinate_space: str
    source_size: ImageSize
    map_size: ImageSize
    class_map_ref: Ref | None
    class_schema_ref: str
    alignment_ref: Ref | None
    status: Status
    producer_ref: str
    config_ref: str
    segmentation_missing: bool = False
    reason: str | None = None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.status is Status.OK and (self.class_map_ref is None or self.segmentation_missing):
            raise ValueError("OK segmentation requires a present class map")
        if self.map_size != self.source_size and self.alignment_ref is None:
            raise ValueError("Resized segmentation requires an alignment_ref")
