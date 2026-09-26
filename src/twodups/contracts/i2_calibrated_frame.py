"""I2 CalibratedFrame: source-frame metadata and calibration state."""
from __future__ import annotations

from dataclasses import dataclass

from .base import CalibrationState, Ref


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")


@dataclass
class CalibratedFrame:
    """coordinate_space 必填；标定状态变化必须逐帧反映，不允许沿用过期状态。"""

    INTERFACE = "I2"

    sequence_id: str
    frame_id: str
    frame_index: int
    image_size: ImageSize
    media_ref: Ref
    coordinate_space: str                      # pixel / rectified / reference_plane
    calibration: CalibrationState
    camera_id: str | None = None
    timestamp: float | None = None
    source_timestamp: float | None = None
    rotation_degrees_clockwise: int = 0
    manifest_ref: str | None = None
    variant_id: str = "raw"
    schema_version: str = "1.0"
    producer_ref: str = "m1.bdd100k_video_loader"
    config_ref: str = "configs/m1_calibration.yaml"
