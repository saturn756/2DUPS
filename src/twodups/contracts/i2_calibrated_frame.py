"""I2 CalibratedFrame（边 E02 必需、E11 可选）。"""
from __future__ import annotations

from dataclasses import dataclass

from .base import CalibrationState, Ref


@dataclass
class CalibratedFrame:
    """coordinate_space 必填；标定状态变化必须逐帧反映，不允许沿用过期状态。"""

    INTERFACE = "I2"

    frame_id: str
    media_ref: Ref
    coordinate_space: str                      # pixel / rectified / reference_plane
    calibration: CalibrationState
    camera_id: str | None = None
    timestamp: float | None = None
    manifest_ref: str | None = None
