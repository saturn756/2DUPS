"""I6 TrackSet（边 E08 必需）。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import Ref, TrackState


@dataclass
class Track:
    track_id: str
    class_ref: str
    state: TrackState
    frames: list[str] = field(default_factory=list)
    confidence: float | None = None


@dataclass
class TrackSet:
    """unreliable 轨迹不得单独作为冲突/相邻关系判据；是否启用几何补偿要可读出。"""

    INTERFACE = "I6"

    frame_id: str
    tracks: list[Track] = field(default_factory=list)
    detection_missing: bool = False
    association_ref: Ref | None = None         # 是否使用相机运动补偿
