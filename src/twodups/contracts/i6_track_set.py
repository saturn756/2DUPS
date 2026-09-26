"""I6 TrackSet: cross-frame identities on the current image plane."""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import BBox, Ref, Status, TrackState


@dataclass
class Track:
    track_id: str
    class_ref: str
    state: TrackState
    current_bbox: BBox | None = None
    confidence: float | None = None
    age: int = 0
    missed_frames: int = 0
    last_detection_id: str | None = None
    history_ref: Ref | None = None

    def __post_init__(self) -> None:
        if self.state is TrackState.LOST and self.current_bbox is not None:
            raise ValueError("Lost track cannot present a stale bbox as current")


@dataclass
class Association:
    method: str
    motion_compensation_used: bool
    match_result_ref: Ref | None = None
    reason: str | None = None


@dataclass
class TrackSet:
    """unreliable 轨迹不得单独作为冲突/相邻关系判据；是否启用几何补偿要可读出。"""

    INTERFACE = "I6"

    sequence_id: str
    frame_id: str
    variant_id: str
    coordinate_space: str
    status: Status
    association: Association
    producer_ref: str
    config_ref: str
    tracks: list[Track] = field(default_factory=list)
    detection_missing: bool = False
    reason: str | None = None
    schema_version: str = "1.0"
