"""I4 MatchResult: optional image-plane evidence for the current frame pair."""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import Ref, Status


@dataclass
class FramePair:
    frame_a: str
    frame_b: str
    variant_id: str
    coordinate_space: str

    def __post_init__(self) -> None:
        if not self.frame_a or not self.frame_b or self.frame_a == self.frame_b:
            raise ValueError("I4 requires two distinct frame IDs")
        if not self.variant_id or not self.coordinate_space:
            raise ValueError("I4 pair lacks variant or coordinate space")


@dataclass
class Transform:
    type: str
    semantics: str = "image_plane_transform"
    matrix: list[list[float]] = field(default_factory=list)
    inlier_ratio: float | None = None
    residual: float | None = None

    def __post_init__(self) -> None:
        if self.semantics != "image_plane_transform":
            raise ValueError("I4 cannot claim metric motion")


@dataclass
class MatchResult:
    """status 非 ok 时 transform 不得作为几何证据；结果只能用于 frame_b 及其之后的帧。"""

    INTERFACE = "I4"

    sequence_id: str
    pair: FramePair
    status: Status
    producer_ref: str
    config_ref: str
    keypoints_ref: Ref | None = None
    matches_ref: Ref | None = None
    inlier_mask_ref: Ref | None = None
    transform: Transform | None = None
    reason: str | None = None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.status is Status.OK and self.transform is None:
            raise ValueError("OK I4 result requires a transform")
        if self.status is not Status.OK and self.transform is not None:
            raise ValueError("Non-ok I4 result cannot carry a usable transform")
